import torch
import pytorch_lightning as pl
from deepspeed.accelerator import get_accelerator
from einops import rearrange 
from torch.nn.functional import normalize
from torch import nn
from vggt.models.vggt import VGGT
from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from loss.gaussian_loss import get_gaussian_train_loss
from model.gaussian_prediction import GroupGaussianPredictor
from model.utils import OPTIMIZER_DICT, SCHEDULER_DICT, dropout_gaussian
from gsplat import rasterization
from utils.metrices import TestMetrics
from utils.utils_2d import rescale_and_crop
from utils.utils_3d import depth_to_world_coords_points, pose_encoding_to_extri, pose_encoding_to_intri, rotation_6d_to_matrix
from utils.utils_quaternion import xyzw2wxzy, quaternion_multiply, quaternion_conjugate, ensure_positive_hemisphere_quaternion
from pytorch_lightning.utilities import rank_zero_only
from model.semantic_encoder import get_semantic_encoder

      
def build_gaussian_wrapper(module_dict, cfg):
    wrapper = GaussianWrapper(cfg)
    vggt = VGGT.from_pretrained(
        cfg.module.geo_encoder.path, 
    )

    wrapper.geo_encoder.load_state_dict(vggt.aggregator.state_dict())
    for param in wrapper.geo_encoder.parameters():
        param.requires_grad = False

    wrapper.geo_encoder.eval()

    wrapper.camera_decoder.load_state_dict(vggt.camera_head.state_dict())
    for param in wrapper.camera_decoder.parameters():
        param.requires_grad = False

    wrapper.camera_decoder.eval()

    if cfg.module.sem_encoder.type is not None:
        for param in wrapper.sem_encoder.parameters():
            param.requires_grad = False

        wrapper.sem_encoder.eval()

    if cfg.module.tex_encoder.enabled:
        for name,param in wrapper.gaussian_predictor.tex_encoder.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif "bias" in name:
                nn.init.zeros_(param)

    for i in range(5):
        for name,param in wrapper.gaussian_predictor.output_convs[i].named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif "bias" in name:
                nn.init.zeros_(param)

    if "gaussian_predictor" in module_dict:
        wrapper.gaussian_predictor.load_state_dict(module_dict["gaussian_predictor"])

    for param in wrapper.gaussian_predictor.parameters():
        param.requires_grad = True
    
    return wrapper


class GaussianWrapper(pl.LightningModule):
    def __init__(self, cfg, patch_size=14, embed_dim=1024):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        img_size = cfg.module.geo_encoder.img_size
        self.geo_encoder = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_decoder = CameraHead(dim_in=2 * embed_dim)
        
        self.gaussian_predictor = GroupGaussianPredictor(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            activations=cfg.module.gaussian_decoder.activations,
            features=cfg.module.gaussian_decoder.channels,
            pos_embed=cfg.module.gaussian_decoder.pos_embed,
            padding_mode=cfg.module.gaussian_decoder.padding_mode,
            gaussians_per_pixel=cfg.module.gaussian_decoder.gaussians_per_pixel,
            sem_encoder_type=cfg.module.sem_encoder.type,
            with_tex_encoder=cfg.module.tex_encoder.enabled,
        )

        self.sem_encoder = get_semantic_encoder(cfg.module.sem_encoder.type, cfg.module.sem_encoder.path, cfg.gaussian_training_stage.data.resize.new_size)
        self.gaussian_loss = get_gaussian_train_loss(cfg.gaussian_training_stage.loss)


    def forward(
        self,
        src: torch.Tensor,
        frame_chunk_size = 8
    ):  
        if len(src.shape) == 4:
            src = src.unsqueeze(0)    # B S C H W
        with torch.no_grad():
            _, _, _, H, W = src.shape
            geo_tokens_list, patch_start_idx = self.geo_encoder(src)

            camera = self.camera_decoder(geo_tokens_list)[-1]
            extrinsic = pose_encoding_to_extri(camera)
            intrinsic = pose_encoding_to_intri(camera, (H, W))
            cam_quat = self.get_cam_quat(camera)  # (B, S, 4)

            sem_feature_list = self.sem_encoder(src)[1:] if self.sem_encoder is not None else None

        if self.sem_encoder is not None:
            depth, opacity, color, scale, raw_rotation, rotation = self.gaussian_predictor(
                geo_tokens_list, images=src, patch_start_idx=patch_start_idx, frames_chunk_size=frame_chunk_size,
                sem_feature_list=sem_feature_list
            )

        depth = depth.squeeze(3)
        del geo_tokens_list
        
        coordinate, opacity, color, scale, rot_quat = self.output2gaussian(
            depth,
            opacity,
            scale,
            color,
            rotation,
            extrinsic,
            intrinsic,
            cam_quat,
            self.cfg.gs_setting.scale_modifier
        )
        
        predictions = {
            "images": src,
            "camera": camera,
            "coordinate": coordinate,
            "opacity": opacity,
            "color": color.squeeze(-2),
            "scale": scale,
            "raw_rotation": raw_rotation,
            "rotation": rot_quat,
            "depth": depth.mean(2),

            "camera_full":{
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "quat": cam_quat,
            }
        }

        return predictions

    def on_load_checkpoint(self, checkpoint):
        scheduler_state = self.lr_schedulers().state_dict()  # pyright: ignore[reportAttributeAccessIssue]
        scheduler_state["last_epoch"] = checkpoint["lr_schedulers"][0]["last_epoch"]
        checkpoint["lr_schedulers"][0] = scheduler_state
        super().on_load_checkpoint(checkpoint)


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        get_accelerator().empty_cache()

        img, sorting_idx, src_number = batch
        if self.cfg.gaussian_training_stage.deepspeed_config.bf16.enabled:
            converted_img = img.to(torch.bfloat16)
        else:
            converted_img = img.clone()

        src_number = src_number.unique().item()
        rendered_images, rendered_depthes, src_predictions = self.splat(converted_img, sorting_idx, src_number)
        
        with torch.autocast(enabled=self.cfg.gaussian_training_stage.deepspeed_config.bf16.enabled, dtype=torch.float32, device_type="cuda", cache_enabled=False):
            img = rearrange(img, "b s c h w -> (b s) c h w")
            loss = self.gaussian_loss(
                rendered_images,
                img,
                src_predictions["depth"].to(torch.float32), 
            )
            self.log('train_loss', loss, prog_bar=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

            if self.cfg.print_info:
                self.print_info({
                    "train step": batch_idx,
                    "train_loss": loss,
                    "lr": self.trainer.optimizers[0].param_groups[0]['lr']
                })

            return {
                "loss": loss,
                "rendered_images": rendered_images,
                "rendered_depth": rendered_depthes,
                "estimated_depth": src_predictions["depth"],
                "gaussians": {
                    "coordinate": src_predictions["coordinate"],
                    "opacity": src_predictions["opacity"],
                    "color": src_predictions["color"],
                    "scale": src_predictions["scale"],
                    "raw_rotation": src_predictions["raw_rotation"],
                },
            }
        
    @rank_zero_only
    def print_info(self, info_dict):
        print()
        for k, v in info_dict.items():
            print(f"{k}: {v}")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pass 
    
    @torch.no_grad()
    def on_test_start(self):
        super().on_test_start()
        self.test_metrics = TestMetrics(self.device)
        self.inf_time = []
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        img, sorting_idx, scene_dir, src_number = batch
        src_number = src_number.unique().item()
        rendered_images, rendered_depthes, src_predictions = self.splat(img, sorting_idx, src_number, including_src=False)
        
        B, S, C, H, W = img.shape
        tar = rearrange(img[:, src_number:].clamp(0.0, 1.0), "b s c h w -> (b s) c h w") 
        rendered_images = rendered_images.clamp(0.0, 1.0) 

        if hasattr(self.cfg.gaussian_evaluation_stage, "resolution") and self.cfg.gaussian_evaluation_stage.resolution is not None:
            rendered_images = rescale_and_crop(rendered_images, self.cfg.gaussian_evaluation_stage.resolution)
            tar = rescale_and_crop(tar, self.cfg.gaussian_evaluation_stage.resolution)
            
            pds = torch.zeros_like(rendered_depthes)
            rendered_depthes = torch.cat([rendered_depthes, pds, pds], dim=1)
            rendered_depthes = rescale_and_crop(rendered_depthes, self.cfg.gaussian_evaluation_stage.resolution)
            rendered_depthes = rendered_depthes[:,0]


        self.test_metrics.update_metrics(tar, rendered_images)
        print()
        print("step", batch_idx)
        print("scene_dir or scene_key", scene_dir)
        print("psnr", self.test_metrics.test_psnr_list[-1])
        print("ssim", self.test_metrics.test_ssim_list[-1])
        print("lpips", self.test_metrics.test_lpips_list[-1])

        return {
            "psnr": self.test_metrics.test_psnr_list[-1],
            "ssim": self.test_metrics.test_ssim_list[-1],
            "lpips": self.test_metrics.test_lpips_list[-1],
            "target_images": tar,
            "rendered_images": rendered_images,
            "rendered_depth": rendered_depthes,
            "estimated_depth": src_predictions["depth"],
            
            "gaussians": {
                "coordinate": src_predictions["coordinate"], # (B, N, 3)
                "opacity": src_predictions["opacity"], # (B, N)
                "color": src_predictions["color"], # (B, N, 3)
                "scale": src_predictions["scale"], # (B, N, 3)
                "raw_rotation": src_predictions["raw_rotation"], # (B, S, 4, H, W)
                "rotation": src_predictions["rotation"] # (B, N, 4)
            }
        }
    

    @torch.no_grad()
    def on_test_end(self):
        avg_psnr, avg_ssim, avg_lpips = self.test_metrics.get_avg_metrics()
        print()
        print("="*10 + "> Final Average Test Metrics <" + "="*10)
        print("psnr", avg_psnr)
        print("ssim", avg_ssim)
        print("lpips", avg_lpips)
        print("time", sum(self.inf_time[10:]) / len(self.inf_time[10:]))


    def configure_optimizers(self):
        optimizer_meta = self.cfg.gaussian_training_stage.optimizer
        OptimizerClass = OPTIMIZER_DICT[optimizer_meta.type] if optimizer_meta.type in OPTIMIZER_DICT else getattr(torch.optim, optimizer_meta.type)
        parameters = list(self.gaussian_predictor.parameters())
        optimizer = OptimizerClass(
            parameters,
            lr = optimizer_meta.params.lr,
            weight_decay = optimizer_meta.params.weight_decay
            # **optimizer_meta.params
        )

        scheduler_meta = self.cfg.gaussian_training_stage.scheduler
        scheduler_class = SCHEDULER_DICT[scheduler_meta.type] if scheduler_meta.type in SCHEDULER_DICT else getattr(torch.optim.lr_scheduler, scheduler_meta.type)
        scheduler = scheduler_class(
            optimizer,
            **scheduler_meta.params
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
        
    
    def splat(self, img, sorting_idx, src_number, including_src=True):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        src = img[:, :src_number]  # (B, S_src, C, H, W)
        B, S, _, H, W = img.shape

        if not self.training:
            start_event.record() # pyright: ignore[reportCallIssue]
        src_predictions = self(src)
        if not self.training:
            end_event.record() # pyright: ignore[reportCallIssue]
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            self.inf_time.append(elapsed_time_ms)


        src_cam = src_predictions["camera"]  # (B, S_src, 9)
        tar_cam = self.get_tar_camera(img, sorting_idx, src_cam)  # (B, S_tar, 9)
        camera = torch.cat([src_cam, tar_cam], 1)  # (B, S, 9)

        extrinsic, intrinsic = self.get_splatting_camera_parameters(camera, (H, W))

        s_start = 0 if including_src else src_number
        s_chunk = 12

        rendered_results = []
        
        for k in ("coordinate", "rotation", "scale", "opacity", "color"):
            src_predictions[k] = src_predictions[k].to(torch.float32)
        extrinsic = extrinsic.to(torch.float32)  
        intrinsic = intrinsic.to(torch.float32)
        for b in range(B):
            if self.training and hasattr(self.cfg.gs_setting, "dropout"):
                dropout = self.cfg.gaussian_training_stage.gaussian_dropout
                means, quats, scales, opacities, colors = dropout_gaussian(
                    [
                        src_predictions["coordinate"][b],
                        src_predictions["rotation"][b],
                        src_predictions["scale"][b],
                        src_predictions["opacity"][b],
                        src_predictions["color"][b],
                    ],
                    p=dropout,
                )
            else:
                means = src_predictions["coordinate"][b]
                quats = src_predictions["rotation"][b]
                scales = src_predictions["scale"][b]
                opacities = src_predictions["opacity"][b]
                colors = src_predictions["color"][b]

            group_ext = extrinsic[b]
            group_int = intrinsic[b]

            if (not self.training) and self.cfg.gaussian_evaluation_stage.pose_post_opt.enabled:
                group_ext = self.pose_post_opt(
                    means, quats, scales, opacities, colors, group_ext, group_int, img[b], H, W, s_start,S, s_chunk
                )

            rendered_results.extend(
                self._render(
                    means, quats, scales, opacities, colors, group_ext, group_int, img.device, H, W, s_start,S, s_chunk
                )
            )
        rendered_results = torch.cat(rendered_results, 0)
        rendered_results = rearrange(rendered_results, "n h w c -> n c h w")
        rendered_images, rendered_depthes = torch.split(rendered_results, [3, 1], dim=1)
        
        return rendered_images, rendered_depthes, src_predictions

    def _render(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        group_ext,
        group_int,
        device,
        H, W,
        s_start,
        S,
        s_chunk
    ):  
        results = []
        for s in range(s_start, S, s_chunk):
            real_s_chunk = min(s_chunk, S - s)
            rendered_color, _, _ = rasterization(
                means = means,
                quats = quats,
                scales = scales,
                opacities = opacities,
                colors = colors,
                viewmats = group_ext[s:s+s_chunk],
                Ks = group_int[s:s+s_chunk],
                width = W,
                height = H,
                render_mode = "RGB+D",
                absgrad=self.cfg.gs_setting.absgrad,
                rasterize_mode=self.cfg.gs_setting.rasterize_mode,
                packed=False,
                backgrounds = torch.ones(real_s_chunk, 3, dtype=torch.float32, device=device) if self.cfg.gs_setting.white_bg else torch.zeros(real_s_chunk, 3, dtype=torch.float32, device=device),
            )
            results.append(rendered_color)
        return results

    @torch.inference_mode(False)
    @torch.enable_grad()
    def pose_post_opt(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        group_ext,
        group_int,
        img,
        H, W,
        s_start,
        S,
        s_chunk
    ):
        if not hasattr(self, "pose_post_opt_loss"):
            self.pose_post_opt_loss = get_gaussian_train_loss(self.cfg.gaussian_evaluation_stage.pose_post_opt.loss)
            self.pose_post_opt_loss.to(self.device)
        S_tar = S - s_start
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=group_ext.device))
        cam_rot_delta = nn.Parameter(torch.zeros([S_tar, 6], requires_grad=True, device=group_ext.device))
        cam_trans_delta = nn.Parameter(torch.zeros([S_tar, 3], requires_grad=True, device=group_ext.device))

        opt_params = [
            {
                "params": [cam_rot_delta],
                "lr": self.cfg.gaussian_evaluation_stage.pose_post_opt.lr_rot,
            },
            {
                "params": [cam_trans_delta],
                "lr": self.cfg.gaussian_evaluation_stage.pose_post_opt.lr_trans,
            }
        ]

        group_ext = group_ext.clone().detach().requires_grad_(True)
        group_int = group_int.clone().detach().requires_grad_(True)
        means = means.clone().detach().requires_grad_(True)
        quats = quats.clone().detach().requires_grad_(True)
        scales = scales.clone().detach().requires_grad_(True)
        opacities = opacities.clone().detach().requires_grad_(True)
        colors = colors.clone().detach().requires_grad_(True)

        pose_optimizer = torch.optim.Adam(opt_params)
        for _ in range(self.cfg.gaussian_evaluation_stage.pose_post_opt.iteration):
            pose_optimizer.zero_grad()
            rot = rotation_6d_to_matrix(
                cam_rot_delta + self.identity.expand(S_tar, -1)
            )  # (..., 3, 3)
            transform = torch.eye(4, device=group_ext.device, requires_grad=True).repeat((S_tar, 1, 1))
            transform[..., :3, :3] = rot
            transform[..., :3, 3] = cam_trans_delta
            new_tar_ext = torch.matmul(group_ext[s_start:], transform)
            new_ext = torch.cat([group_ext[:s_start], new_tar_ext], dim=0)
            
            rendered_images = self._render(
                means, 
                quats, 
                scales, 
                opacities, 
                colors, 
                new_ext, 
                group_int, 
                img.device, 
                H, W, s_start, S, s_chunk
            )

            
            rendered_images = torch.cat(rendered_images, 0)
            rendered_images = rearrange(rendered_images, "n h w c -> n c h w")
            rendered_images, rendered_depthes = torch.split(rendered_images, [3, 1], dim=1)
            tar = img[s_start:]
            tar = tar.clamp(0.0, 1.0)
            if hasattr(self.cfg.gaussian_evaluation_stage, "resolution") and self.cfg.gaussian_evaluation_stage.resolution is not None:
                rendered_images = rescale_and_crop(rendered_images, self.cfg.gaussian_evaluation_stage.resolution)
                tar = rescale_and_crop(tar, self.cfg.gaussian_evaluation_stage.resolution)
            
            loss = self.pose_post_opt_loss(
                rendered_images,
                tar,
                None
            )
            loss.backward()
            pose_optimizer.step()

        with torch.no_grad():
            rot = rotation_6d_to_matrix(
                cam_rot_delta + self.identity.expand(S_tar, -1)
            )
            transform = torch.eye(4, device=group_ext.device).repeat((S_tar, 1, 1))
            transform[..., :3, :3] = rot
            transform[..., :3, 3] = cam_trans_delta
            new_tar_ext = torch.matmul(group_ext[s_start:], transform)
            new_ext = torch.cat([group_ext[:s_start], new_tar_ext], dim=0)
            return new_ext
        
    # Camera Alignment Implementation.
    @torch.no_grad()
    def get_tar_camera(self, img:torch.Tensor, sorting_idx:torch.Tensor, src_cam1):
        """
        Get the target camera pose from the source and target images.
        Args:
            img (torch.Tensor): Input images of shape (B, S_src + S_tar, C, H, W).
            sorting_idx (torch.Tensor): Sorting index to reorder images.
            src_cam1 (torch.Tensor): Camera pose computed from only the source images.
        Returns:
            torch.Tensor: Target camera pose of shape (B, S_tar, 9).
        """
        
        B, S, C, H, W = img.shape
        _, S_src, _ = src_cam1.shape

        addition_index = torch.arange(0, B*S, S, dtype=torch.int64, device=sorting_idx.device).unsqueeze(-1)
        img = img.view((B*S), C, H, W)[
            (sorting_idx + addition_index).view(-1)
        ].view(B, S, C, H, W)

        geo_tokens_list, _ = self.geo_encoder(img)
        camera = self.camera_decoder(geo_tokens_list)[-1]
        del geo_tokens_list
        
        camera = camera.view((B*S), 9)[
            (sorting_idx.argsort() + addition_index).view(-1)
        ].view(B, S, 9)
        del addition_index

        # Align rotation of target camera
        cam_quat = self.get_cam_quat(camera)
        src_cam2_quat = cam_quat[:,:S_src]  # (B S_src 4)  
        src_cam1_quat = self.get_cam_quat(src_cam1)  # (B S_src 4)
        q1q2c = quaternion_multiply(src_cam1_quat, quaternion_conjugate(src_cam2_quat))  # (B S_src 4)
        q1q2c = ensure_positive_hemisphere_quaternion(q1q2c)
        dq = normalize(q1q2c.sum(dim=1, keepdim=True), dim=-1)  # (B 1 4)
        tar_cam_quat = quaternion_multiply(dq, cam_quat[:,S_src:])  # (B S_tar 4)

        # Align translation of target camera
        cam_t = camera[..., :3]
        src_cam_t2 = cam_t[:,:S_src]   # (B S_src 3)
        src_cam_t1 = src_cam1[..., :3]  # (B S_src 3)
        t2t2_sum = torch.einsum("bsc,bsc->b", src_cam_t2, src_cam_t2)  # (B)
        t2t1_sum = torch.einsum("bsc,bsc->b", src_cam_t2, src_cam_t1)  # (B)
        t2_sum = torch.einsum("bsc->bc", src_cam_t2)  # (B 3)
        t1_sum = torch.einsum("bsc->bc", src_cam_t1)  # (B 3)
        t2_sum_t2_sum = torch.einsum("bc,bc->b", t2_sum, t2_sum)  # (B)
        t2_sum_t1_sum = torch.einsum("bc,bc->b", t2_sum, t1_sum)  # (B)
        s = (t2t1_sum - t2_sum_t1_sum/S_src) / (t2t2_sum - t2_sum_t2_sum/S_src)  # (B)
        s = s.unsqueeze(-1)  # (B 1)
        dt = (t1_sum - s * t2_sum)/S_src # (B 3)
        tar_cam_t = s.unsqueeze(1) * cam_t[:, S_src:] + dt.unsqueeze(1) # (B S_tar 3)


        tar_cam = torch.zeros_like(camera[:, S_src:])
        tar_cam[..., :3] = tar_cam_t
        tar_cam[..., 3:6] = tar_cam_quat[..., 1:] # XYZ
        tar_cam[..., 6] = tar_cam_quat[..., 0] # W
        tar_cam[..., 7:] = camera[:, S_src:, 7:] # FOV

        return tar_cam # tar_cam[..., 3:7] XYZW
    
    def output2gaussian(
        self,
        depth,
        opacity,
        scale,
        sh,
        rot_quat,
        extrinsic, 
        intrinsic,
        cam_quat,
        scale_modifier=1.0
    ):
        coordinate, _ = depth_to_world_coords_points(depth, extrinsic, intrinsic) # (B S N H W 3)
        coordinate = rearrange(coordinate, "b s g h w c -> b (s g h w) c")
        opacity = rearrange(opacity, "b s g 1 h w -> b (s g h w)")
        sh = rearrange(sh, "b s g (c1 c2) h w  -> b (s g h w) c1 c2", c2=3)
        scale = rearrange(scale, "b s g c h w -> b (s g h w) c") * scale_modifier
        rot_quat = rearrange(rot_quat, "b s g c h w -> b s (g h w) c")
        cam_quat = quaternion_conjugate(cam_quat)  # (B S 4), from w2c to c2w
        cam_quat = rearrange(cam_quat, "b s c -> b s 1 c")
        rot_quat = quaternion_multiply(cam_quat, rot_quat) # c2w
        rot_quat = rearrange(rot_quat, "b s n c -> b (s n) c")

        return coordinate, opacity, sh, scale, rot_quat


    def get_cam_quat(self, camera):
        """ Extract the camera quaternion from the camera pose encoding.
        Args:
            camera (torch.Tensor): Camera pose encoding of shape (B, S, 9).
        Returns:
            cam_quat (torch.Tensor): Camera quaternion of shape (B, S, 4).
        """
        cam_quat = normalize(camera[..., 3:7], dim=-1)  # Normalize the quaternion
        cam_quat = xyzw2wxzy(cam_quat)  # Convert from XYZW to WXYZ format
        return cam_quat 


    def get_splatting_camera_parameters(self, cameras:torch.Tensor, image_size_hw):
        extrinsic = pose_encoding_to_extri(cameras) # (B, S, 4, 4)
        intrinsic = pose_encoding_to_intri(cameras, image_size_hw) # (B, S, 3, 3)

        return extrinsic, intrinsic




