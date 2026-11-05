from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import save_image
import pytorch_lightning as pl
import os
import torch

class RecordingAndVisualizationCallback(pl.Callback):
    def __init__(self, saving_dir:str):
        super().__init__()
        self.saving_dir = saving_dir
        os.makedirs(self.saving_dir, exist_ok=True)


    @torch.no_grad()
    def on_test_batch_end(self, trainer, model, outputs, batch, batch_idx):
        img, sorting_idx, keys, src_number = batch
        B, S, C, H, W = img.shape
        S_src = src_number.unique().item()
        S_tar = S - S_src

        src = img[:, :S_src].float()
        estimated_depth = outputs["estimated_depth"].float().cpu().numpy()
        tar = outputs["target_images"].float()
        rendered_images = outputs["rendered_images"].float()
        rendered_depth = outputs["rendered_depth"].squeeze(1).float().cpu().numpy()

        gaussians = outputs["gaussians"]

        if model.cfg.gaussian_evaluation_stage.data.name == "dl3dv":
            keys = [key.split('/')[-2] for key in keys]

        for k,b in zip(keys, range(B)):
            vis_dir = os.path.join(self.saving_dir, "visualization", f"scene-{k}")
            os.makedirs(vis_dir, exist_ok=True)
            str_length = len(str(S_src-1))

            for s_src in range(S_src):
                group_dir = os.path.join(vis_dir, f"src-{str(s_src).zfill(str_length)}")
                os.makedirs(group_dir, exist_ok=True)
                _save_src_images(
                    group_dir,
                    src[b, s_src],
                    estimated_depth[b, s_src],   
                )
            str_length = len(str(S_tar-1))
            for s_tar in range(S_tar):
                group_dir = os.path.join(vis_dir, f"tar-{str(s_tar).zfill(str_length)}")
                os.makedirs(group_dir, exist_ok=True)
                tar_idx = b*S_tar + s_tar
                _save_tar_images(
                    group_dir, 
                    tar[tar_idx],
                    rendered_images[tar_idx], 
                    rendered_depth[tar_idx],           
                )

    def on_test_end(self, trainer, model):
        avg_psnr, avg_ssim, avg_lpips = model.test_metrics.get_avg_metrics()
        with open(os.path.join(self.saving_dir, "results.txt"), 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")

@torch.no_grad()
def _save_src_images(group_dir, img, estimated_depth):  
    im_name = os.path.join(group_dir, "View.png")
    save_image(img.clip(0,1), im_name)

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(estimated_depth)
    fig.colorbar(im)
    im_name = os.path.join(group_dir, "EstimatedDepth.png")
    plt.savefig(im_name)
    plt.close()

    plt.imsave(os.path.join(group_dir, "EstimatedDepth1.png"), estimated_depth, vmin=0, vmax=2)

@torch.no_grad()
def _save_tar_images(group_dir, img, rendered_image, rendered_depth):  
    im_name = os.path.join(group_dir, "View.png")
    save_image(img.clip(0,1), im_name)
    
    im_name = os.path.join(group_dir, "RenderedImage.png")
    save_image(rendered_image.clip(0,1), im_name)

    l1_errormap = torch.abs(rendered_image - img).mean(dim=0)
    im_name = os.path.join(group_dir, "L1ErrorMap.png")
    plt.imsave(im_name, l1_errormap.cpu().numpy(), cmap='inferno', vmin=0, vmax=1)

    mse_map = ((rendered_image - img)**2).mean(dim=0)
    im_name = os.path.join(group_dir, "MSEMap.png")
    plt.imsave(im_name, mse_map.cpu().numpy(), cmap='inferno', vmin=0, vmax=1)

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(rendered_depth, cmap='inferno')
    fig.colorbar(im)
    im_name = os.path.join(group_dir, "RenderedDepth.png")
    plt.savefig(im_name)
    plt.close()

    vmax = np.percentile(rendered_depth, 70)

    plt.imsave(os.path.join(group_dir, "RenderedDepth1.png"), rendered_depth, vmin=0, vmax=vmax, cmap='inferno')

@torch.no_grad()
def save_as_splat(path, xyz, scale, rotation, opacity, color):    
    N = xyz.shape[0]
    print(f"Saving {N} Gaussians to .splat...")

    rgb = color
    
    rgba = np.concatenate([rgb, opacity], axis=1)
    rgba = np.clip(rgba, 0, 1) * 255
    rgba = rgba.astype(np.uint8)

    rot_uint8 = (rotation * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),       # 12 bytes
        ('sx', 'f4'), ('sy', 'f4'), ('sz', 'f4'),    # 12 bytes
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'), # 4 bytes
        ('rot_0', 'u1'), ('rot_1', 'u1'), ('rot_2', 'u1'), ('rot_3', 'u1') # 4 bytes
    ]
    
    buffer = np.zeros(N, dtype=dtype)
    
    buffer['x'] = xyz[:, 0]
    buffer['y'] = xyz[:, 1]
    buffer['z'] = xyz[:, 2]
    
    buffer['sx'] = scale[:, 0]
    buffer['sy'] = scale[:, 1]
    buffer['sz'] = scale[:, 2]
    
    buffer['r'] = rgba[:, 0]
    buffer['g'] = rgba[:, 1]
    buffer['b'] = rgba[:, 2]
    buffer['a'] = rgba[:, 3]
    
    buffer['rot_0'] = rot_uint8[:, 0]
    buffer['rot_1'] = rot_uint8[:, 1]
    buffer['rot_2'] = rot_uint8[:, 2]
    buffer['rot_3'] = rot_uint8[:, 3]

    buffer.tofile(path)
    print(f"Saving Success: {path} ({os.path.getsize(path) / 1024 / 1024:.2f} MB)")
