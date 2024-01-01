# used for rendering video...
import torch
import imageio
import numpy as np
from tqdm import tqdm

from util import *
from loader import *


def get_translation_t(t):
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix)


def get_rotation_phi(phi):
    matrix = [
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix)


def get_rotation_theta(theta):
    matrix = [
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix)


def pose_spherical(theta, phi, t):
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w.numpy()
    return c2w


def render_video(model, height, width, focal, batch_size, filename="rgb_video.mp4"):
    batch_t = []
    batch_flat = []
    rgb_frames = []

    for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
        c2w = pose_spherical(torch.tensor(theta, dtype=torch.float32), torch.tensor(-30.0), torch.tensor(4.0))

        ray_origin, ray_dir = get_rays(height, width, focal, c2w)
        ray_flat, t_vals = render_flat_rays(
            ray_origin, ray_dir, 2.0, 6.0, NUM_SAMPLES
        )

        if index % batch_size == 0 and index > 0:
            batched_flat = torch.stack(batch_flat, dim=0).float()
            batch_flat = [ray_flat]

            batched_t = torch.stack(batch_t, dim=0).float()
            batch_t = [t_vals]

            rgb = render_rgb(
                model, batched_flat, batched_t, batch_size, height, width,
                NUM_SAMPLES, rand=False, train=False
            )

            temp_rgb = [np.clip(255 * img.numpy(), 0.0, 255.0).astype(np.uint8) for img in rgb]

            rgb_frames = rgb_frames + temp_rgb
        else:
            batch_flat.append(ray_flat)
            batch_t.append(t_vals)
    
    imageio.mimwrite(filename, rgb_frames, fps=30, quality=7, macro_block_size=None)
