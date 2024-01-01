import os
import time
import pickle
import imageio
import numpy as np
from tqdm import tqdm

from render import *
from nerf_helpers import to8b


def render_path(render_poses, hwf, K, chunk, render_kwargs, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        rgb,  _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
    
    print(time.time() - t)

    rgbs = np.stack(rgbs, 0)

    return rgbs
