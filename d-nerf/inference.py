import time
import numpy as np
from tqdm import tqdm

from render import *
from nerf_helpers import to8b


def render_path(render_poses, render_times, hwf, chunk, render_kwargs, render_factor=0):
    H, W, focal = hwf
    rgbs = []

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    t = time.time()
    for i, (c2w, frame_time) in enumerate(zip(tqdm(render_poses), render_times)):
        rgb,  _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
    
    print(time.time() - t)
    rgbs = np.stack(rgbs, 0)

    return rgbs
