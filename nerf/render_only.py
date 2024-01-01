import torch
import numpy as np

from train import *
from loader import *
from render import *


if __name__ == "__main__":
    data = np.load("tiny_nerf_data.npz")
    focal = data["focal"]
    images = data['images']
    (num_images, H, W, _) = images.shape

    num_pos = H * W * NUM_SAMPLES
    device = torch.device('cpu')

    Module = TrainNeRF(8, num_pos, device)
    Module.load_param()

    render_video(Module.module, H, W, focal, 1)
