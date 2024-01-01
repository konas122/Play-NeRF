import numpy as np
import multiprocessing
from torch.utils.data import DataLoader

from train import *
from loader import *
from render import *

K = 0.1


def main():
    data = np.load("tiny_nerf_data.npz")
    images = data['images']
    (num_images, H, W, _) = images.shape
    (poses, focal) = (data["poses"], data["focal"])

    split_index = int(num_images * K)

    train_images = images[:split_index]
    # val_images = images[split_index:]

    train_poses = poses[:split_index]
    # val_poses = poses[split_index:]

    train_set = MyDataset(train_images, train_poses, focal)
    # val_set = MyDataset(val_images, val_poses, focal)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        prefetch_factor, batch_size = 4, 8
    else:
        try:
            import torch_directmla
            prefetch_factor, batch_size = 1, 1
            device = torch_directml.device()
        except ImportError:
            device = torch.device('cpu')
            prefetch_factor, batch_size = 1, 1

    num_works = multiprocessing.cpu_count()

    train_ = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_works, prefetch_factor=prefetch_factor)

    num_pos = H * W * NUM_SAMPLES

    Module = TrainNeRF(8, num_pos, device)
    Module.train(train_, batch_size, num_epoch=1)

    render_video(Module.module, H, W, focal, batch_size)

    Module.save_param()


if __name__ == "__main__":
    main()
