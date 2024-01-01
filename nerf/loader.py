import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from util import *

NUM_SAMPLES = 32


class MyDataset(Dataset):
    def __init__(self, images, poses, focal) -> None:
        super(MyDataset, self).__init__()
        self.images = torch.tensor(images)
        self.im_shape = images.shape
        (self.num_images, self.H, self.W, _) = images.shape
        (self.poses, self.focal) = (poses, focal)
        self._process()

    def _process(self):
        self.rays_flat, self.t_vals = [], []
        progress_bar = tqdm(total=self.poses.shape[0])
        for i in range(self.poses.shape[0]):
            pose = self.poses[i]
            (ray_origin, ray_direction) = get_rays(self.H, self.W, self.focal, pose)
            (rays_flat, t_vals) = render_flat_rays(
                ray_origins=ray_origin,
                ray_directions=ray_direction,
                near=2.0,
                far=6.0,
                num_samples=NUM_SAMPLES,
                rand=True
            )
            self.rays_flat.append(rays_flat)
            self.t_vals.append(t_vals)
            progress_bar.update(1)
        progress_bar.close()

    def __getitem__(self, index):
        return (self.images[index], self.rays_flat[index], self.t_vals[index])

    def __len__(self):
        return self.num_images


# test code
if __name__ == "__main__":
    data = np.load("tiny_nerf_data.npz")
    images = data['images']
    im_shape = images.shape
    (num_images, H, W, _) = images.shape
    (poses, focal) = (data["poses"], data["focal"])

    split_index = int(num_images * 0.8)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_poses = poses[:split_index]
    val_poses = poses[split_index:]

    # train_set = MyDataset(train_images, train_poses, focal)
    # val_set = MyDataset(val_images, val_poses, focal)

    print(type(images))
    plt.imshow(images[np.random.randint(low=0, high=num_images)])
    plt.show()
