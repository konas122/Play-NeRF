import torch
import numpy as np

from ray_utils import get_rays, get_ray_directions


BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])

def hash(coords, log2_hashmap_size):
    """
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    # [batch_size, 8]
    xor_result = torch.zeros_like(coords)[..., 0]

    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    """
    bounding_box: min and max x,y,z coordinates of object bbox

    Among the three-dimensional coordinates of the near and far points on all rays in the space,
    find the corresponding minimum and maximum values of xyz. 
    The smallest xyz set is used as the minimum boundary. 
    The largest xyz set is used as the maximum boundary.
    """
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # ray direction in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    for frame in camera_transforms['frames']:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)

        def find_min_max(pt):
            for i in range(3):
                if (min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if (max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        # four angles in the image
        for i in [0, W - 1, H * W - W, H * W - 1]:
            min_point = rays_o[i] + near * rays_d[i]    # near point in the ray
            max_point = rays_o[i] + far * rays_d[i]     # far point in the ray
            # update min_bound
            find_min_max(min_point)
            # update max_bound
            find_min_max(max_point)

    return (
        torch.tensor(min_bound) - torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor(max_bound) + torch.tensor([1.0, 1.0, 1.0])
    )


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """
    xyz:            3D coordinates of samples. B x 3
    bounding_box:   min and max x,y,z coordinates of object bbox
    resolution:     number of voxels per axis
    """
    box_min, box_max = bounding_box[0].to(xyz.device), bounding_box[1].to(xyz.device)

    # check whether some points are outside bounding box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):

        # pdb.set_trace()   # used for debug

        # Some points are outside bounding box. Clipping them!
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    # get grid index
    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()

    # the coordinate of the voxel corresponding to the "bottom_left_idx"(grid index)
    voxel_min_vertex = bottom_left_idx * grid_size + box_min

    # voxel_min_vertex + 1 voxel
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]).to(grid_size.device) * grid_size

    # First get 8 surrounding points corresponding to the current point
    # Then use these 8 points to get their indexes
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS.to(bottom_left_idx.device)

    # get hash value
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    # first 2 are coordinates, the last one is the index
    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices
