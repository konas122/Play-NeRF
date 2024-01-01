# used for training
import torch
import torch.nn.functional as F


POS_ENCODE_DIMS = 16


def encode_position(x):
    position = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [torch.sin, torch.cos]:
            position.append(fn(2.0 ** i * x))
    return torch.cat(position, axis=-1)


def get_rays(height, width, focal, pose):
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = torch.stack([transformed_i, -transformed_j, -torch.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_matrix = torch.tensor(camera_matrix)
    camera_dirs = transformed_dirs * camera_matrix

    ray_directions = torch.sum(camera_dirs, dim=-1)
    ray_origins = torch.broadcast_to(torch.tensor(height_width_focal), ray_directions.shape)

    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = torch.linspace(near, far, num_samples)

    if rand:
        # Inject uniform noise into sample space to make the sampling
        # continuous.
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = torch.rand(size=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )

    rays_flat = torch.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def render_rgb(model, rays_flat, t_vals, batch_size=8, height=100, width=100,
               num_samples=32, rand=True, train=True):
    if train:
        model.train()
        predictions = model(rays_flat)
    else:
        model.eval()
        with torch.no_grad():
            predictions = model(rays_flat)
    predictions = torch.reshape(predictions, shape=(batch_size, height, width, num_samples, 4))

    # Slice the predictions into rgb and sigma.
    rgb = F.sigmoid(predictions[..., :-1])
    sigma_a = F.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]

    # delta shape == (num_samples)
    if rand:
        delta = torch.cat([delta, torch.broadcast_to(torch.tensor([1e10]).to(delta.device), size=(batch_size, height, width, 1))], axis=-1)
        alpha = 1.0 - torch.exp(-sigma_a * delta)
    else:
        delta = torch.cat([delta, torch.broadcast_to(torch.tensor([1e10]).to(delta.device), size=(batch_size, 1))], axis=-1)
        alpha = 1.0 - torch.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = torch.cumprod(exp_term + epsilon, dim=-1)
    weights = alpha * transmittance
    rgb = torch.sum(weights[..., None] * rgb, dim=-2)

    return rgb
