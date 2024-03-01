import torch.nn.functional as F

from nerf_helpers import *


def render(H, W, K, chunk=1024 * 32,
           rays=None, c2w=None,
           near=0., far=1.,
           frame_time=None,
           **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape

    # create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    frame_time = frame_time * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far, frame_time], -1)

    rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
    k_extract = ['rgb']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                N_importance=0,
                network_fine=None):

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:]
    bounds = torch.reshape(ray_batch[..., 6:9], [-1, 1, 3])
    near, far, frame_time = bounds[..., 0], bounds[..., 1], bounds[...,2]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(far.device)

    z_vals = near * (1. - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape).to(lower.device)

    z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    raw, position_delta_0 = network_query_fn(pts, viewdirs, frame_time, network_fn)
    rgb_map0, weights = raw2outputs(raw, z_vals, rays_d)

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance)
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    # [N_rays, N_samples + N_importance, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    run_fn = network_fn if network_fine is None else network_fine

    raw, position_delta = network_query_fn(pts, viewdirs, frame_time, run_fn)
    rgb_map, _ = raw2outputs(raw, z_vals, rays_d)

    ret = {
        "rgb0": rgb_map0,
        "rgb": rgb_map,
        'position_delta_0': position_delta_0,
        'position_delta': position_delta,
    }
    return ret


def raw2outputs(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(z_vals.device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[..., 3], dists)

    exp_term = 1.0 - alpha + 1e-10
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(exp_term.device), exp_term], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, weights
