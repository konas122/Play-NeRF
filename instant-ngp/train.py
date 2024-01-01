import os
import time
import torch
import imageio
import numpy as np
from tqdm import trange

from render import render
from model import NeRF_NGP
from optimizer import RAdam
from loader import load_data
from inference import render_path
from loss import total_variation_loss
from nerf_helpers import get_embedder, get_rays, to8b


img2mse = lambda x, y: torch.mean((x - y) ** 2)

mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    input_dirs = viewdirs[:, None].expand(inputs.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    # Hash Table Encoder
    embed_fn, input_ch = get_embedder(args, i=1)
    embedding_params = list(embed_fn.parameters())

    input_ch_views = 0

    # If use_viewdirs
    # common position embedder
    embeddirs_fn, input_ch_views = get_embedder(args, i=0)

    model = NeRF_NGP(num_layers=2,
                     hidden_dim=64,
                     geo_feat_dim=15,
                     num_layers_color=3,
                     hidden_dim_color=64,
                     input_ch=input_ch,
                     input_ch_views=input_ch_views).to(args['device'])
    print(model)
    grad_vars = list(model.parameters())

    model_fine = NeRF_NGP(num_layers=2,
                          hidden_dim=64,
                          geo_feat_dim=15,
                          num_layers_color=3,
                          hidden_dim_color=64,
                          input_ch=input_ch,
                          input_ch_views=input_ch_views).to(args['device'])
    
    grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args['netchunk'])
    optimizer = RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args['lr'], betas=(0.9, 0.99))

    render_kwargs = {
        'network_query_fn': network_query_fn,
        'N_importance': args['N_importance'],
        'network_fine': model_fine,
        'N_samples': args['N_samples'],
        'network_fn': model,
        'embed_fn': embed_fn,
    }

    return render_kwargs, optimizer


def train(args):
    device = args['device']
    images, poses, render_poses, hwf, i_split, bounding_box = load_data(args['basedir'], True, 8)
    args['bounding_box'] = bounding_box
    i_train, _, _ = i_split

    # white background
    images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    # Cast intrinsics to right types 
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # Create nerf model
    render_kwargs, optimizer = create_nerf(args)
    bds_dict = {
        'near': 2,
        'far': 6,
    }
    render_kwargs.update(bds_dict)

    poses = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)

    N_rand = args['N_rand']
    N_iters = 5000 + 1

    time0 = time.time()
    for i in trange(1, N_iters):
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]

        # N_rand
        rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))

        if i < 500:                 # precrop_iters=500
            dH = int(H // 2 * 0.5)  # precrop_frac=0.5
            dW = int(W // 2 * 0.5)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                ), -1)
        else:
            # (H, W, 2)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)

        coords = torch.reshape(coords, [-1, 2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]   # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]   # (N_rand, 3)

        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 3)

        rgb, extras = render(H, W, K, chunk=args['chunk'], rays=batch_rays, **render_kwargs)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss = loss + img_loss0
        # psnr0 = mse2psnr(img_loss0)

        sparse_loss_weight = 1e-10
        sparsity_loss = sparse_loss_weight * (extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
        loss = loss + sparsity_loss

        # add Total Variation loss
        n_levels = render_kwargs["embed_fn"].n_levels
        min_res = render_kwargs["embed_fn"].base_resolution
        max_res = render_kwargs["embed_fn"].finest_resolution
        log2_hashmap_size = render_kwargs["embed_fn"].log2_hashmap_size
        TV_loss = sum(total_variation_loss(render_kwargs["embed_fn"].embeddings[i], \
                                           device, min_res, max_res,    \
                                           i, log2_hashmap_size,        \
                                           n_levels=n_levels) for i in range(n_levels))
        tv_loss_weight = 1e-06
        loss = loss + tv_loss_weight * TV_loss
        if i > 1000:
            tv_loss_weight = 0.0
        
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f'Iter: {i}\n\tLoss: {loss.item()}\tPSNR: {psnr.item()}\n\ttime: {time.time() - time0}')

    now = int(time.time())

    print(f'\nSaving ...\n')
    path = os.path.join(args['savedir'], '{:06d}.tar'.format(now))
    torch.save({
        'network_fn_state_dict': render_kwargs['network_fn'].state_dict(),
        'network_fine_state_dict': render_kwargs['network_fine'].state_dict(),
        'embed_fn_state_dict': render_kwargs['embed_fn'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

    print(f'\nRendering ...\n')
    with torch.no_grad():
        rgbs = render_path(render_poses, hwf, K, 4096, render_kwargs)
    print('Done, saving', rgbs.shape)
    moviebase = os.path.join(args['savedir'], '_spiral_{:06d}_'.format(now))
    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
