import os
import time
import torch
import imageio
import numpy as np
from tqdm import trange

from render import render
from loader import load_data
from inference import render_path
from model import DirectTemporalNeRF
from nerf_helpers import get_embedder, get_rays, to8b


img2mse = lambda x, y: torch.mean((x - y) ** 2)

mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs_pos, inputs_time):
        num_batches = inputs_pos.shape[0]

        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            out, dx = fn(inputs_pos[i:i+chunk], [inputs_time[0][i:i+chunk], inputs_time[1][i:i+chunk]])
            out_list += [out]
            dx_list += [dx]
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)

    return ret


def run_network(inputs, viewdirs, frame_time, fn, embed_fn, embeddirs_fn, embedtime_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    assert len(torch.unique(frame_time)) == 1, "Only accepts all points from same time"
    cur_time = torch.unique(frame_time)[0]

    # embed position
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    B, N, _ = inputs.shape
    input_frame_time = frame_time[:, None].expand([B, N, 1])
    input_frame_time_flat = torch.reshape(input_frame_time, [-1, 1])
    embedded_time = embedtime_fn(input_frame_time_flat)
    embedded_times = [embedded_time, embedded_time]

    # embed views
    input_dirs = viewdirs[:, None].expand(inputs.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat, position_delta_flat = batchify(fn, netchunk)(embedded, embedded_times)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args, 3)
    embedtime_fn, input_ch_time = get_embedder(args, 1)

    input_ch_views = 0

    # If use_viewdirs
    # common position embedder
    embeddirs_fn, input_ch_views = get_embedder(args, 3)

    model = DirectTemporalNeRF(input_ch=input_ch,
                               input_ch_views=input_ch_views,
                               input_ch_time=input_ch_time,
                               embed_fn=embed_fn).to(args['device'])
    print(model)
    grad_vars = list(model.parameters())

    model_fine = DirectTemporalNeRF(input_ch=input_ch,
                                    input_ch_views=input_ch_views,
                                    input_ch_time=input_ch_time,
                                    embed_fn=embedtime_fn).to(args['device'])
    grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, ts, network_fn: run_network(inputs, viewdirs, ts, network_fn,
                                                                            embed_fn=embed_fn,
                                                                            embeddirs_fn=embeddirs_fn,
                                                                            embedtime_fn=embedtime_fn,
                                                                            netchunk=args['netchunk'])
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars,
                                 lr=args['lr'],
                                 betas=(0.9, 0.999))

    render_kwargs = {
        'network_query_fn': network_query_fn,
        'N_importance': args['N_importance'],
        'network_fine': model_fine,
        'N_samples': args['N_samples'],
        'network_fn': model,
    }

    return render_kwargs, optimizer


def train(args):
    device = args['device']
    images, poses, times, render_poses, render_times, hwf, i_split = load_data(args['basedir'], True, args['testskip'])
    i_train, _, _ = i_split

    # white background
    images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    min_time, max_time = times[i_train[0]], times[i_train[-1]]
    assert min_time == 0., "time must start at 0"
    assert max_time == 1., "max time must be 1"

    # Cast intrinsics to right types 
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # K = np.array([
    #     [focal, 0, 0.5 * W],
    #     [0, focal, 0.5 * H],
    #     [0, 0, 1]
    # ])

    # Create nerf model
    render_kwargs, optimizer = create_nerf(args)
    start = 0
    bds_dict = {
        'near': 2,
        'far': 6,
    }
    render_kwargs.update(bds_dict)

    poses = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)

    N_rand = args['N_rand']
    N_iters = args['N_iter'] + 1

    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)

    print('Begin')
    start = start + 1
    time0 = time.time()
    for i in trange(1, N_iters):
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]
        frame_time = times[img_i]

        # N_rand
        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))

        if i < 500:                 # precrop_iters=500
            dH = int(H // 2 * 0.5)  # precrop_frac=0.5
            dW = int(W // 2 * 0.5)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                ), -1)
            if i == start:
                print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {500}")
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

        rgb, extras = render(H, W, focal, chunk=args['chunk'], rays=batch_rays, frame_time=frame_time, **render_kwargs)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss = loss + img_loss0
        # psnr0 = mse2psnr(img_loss0)

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
        rgbs = render_path(render_poses, render_times, hwf, 4096, render_kwargs)
    print('Done, saving', rgbs.shape)
    moviebase = os.path.join(args['savedir'], '_spiral_{:06d}_'.format(now))
    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
