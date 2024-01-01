import torch

from train import train


args = {
    'basedir': "./data/lego",
    'savedir': "./logs",

    'lr': 0.01,

    'chunk': 1024 * 32,
    'netchunk': 1024 * 64,

    'N_samples': 64,
    'N_importance': 128,
    'N_rand': 1024,

    'multires': 10,
    'log2_hashmap_size': 19,

    'precrop_iters': 500,
    'precrop_frac': 0.5
}


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else "torch.FloatTensor")

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device('cpu')

    args['device'] = device
    
    train(args)


if __name__ == "__main__":
    main()
