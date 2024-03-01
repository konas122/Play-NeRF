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
    'N_rand': 500,
    'N_iter': 800000,

    'multires': 4,
    'testskip': 1,
}


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else "torch.FloatTensor")

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        try:
            import torch_directmla
            device = torch_directml.device()
        except ImportError:
            device = torch.device('cpu')

    args['device'] = device
    
    train(args)


if __name__ == "__main__":
    main()
