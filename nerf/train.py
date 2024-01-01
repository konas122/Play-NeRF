import torch
from util import *
from module import *
from loader import *


class TrainNeRF:

    def __init__(self, num_layers, num_pos, device, pos_encode_dims=16) -> None:
        self.device = device
        self.num_layers = num_layers
        self.num_pos = num_pos
        self.pos_encode_dims = pos_encode_dims

        self.loss = nn.MSELoss()
        self.module = NeRF(num_layers, num_pos, pos_encode_dims)

    def train(self, train_iter, batch_size=8, num_epoch=20, lr=0.1, wd=2e-4):
        self.module.train()
        self.module.to(self.device)
        self.trainer = torch.optim.Adam(params=self.module.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(num_epoch):
            print(f"epoch {epoch + 1}: ")
            for images, rays_flat, t_vals in train_iter:
                images, rays_flat, t_vals   \
                    = images.to(self.device), rays_flat.to(self.device), t_vals.to(self.device)
                self.trainer.zero_grad()
                
                rgb = render_rgb(self.module, rays_flat, t_vals, batch_size)

                loss = self.loss(images, rgb)
                loss.backward()
                self.trainer.step()

                with torch.no_grad():
                    psnr = 10. * torch.log10(1 / loss)
                print(f"\tloss: {loss:.3f} ") 
                print(f"\tpsnr: {psnr:.3f} \n")

    def save_param(self, path="param.model"):
        self.module.save_param(path)

    def load_param(self, path="param.model"):
        self.module.load_param(self.device, path)
