import torch
from torch import nn
from torch.nn import functional as F


class NeRF(nn.Module):
    
    def __init__(self, num_layers, num_pos, pos_encode_dims=16) -> None:
        super(NeRF, self).__init__()
        self.num_layers = num_layers
        self.num_pos = num_pos
        self.pos_encode_dims = pos_encode_dims

        dim = 128
        self.input_layer = nn.Linear(2 * 3 * pos_encode_dims + 3, dim)

        self.hidden_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i % 4 == 0 and i > 0:
                self.hidden_layers.append(nn.Linear(2 * 3 * pos_encode_dims + 3 + dim, dim))
            else:
                self.hidden_layers.append(nn.Linear(dim, dim))
        self.output_layer = nn.Linear(dim, 4)

    def forward(self, inputs):
        x = F.relu(self.input_layer(inputs))
        for i, layer in enumerate(self.hidden_layers):
            if i % 4 == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

    def save_param(self, path="param.model"):
        torch.save(self.state_dict(), path)

    def load_param(self, device, path="param.model"):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=device)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
