import torch
import torch.nn as nn

from utils import get_voxel_vertices


class HashEmbedder(nn.Module):

    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        """
        bounding_box
        n_levels:               L in the paper
        n_features_per_level:   F in the paper
        log2_hashmap_size:      T in the paper
        base_resolution:        Nmin (16)
        finest_resolution:      Nmax
        """
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)

        self.out_dim = self.n_levels * self.n_features_per_level    # 16 * 2 = 32

        # Formula 3 in the paper
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        # [2**19, 2]
        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size,
                                                      self.n_features_per_level) for _ in range(n_levels)])

        # custom uniform initialization
        # embeddings Initialise
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()


    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        """
        Trilinear interpolation of the 8 points in the cube
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3
        # first interpolation 8 -> 4

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]
        # second interpolation 4 -> 2

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]
        # third interpolation 2 -> 1

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c


    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []

        # n_levels is 16
        for i in range(self.n_levels):
            # base_resolution 16
            # formula 3 in the paper, resolution is Nl of formula 2 in the paper
            resolution = torch.floor(self.base_resolution * self.b ** i)

            # The hash index: hashed_voxel_indices [batch_size, 8]
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(
                x, self.bounding_box,
                resolution, self.log2_hashmap_size)

            # [batch_size, 8, 2]
            # Get the value corresponding to the index value in the hash table
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            # [batch_size, 2]
            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)

            x_embedded_all.append(x_embedded)

        # concatenate 16 layers [batch_size, 16 * 2]
        return torch.cat(x_embedded_all, dim=-1)
