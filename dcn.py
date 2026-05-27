import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x

class DCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, cross_layers=3, dropout=0.0):
        super().__init__()
        self.cross_net = CrossNetwork(input_dim, cross_layers)

        deep_layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            deep_layers.append(nn.Linear(in_dim, out_dim))
            deep_layers.append(nn.ReLU())
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.deep_net = nn.Sequential(*deep_layers)

        self.combine = nn.Linear(input_dim + hidden_dims[-1], input_dim)

    def forward(self, x):
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)
        concat = torch.cat([cross_out, deep_out], dim=-1)
        out = self.combine(concat)
        return out