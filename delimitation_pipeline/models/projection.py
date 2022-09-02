from torch import nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = []
        n_in = self.input_dim
        n_out = self.hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
                nn.Linear(n_out, n_out, bias=False)
            ))

            n_in = self.hidden_dim
            
        layers.append(nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.Linear(n_out, self.output_dim, bias=False)
        ))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        feats = self.model(x)
        return F.normalize(feats, dim=1)

    