
import torch
import torch.nn as nn
import GELU as gelu

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear( cfg["emb_dim"], 4 * cfg["emb_dim"]),  # layer 1: expand by 4
            gelu.GELU(),  # gelu activation
            nn.Linear( 4 * cfg["emb_dim"], cfg["emb_dim"])   # layer 2: contract by 4, to match input
        )

    def forward(self, x):
        return self.layers(x)




