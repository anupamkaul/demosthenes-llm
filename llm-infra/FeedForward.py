
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


# test the class

GPT_CONFIG_124M = {
    "vocab_size"     : 50257,       # vocabulary size
    "context_length" : 1024,        # context length
    "emb_dim"        : 768,         # embedding dimension
    "n_heads"        : 12,          # number of attention heads
    "n_layers"       : 12,          # number of layers
    "drop_rate"      : 0.1,         # dropout rate
    "qkv_bias"       : False       # query-key-value bias

}

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("out shape: \n", out.shape)
print("out: \n", out)



