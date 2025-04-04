# test the FeedForward class

import torch
import FeedForward as ff

GPT_CONFIG_124M = {
    "vocab_size"     : 50257,       # vocabulary size
    "context_length" : 1024,        # context length
    "emb_dim"        : 768,         # embedding dimension
    "n_heads"        : 12,          # number of attention heads
    "n_layers"       : 12,          # number of layers
    "drop_rate"      : 0.1,         # dropout rate
    "qkv_bias"       : False       # query-key-value bias

}

ffn = ff.FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("out shape: \n", out.shape)
print("out: \n", out)

