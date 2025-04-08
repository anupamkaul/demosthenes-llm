# test the FeedForward class

import torch
import FeedForward as ff
import GPT_CONFIG_124M as gpt2_cfg

ffn = ff.FeedForward(gpt2_cfg.get_GPT_CONFIG_124M())
x = torch.rand(2, 3, 768)
out = ffn(x)
print("out shape: \n", out.shape)
print("out: \n", out)

