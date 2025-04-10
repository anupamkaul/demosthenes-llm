# test the FeedForward class

import torch
import FeedForward as ff
import GPT_CONFIG_124M as gpt2_cfg

ffn = ff.FeedForward(gpt2_cfg.get_GPT_CONFIG_124M())
x = torch.rand(2, 3, 768)
out = ffn(x)
print("out shape: \n", out.shape)
print("out: \n", out)

'''
Model analysis : let's understand how many parameters (wts) are in this 
feed fwd class
'''

params_ff = sum(p.numel() for p in ffn.parameters()) 
print(f"params of the feedforward class: {params_ff:,}")

'''
params of the feedforward class: 4,722,432

(this is 6149 times 768 (emb_dim)) 

'''
