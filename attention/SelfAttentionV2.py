'''
Implementing a compact self-attention python class (following self_attention_trainable_wts.py)

Self-attention involves trainable wt matrices Wq, Wk, Wv. These matrices transform input data into queries,
keys and values (following a general database search mechanism). As the model is exposed to more data during training,
it adjusts these trainable wts (as we will see later)

We can improve SelfAttention_v1 impl by utilizing PyTorch's nn.Linear layers, which effectively perform matrix multiplication
when the bias units are disabled. Additionally, a signf adv of using nn.Linear instead of manually impl (torch.rand(...)) is that
nn.Linear has an optimized wt initialization scheme, contributing to more stable and effective model training.
'''

import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias) # use Linear instead of Parameters for better optimized inits
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        keys =    self.W_key(x)    # use better optimized vn of matmul (see V1)
        queries = self.W_query(x)  # use better optimized vn of matmul (see V1)`
        values =  self.W_value(x)  # use better optimized vn of matmul (see V1)

        attn_scores  = queries @ keys.T # omega
        attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 ) # scaled dot product to avoid vanishing gradients

        context_vec = attn_weights @ values
        return context_vec 
