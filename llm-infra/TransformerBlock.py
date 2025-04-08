'''
This is where it all comes together : the transformer block encapsulates
multiheaded attention module (with dropout), layer normalization applied before and after MHA,
a feed forward network with dropout, with GELU activation, and with shortcuts. This transformer
block is then repeated in the LLM (a dozen times in 124M param GPT-2 arch, but obviously much more now)
'''


'''
We need to import modules from attention folder. Temporarily add to python's search path
to do so. (tokenizers and is module is in ../ )
'''

import sys 
import os

# get current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir: ", current_dir, "\n")

# add the actual path where import module is
module_path = os.path.join(current_dir, '../attention/')
print("addnl module path to be used for import: ", module_path, "\n")

# add new path to sys.path (what python uses to search imported modules)
sys.path.append(module_path)

import torch
import torch.nn as nn

import MultiHeadAttention as _mha
import FeedForward as _ff
import LayerNorm as _ln

class TransformerBlock(nn.Module):

    def __init__(self, cfg): # initialize blocks
        super().__init__()

        self.att = _mha.MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = _ff.FeedForward(cfg)
        self.norm1 = _ln.LayerNorm(cfg["emb_dim"])
        self.norm2 = _ln.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) 
        # to prevent over fitting, and seems like the pytorch dropout might have implemented shortcut as well

    def forward(self, x): # how the data flows

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # add the input of the block to its output

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

'''

The above code defines a TransformerBlock class in PyTorch that includes a multi-head attention mechanism (MultiHeadAttention) 
and a feed forward network (FeedForward), both configured based on a provided configuration dictionary (cfg), such as GPT_CONFIG_124M.

Layer normalization (LayerNorm) is applied before each of these two components, and dropout is applied after them to regularize the model 
and prevent overfitting. This is also known as Pre-LayerNorm. Older architectures, such as the original transformer model, applied layer 
normalization after the self-attention and feed forward networks instead, known as Post-LayerNorm, which often leads to worse training dynamics.

The class also implements the forward pass, where each component is followed by a shortcut connection that adds the input of the block to its 
output. This critical feature helps gradients flow through the network during training and improves the learning of deep models.

'''



