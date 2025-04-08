import torch
import torch.nn as nn
import GPT_CONFIG_124M as gpt2_cfg
import TransformerBlock as trf

torch.manual_seed(123)
x = torch.rand(2, 4, 768) # 768 is emb_dim of tokens
block = trf.TransformerBlock(gpt2_cfg.get_GPT_CONFIG_124M())
output = block(x)

print("Transformer: Input Shape: ", x.shape, "\n")
print("Transformer: Output Shape: ", output.shape, "\n")
print("Transformer Output: \n", output)





