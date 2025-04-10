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

'''

As we can see from dump_TransformBlock.txt, the transformer block maintains the input dimensions in its output,
indicating that the transformer architecture processes sequences of data without altering their shape throughout the network.

The preservation of shape throughout the transformer block architecture is not incidental but a crucial aspect of its design. 
This design enables its effective application across a wide range of sequence-to-sequence tasks, where each output vector directly 
corresponds to an input vector, maintaining a one-to-one relationship. However, the output is a context vector that encapsulates information 
from the entire input sequence. This means that while the physical dimensions of the sequence (length and feature size) remain unchanged as 
it passes through the transformer block, the content of each output vector is re-encoded to integrate contextual information from across the
entire input sequence.

With the transformer block implemented, we now have all the building blocks needed to implement the GPT architecture.The transformer block 
combines layer normalization, the feed forward network, GELU activations, and shortcut connections. As we will eventually see, this transformer 
block will make up the main component of the GPT architecture.

'''

'''
block analysis:
calculate the number of parameters (wts) of the transformer block
(note that GPT2 uses 12 of these blocks (as specified by cfg["n_layers"]
'''

params_trf = sum(p.numel() for p in block.parameters()) 
print(f"params of a single transformer block: {params_trf:,}")


