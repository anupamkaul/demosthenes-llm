'''

Class for layer normalization (applied before and after multi-headed attention modules as 
part of the transformer block. Layer normalization is used typically for zero mean and unit variance
to resolve vanishing and exploding gradients to optimize and converge neural network training. Normalization
to a layer (instead of batch) also helps in distributed training..

See layer_normalization_example.py for concepts and examples

'''

import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, emb_dim):

        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x): # data flow
  
        mean = x.mean(dim = -1, keepdim = True)
        var  = x.var(dim = -1, keepdim = True, unbiased = False)
		
        norm_x = (x - mean) / torch.sqrt(var + self.epsilon)
        return (self.scale * norm_x) + self.shift


'''

Notes:

This specific implementation of layer normalization operates on the last dimension of the input tensor x,
which represents the embedding dimension (emb_dim). The variable eps is a small constant (epsilon) added 
to the variance to prevent division by zero during normalization. The scale and shift are two trainable parameters 
(of the same dimension as the input) that the LLM automatically adjusts during training if it is determined that 
doing so would improve the model’s performance on its training task. This allows the model to learn appropriate 
scaling and shifting that best suit the data it is processing.

''' 
    
