import torch
import torch.nn as nn
import SelfAttentionV2Causal as SelfAttn

'''

Exted causal attention class over multiple heads (stacked up causal self-attention units). This is multi-head attention.
The term 'multi-head' refers to the same input stream but dividing the attention mechanisms into multiple "heads" with each
head operating independently. In this context, a single causal attention module can be considered single-head attention, where
there is only 1 set of attention weights processing the input sequentially.

First, we will intuitively build a multi-head attention module by stacking multiple Causal-Attention modules (this file). We use
nn.ModuleList to help us with the class stacking.

Next, we will implement the same multi-=head attention in a slightly more complicated but more computationally efficient way.

Input X remains the same. Instead of 1 Wq Wk and Wv now we have multiple. Similarly there are multiple Qx Kx and Vx and multiple 
context vectors that get generated. (Its like input is the same but everything else is directly stacked up in Z-order). E.g. if there
were 2 Causal SA modules, we now have multi-headed attention with 2 modules. Two value weight matrices for ex. Wv1 and Wv2. Same applies
to other Wt matrices, Wq and Wk. We obtain 2 sets of context vectors Z1 and Z2 that we can finally combine into a single context vector matrix Z.

Remember this is still infra/shell, (the wts haven't been trained yet and so wt matrices are default/random). We run the attention mechanism
multiple times (in parallel) with different, learned linear projections - the results of multiplying the input data (like query, key and value
vectors in attention mechanisms) by a weight matrix. 

To do this we implement MultiHeadAttentionWrapper class that stacks multiple instances of our previously implemented CausalAttention module.

'''

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
 
        # stack up as many heads as num_heads indicates
        # notice the [ <> .. for loop that uses <> ] syntax

        self.heads = nn.ModuleList(

            [SelfAttn.SelfAttentionV2Causal(
                d_in, d_out- context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat( [head(x) for head in self.heads], dim=-1) 




