'''
Up until now we have implemented a MultiHeadAttentionWrapper that combined multiple single-head attention (causal) modules. But these are 
processed sequentially (see class MultiHeadAttentionWrapper's forward method defn) as head(x) for head in self.heads. We can improve this 
by processing the heads in parallel (where head was same input but parallelized action units in z-order and combining context vectors into a 
context vector matrix). One way to achieve this is by computing the outputs of all attention heads simultaneously via matmul.

Instead of maintaining 2 classes MultiHeadAttentionWrapper and SelfAttentionV2Causal, we can combine these concepts into a single
MultiHeadAttention class with some other code modifications to implement multi-head attention more efficiently. Unlike the previous approach
where SelfAttentionV2Causal implemented a single attention unit and the Wrapper basically stacked them, now we split the input into multiple heads
by reshaping the projected query, key and value tensors, and then combine the results from these heads after computing attention.

'''

import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

    def forward(self, x):
        context_vec = 0
        return context_vec
