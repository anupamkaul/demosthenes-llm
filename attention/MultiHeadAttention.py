'''
Up until now we have implemented a MultiHeadAttentionWrapper that combined multiple single-head attention (causal) modules. But these are 
processed sequentially (see class MultiHeadAttentionWrapper's forward method defn) as head(x) for head in self.heads. We can improve this 
by processing the heads in parallel (where head was same input but parallelized action units in z-order and combining context vectors into a 
context vector matrix). One way to achieve this is by computing the outputs of all attention heads simultaneously via matmul.

Instead of maintaining 2 classes MultiHeadAttentionWrapper and SelfAttentionV2Causal, we can combine these concepts into a single
MultiHeadAttention class with some other code modifications to implement multi-head attention more efficiently. 

Unlike the previous approach where SelfAttentionV2Causal implemented a single attention unit and the Wrapper basically stacked them, now we split the input into multiple heads by RESHAPING the projected query, key and value tensors, and then combine the results from these heads after computingattention.

'''

import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert(d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
 
        # reduce the projection dim to match the desired output dim
        self.heads_dim = d_out // num_heads # floor div (quotient without decimal part)
        
        # initialize the weight matrices
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)

        # use a linear layer to combine the head outputs
        self.out_proj = nn.Linear(d_in, d_out)

        self.dropout = nn.Dropout(dropout)

        # register and optimize buffer for hw (CPU/GPU/other) - the one used for creating future masks
        # TODO : this needs an assignment

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # calculate queries, keys and values 
        # the tensor shape of the following is (b, num_tokens, d_out)

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        '''
        unlike the wrapper class previously where we stacked two weight matrices and derived two query matrices,
        here we create a single large Weight matrix (see __init_), we do only 1 multiplication to obtain a single
        query matrix, but then we split the query matrix into 2 (Q1 and Q2). We use the .views operator to do these
        matrix transformations (see views_example.py)

        We use .view and we implicitly split the matrix by adding a "num_heads" dimension. Then we unroll the last dim
        (b, num_tokens, d_out) --> (b, num_tokens, num_heads, heads_dim)
        '''

        keys   = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # now we use the .transpose operator to transpose the 1 and 2 dimensions 
        # (see transpose_example.py, category is tensor.transpose
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # calculate attention score
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        print(mask_bool)

        attn_scores.masked_fill (mask_bool, -torch.inf)

        attn_weights = torch.softmax (attn_scores / keys.shape[-1]**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, tokens, self.d_out) # this is the reconstruction
        context_vec = self.out_proj(context_vec)

        return context_vec
