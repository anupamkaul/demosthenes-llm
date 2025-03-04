'''
Implementing a compact self-attention python class (following self_attention_trainable_wts.py)

Adding causal attention to V2. We assume the code handles batches consisting of more than one input, so 
that this class supports batch outputs produced by our dataloader in tokenizer folder. We also add a dropout
layer over the V2 self attention class (and we combine concepts from causal_mask_attn.py)

Self-attention involves trainable wt matrices Wq, Wk, Wv. These matrices transform input data into queries,
keys and values (following a general database search mechanism). As the model is exposed to more data during training,
it adjusts these trainable wts (as we will see later)

We can improve SelfAttention_v1 impl by utilizing PyTorch's nn.Linear layers, which effectively perform matrix multiplication
when the bias units are disabled. Additionally, a signf adv of using nn.Linear instead of manually impl (torch.rand(...)) is that
nn.Linear has an optimized wt initialization scheme, contributing to more stable and effective model training.
'''

import torch
import torch.nn as nn

class SelfAttentionV2Causal(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        # explcit init
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias) # use Linear instead of Parameters for better optimized inits
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        # we add a dropout layer to previous SelfAttention_V2
        self.dropout = nn.Dropout(dropout)

        '''
        we add a register buffer call in __init__ method. While usage of register_buffer in Pytorch is not necessary for all
        usecases, for LLMs the adv is: when we use SelAttentionV2Causal class buffers are automatically moved to the appropriate
        device (CPU or GPU or something unique), which will be relevant when training our LLM. This means I don't need to manually
        ensure these tensors are on the same device as my model parameters, avoiding device mismatch errors.
        '''
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        # new addition of params:
        b, num_tokens, d_in = x.shape # b is declated but not used but it stands for batch and ensures other params get corect values

        keys =    self.W_key(x)    # use better optimized vn of matmul (see V1)
        queries = self.W_query(x)  # use better optimized vn of matmul (see V1)`
        values =  self.W_value(x)  # use better optimized vn of matmul (see V1)

        # attn_scores  = queries @ keys.T # omega
        
        # When calculating attention scores we now transpose dimensions 1 and 2, keeping the batch dimension in the first position (0)
        attn_scores  = queries @ keys.transpose(1, 2)

        print("\nclass SelfAttentionV2Causal: attn_scores pre Mask:\n", attn_scores)

        # now we apply the mask to attn_scores (code is different from causal_mask_attn.py) 
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        print("\nclass SelfAttentionV2Causal: attn_scores post Mask:\n", attn_scores)

        attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 ) # scaled dot product to avoid vanishing gradients
        print("\nclass SelfAttentionV2Causal: att_wts post scaled dot:\n", attn_weights)

        attn_weights = self.dropout(attn_weights) # apply dropout
        print("\nclass SelfAttentionV2Causal: att_wts post scaled dot and dropout:\n", attn_weights)

        context_vec = attn_weights @ values
        return context_vec 
