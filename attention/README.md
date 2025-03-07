Looking at attention mechanisms in isolation.
Focus on them at a mechanistic level.
Code remaining parts of LLM surrounding Self Attention mechanism to 
see it in action and to create a model to generate text.

Implement 4 different tyles of attention mechanisms:
1. Simplified self-attention : A simplified self-attention technique to introduce the broader idea

2. Self-attention : Self-attention with trainable weights that forms the basis of the mechanism used in LLMs

3. Causal attention : A type of self-attention used in LLMs tht allows a model to consider only previous and current inputs in a sequence,
                      ensuring a temporal order during the text generation.

4. Multi-head attention : An extension of self-attention and causal attention that enables the model to simultaneously attend to information
from different representation subspaces.

Start with a simplified version of self-attention, before adding trainable weights. The causal attention mechanism adds a mask to self-attention
that allows the LLM to generate one word at a time. Finally multi-heade attention organizes the attention mechanism into multiple heads, allowing
the model to capture various aspects of the input data in parallel.

With these 4, goal would be to arrive at a compact and efficient implementation of multi-headed attention that we can then plug into the 
LLM architecture that we will code up next.

File order for perusal (in order):

dot_product.py
simple-self-attention-no-wts.py
context-vec-simple-attn-nowts.py
self-attention-trainable-wts.py
SelfAttentionV1.py
test_SelfAttentionV1.py
SelfAttentionV2.py
test_SelfAttentionV2.py
causal_mask_attn.py
SelfAttentionV2Causal.py
test_SelfAttentionV2Causal.py
MultiHeadAttn_StackedCausalSA.py


