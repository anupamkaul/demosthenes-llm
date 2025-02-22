'''

Previous code was for a simplified attention mechanism to understand how attention works (with dot products).
Now we add trainable weights to this attention mechanism. Later we will extend by adding causal mask and multiple threads.

Trainable weights builds on previous concepts: we want to compute context vectors as Weighted Sums over the input vectors
specfiic to a certain input element (last time it was just a sum, without weights). Most notably we introduce weight matrices
that are updated during model training. These traininable weight matrices are most crucial so that the model (esp the attention
module inside the model) can learn to produce "good" context vectors (that really embedded "good" context). This will be explored
further during LLM training. (Context vectors are described in prev files)

Two approaches:
1. We code this step by step (as we did in Simple..)
2. We organize the code into a compact Python class that can be imported into the LLM architecture.

'''

'''

Part 1: Computing the attention weights step by step:

We implement the attention mechanism with traninable weights step by step, by introducing 3 trainable weight matrices:
Wq, Wk and Wv. These 3 matrices are used to project the embedded input tokens x(i) into Query, Key and Value vectors respectively.
The W matrices have weight values that get trained.

Basically an x(i) transforms into a q(i), k(i) and v(i) based on matmul with Wq, Wk, Wv (where the 3 Ws are weight matrices that get trained)

Here we will choose "journey" (2nd input token) as the current input vector to create the query, to approach this with a single input token first.
Thus x(1) will yield k(1) and v(1) based on matmuls with Wk and Wv, while x(2) will yield q(2), k(2) and v(2) based on matmuls with Wq, Wk and Wv.

(Assume the sentence is still "Your journey starts with one step")

'''

import torch

# let the following tensor represet a 3-dim embedding
# of tokens of the sentence "Your journey starts with 1 step" :

inputs = torch.tensor(
    [ [0.43, 0.15, 0.89],    # Your      (x^1)
      [0.55, 0.87, 0.66],    # journey   (x^2) 
      [0.57, 0.85, 0.64],    # starts    (x^3)
      [0.22, 0.58, 0.33],    # with      (x^4) 
      [0.77, 0.25, 0.10],    # one       (x^5)
      [0.05, 0.80, 0.55]     # step      (x^6)
    ]
)

x_2 = inputs[1]

d_in = inputs.shape[1]
d_out = 2

'''
Note that in GPT like models, the input and output dimensions (dims) are usually the same, but to better follow the computation, we will
use different input (d_in = 3) and output (d_out = 2) dimensions here
'''

# initialize the 3 weight matrices Wq, Wk and Wv:

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

'''
We set requires_grad=False to reduce clutter in the outputs. If we are to actually use the weight matrices for model training we would
set requires_grad=True to update the W matrices during model training.
'''

print("Randomized W_query weight matrix: \n", W_query, "\n")
print("Randomized W_key weight matrix: \n", W_key, "\n")
print("Randomized Q_value weight matrix: \n", W_value, "\n")

# compute now for x_2 its query, key and value vectors (q_2, k_2, v_2) 
q_2 = x_2 @ W_query
k_2 = x_2 @ W_key
v_2 = x_2 @ W_value

# outputs will be 2-dim vectors since we chouse d_out = 2 for our calculations:
print("Query_2 as a func of W_Query: \n", q_2, "\n")

'''

Note that weight parameters are different from attention weights. In weight matrices, weight is short for "weight parameters",
the values of the neural network that are optimized during training. This is not to be confused with attention weights (in prev file)
which determine the extent to which a context vector depends on (or should 'pay attention to') the different parts of the input (i.e. to 
what extent the network focuses on different parts of the input). Weight parameters are the fundamental, learned coefficients that define 
the network's connections (neural) while attention weights are dynamic, context specific values for an LLM.

'''

'''

Now let's compute the key and value elements for all input elements as they are involved in computing the attention weights with 
respect to the query q_2 later. We can obtain all keys and values via matmul

'''

keys = inputs @ W_key
values = inputs @W_value

print("keys.shape:", keys.shape) # should be torch.Size([6, 2])
print("keys: \n", keys)

print("values.shape:", values.shape) # should be torch.Size([6, 2])
print("values: \n", values)

'''
now we have calculated keys and values of all 6 tokens and these are stored in key and value tensors (6, 2). Remember that they were 
calculated with matmuls with the standard W_query, W_key, W_values which contain randomized weights but iteratively would contain learned
weights based on training. 

Step 2: compute the attention scores for every input. The unscaled and non-normalized attention score is computed as a dot product between
the query and the key vectors only. In order to compute the context vector for the 2nd input token, the query is derived from the 2nd input token. Note that this attention score computation as a dot-product computation is similar to what I used in Simple.. the new aspect is that we are not 
directly computing the dot-product between the input elements, but using the query and key (projections) obtained in transforming the inputs via the respective Weight Matrices

First let's compute attention score w22:

''' 

keys_2 = keys[1]
attn_score_22 = q_2.dot(keys_2)  # note that q_2 was x_2 @ W_query ...
print("attn_score_22: \n", attn_score_22, "\n")

# generalize this computation for all attention scores via matmul

attn_scores_2 = q_2 @ keys.T  # all attention scores for a given query
print("attn scores for element 2: \n", attn_scores_2, "\n")

'''
Step 3: Go from attention_scores to attention_weights. We compute the attention_weights by scaling the attention_scores and using
the softmax function as we did previously. However, now, we scale the attention scores by dividing them by the sq-root of the embedding
dimension of the keys. Note that taking sq-root is mathematically the same as exponentiating by 0.5. 

The reason for normalization by the embedding dimension size is to improve the training performance by avoiding small gradients. For instance,
when scaling up the embedding dimension, which is typically greater than 1000 for GPT-like LLMs, large dot products can result in very small
gradients during back propagation due to the softmax applied to them. As dot products increase, the softmax function behaves
more like a step function, resulting in gradients nearing zero (vanishing gradients problem). Gradient calculation are the lifeline of neural network training so small gradients can drastically slow down learning or cause training to stagnate. The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.
'''

d_k = keys.shape[-1]  # get the embedding dimension
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn weight (scaled and adding to 1 : obtained also by div with sq_root(emb-dim of keys)) for element 2: \n", attn_weights_2, "\n")
