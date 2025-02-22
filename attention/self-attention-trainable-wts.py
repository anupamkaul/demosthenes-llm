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


