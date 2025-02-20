'''
Self-attention is a mechanism that allows each position in the INPUT sequence to consider the relevancy of, or "attend to" all other
positions in the same INPUT sequence when computing the representation of the sequence. 

It asessess and learns the relationships and dependencies between various parts of the INPUT itself, such as words in a sentence, or pixels in an emage (or later - contextual relationships between concepts as part of a larger conversation - TODO ! )

(Follows from Bahdanau attention mechanism that had RNN decoder (output) access all states of the encoder (input) sequence. RNNs focused on relationships between elements of two different sequences, such as in sequence-to-sequence models where the attention might be between an input sequence and and output sequence. Self-attention, and hence transformers, eliminates RNNs entirely and establishes a new order of encode/decode with self-attention)

'''

'''

Let's begin by implementing a simple version of self-attention, free from any trainable weights.
The goal is to illustrate a few key concepts, like calculating the context-vector of each token (represented as a embedded vector)

Let's say there is a sentence "Your journey starts with one step". Let's assume there is a tokenizer (like BPE) that has evalued the
tokens and an embedded vector representation that has 3 dimensions only (for illustration)

The goal of self-attention is to compute a CONTEXT VECTOR for each input element that combines information from all other input elements.

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

print("inputs : \n", inputs, "\n")

'''

The context vector (say for query 2 or Input x^2) will comprise of attention score
a21, a22, a23 ,, a2T --> all of which will be used to create z^2. z^2 is the context vector
for query x^2, and a21..a2T is the attention score associated with x^2. As dot_product.py illustrates
attention at its basic format could be nothing more than a dot-product between 2 elements (to start with)
that maps a similarity-index relationship between an input token and another in a sentence.

In self-atttention, our goal is to calculate context-vectors z(i) for eaxh element x(i) in the input sequence.
A context vector can be interpreted as an enriched embedding vector. 

The file simple-self-attention-no-wts.py shows code for calculating context vector for element x(2) by first calculating
the attention score vector for query2. From here the code diverges to first simultaneously calculate attention scores of
every input token, and then calculate context vectors for every token (embedded vector format as starting points) 

Step 1 is we calculate attention scores for every input token that was expressed as an embedded tensor

'''

#normally we calculate intermediate attention scores between a query token (of the input sentence)
# with each input token. We determine these scores by computing the dot product of the query (for every x(i))
# with every other input token

# attention scores of every query (like x^2) 
# this will be a 6 * 6 array with every row corresponding to an input query and every row's 6 cols representing
# the attention scores of that row's input query
attn_scores_forloop = torch.empty(6, 6)  

print("calculate attn_scores of all 6 input tokens:")
# basically replace query with x_j and run an additional internal for loop

for i,x_i in enumerate(inputs):
    print("i=", i, " x_i = ", x_i)
    for j, x_j in enumerate(inputs):
        print("    j=", j, " x_j = ", x_j)
        attn_scores_forloop[i, j] = torch.dot(x_i, x_j)

print("\nattn score of query element is the dot product of itself with each input token shown below : \n", attn_scores_forloop, "\n")

'''
for loops are generally slow in the above calc. I can also do the same attention score calculation by using matrix multiplication,
and the operator @ as follows:
'''

attn_scores = inputs @ inputs.T

print("mat mul based attention scores: \n", attn_scores)

'''

Step 2: 

In the next step we Normalize each of the attention scores we computed previously. The main goal behind normalization is to 
obrain attention weights that sum up to 1. This normalization is a convention that is useful for interpretation and maintaining 
training stability in an LLM. Here is a straight fwd way for normalization:
'''

attn_weights = torch.softmax(attn_scores, dim=-1)

print("\nNormalization: based on Pytorch's softmax")
print("Normalized Attention weights:\n", attn_weights)

'''
Verify that the above is normalized, i.e. all of the row values add up to 1
'''
print("All row sums:\n", attn_weights.sum(dim=-1)) 

'''
Note on usage of dim=-1 above:

By setting dim=-1, we are instructing the softmax (and sum) function to apply the normalizatio (and sum)
along the last dimension of the attn_scores tensor. If this tensor is a 2-dim tensor (e.g. shape of [rows, col]),
it will normalize across the cols, so that the values in each row (summing over the col dimension) sum up to 1

'''

'''

Step 3: We calculate the Context Vector for every input token
Use matmul with attn_weights tensor and the inputs tensor basically

'''

all_context_vec = attn_weights @ inputs

print("\nContext Vectors:\n", all_context_vec)
