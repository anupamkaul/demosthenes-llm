'''
Takeaway: 

Embedding is basically creating a dimensioned-vector of a singular token so
that the vector can be used "normally" during back propagation. This is so
that the LLM can better understand the relationships between tokens. Basically
so that back-prop and optimizations works. Its a higher dimensional representation
for machine language understanding. 

Initially all that an embedding layer is a set of randomized values associated
to the numer of dimensions of that vector. So: 

The embedding layer is essentially a lookup operation that retrieves rows
from the embedding layer's WEIGHT MATRIX via a Token ID
'''

import torch

# suppose we have the following four input tokens with IDs as:
input_ids = torch.tensor([2, 3, 5, 1])

# for simplicity let vocab only be 6 words
# lets assume we want to creatre embeddings of size 3

vocab_size = 6
output_dim = 3

# we now instantiate an embedding layer in pytorch,
# setting the random seed to 123 for repro purposes

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)  # random initializations (of the underlying weight matrix)

'''
This prints:

Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
'''

# Now, let's apply it to a token ID, to obtain the embedding vector

print(embedding_layer( (torch.tensor([3]) ) ))

'''
This prints
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

This corresponds to data from the 4th row above.
Python starts with zero index, so it's the row corresponding to index 3
3 happens to be the token ID value, so...

The embedding layer is essentially a lookup operation that retrieves rows
from the embedding layer's WEIGHT MATRIX via a Token ID


'''

# if the above principle is correct, let's generalize this to print out
# token embeddings (wieghted matrices) of all of the 4 input IDs (2, 3, 5, 1)

print(embedding_layer(input_ids))

'''
This prints a 4 * 3 matrix:
Each of the row is basically the index into (tokenID+1) lookup of the initial weighted matrix

tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
'''




