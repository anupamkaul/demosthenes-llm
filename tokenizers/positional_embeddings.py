'''

Previously we saw how embeddings can work (embeddings_from_token.py)
LLMs' self-attention mechanisms don't have a notion of the position or
order of the tokens within a sequence. (The same token ID always gets mapped
to the same vector representation). So we need to inject additional position
information into the LLM

Two categories of position-aware embeddings:
a. relative positional embeddings
b. absolute positional embeddings

For each position in the input sequence, a unique embedding is added to the 
token's embedding to convey its exact location (absolute position embedding).
In relative instead of focusing on absolute position, the model learns the relationships
in terms of "how far apart" rather than "at which exact position".

Both embeddings augment the capacity of LLMs to understand the order and relationships
between tokens, ensuring more accurate and context aware predictions. 

'''

'''
Previously we focused on small embedding sizes for simplicity. Now let's consider
more realistic and useful embedding sizes and encode the input tokens into a 
256-dimensional vector representation, which is smaller than what the original GPT-3
model used (GPT-3 had 12,288 dims) but still reasonable for experimentation.

We assume tokens were created with BPE tokenizer implemented earlier, which has a 
vocab of 50257
'''

import torch

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(token_embedding_layer)
print(token_embedding_layer.weight)

'''

using above token_embedding_layer, if we sample data from the data loader
we embed each token in each batch into a 256-dim vector. If we have a batch
of 8 with 4 tokens each, the result would be an 8 * 4 * 256 tensor.

'''
import GPTDatasetV1

print("\nback to the program here..\n")

max_length = 4







