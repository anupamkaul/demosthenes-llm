'''

Previously we saw how embeddings can work (file "embedding_from_token.py")
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
import dataloaderV1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = dataloaderV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, target = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

'''
We can see that token ID tensor is 8 * 4 dim, meaning that the data
batch consists of eight text samples with 4 tokens each

Let us now use the embedding layer to embed these token IDs into 256-dim
vectors:
'''

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)  # returns torch.Size([8, 4, 256])
# print("token embeddings: \n", token_embeddings)

'''
Tghe 8 * 4 * 256 dim vector tensor output shows that each token ID is 
now embedded as a 256 dimensional vector.

For a GPT model's absolute embedding approach, we jsut need to create
another embedding layer that has the same embedding dim as the 
token_embedding_layer

'''

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer( torch.arange(context_length) )
print(pos_embeddings.shape)

'''
The input to pos_embeddings is a placeholder vector usually torch.arange(context_length).
This contains a sequence of numbers 0,1,... up to the maximum input_length-1.
The context length is a variable that represents the supported input size of the LLM.
Here it is equal to maximum length of input text. In practice if it is longer, we truncate
the text.

The output is torch.Size([4, 256]) meaning the positional embedding tensor consists of 
four 256 dim vectors. 

We now add these directly to token embeddings (as promised) where PyTorch will add the
4 * 256-dim pos_embeddings tensor to each of the 4 * 256-dim token embedding tensor in
each of the 8 batches:
'''

input_embeddings = token_embeddings + pos_embeddings
print("input embeddings are now (with positional embedding inserted) : \n", input_embeddings, "\n")
print("input_embeddings shape (adding positional embeddings) is \n", input_embeddings.shape)

'''
This returns torch.Size([8, 4, 256]).

The input_embeddings that we have created, are the input embedded examples that can
now be processed by the main LLM modules.
'''
