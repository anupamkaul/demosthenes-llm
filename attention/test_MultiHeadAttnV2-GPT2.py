import torch
import MultiHeadAttention as MhaAttn

# modify params to match GPT2 scale

'''
Using the MultiHeadAttention class, initialize a multi-head attention module that has the same number of attention heads 
as the smallest GPT-2 model (12 attention heads). 

In this test ensure that you use the respective input and output embedding sizes similar to GPT-2 (768 dimensions). 
Note that the smallest GPT-2 model supports a context length of 1,024 tokens.

'''

'''
inputs = torch.tensor(
    [ [0.43, 0.15, 0.89],    # Your      (x^1)
      [0.55, 0.87, 0.66],    # journey   (x^2) 
      [0.57, 0.85, 0.64],    # starts    (x^3)
      [0.22, 0.58, 0.33],    # with      (x^4) 
      [0.77, 0.25, 0.10],    # one       (x^5)
      [0.05, 0.80, 0.55]     # step      (x^6)
    ]
)
'''

'''
Instead of the 1 sentence input above:
Use actual vocabulary (story file), convert input to positional embedding (768 dimensions)
and proceed

We need to import modules from tokenizers. Temporarily add to python's search path
to do so. (tokenizers and is module is in ../ )
'''

import sys 
import os

# get current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir: ", current_dir, "\n")

# add the actual path where import module is
module_path = os.path.join(current_dir, '../tokenizers/')
print("module path to be used for import: ", module_path, "\n")

# add new path to sys.path (what python uses to search imported modules)
sys.path.append(module_path)

import dataloaderV1

with open("../tokenizers/the-verdict.txt", "r", encoding="utf-8") as f:
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

vocab_size = 50257
output_dim = 768

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("token embedding layer:\n", token_embedding_layer, "\n")
print("weights of this layer: \n", token_embedding_layer.weight, "\n")

'''
We can see that token ID tensor is 8 * 4 dim, meaning that the data
batch consists of eight text samples with 4 tokens each

Let us now use the embedding layer to embed these token IDs into 768-dim
vectors:
'''

token_embeddings = token_embedding_layer(inputs)

print("token embedding shape: ", token_embeddings.shape)  # returns torch.Size([8, 4, 768])
# print("token embeddings: \n", token_embeddings)

'''
The 8 * 4 * 768 dim vector tensor output shows that each token ID is 
now embedded as a 768 dimensional vector.

For a GPT model's absolute embedding approach, we just need to create
another embedding layer that has the same embedding dim as the 
token_embedding_layer

'''

context_length = 4 # this length has to match the max length of token_embedding_layer
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer( torch.arange(context_length) )
print("shape of positional embedding: ", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings

print("input embeddings are now (with positional embedding inserted) : \n", input_embeddings, "\n")
print("input_embeddings shape (adding positional embeddings) is \n", input_embeddings.shape)

'''
for causal and multi headed self attention we assume code can handle batches of input data streams
'''

#batch = torch.stack((inputs, inputs), dim=0)
#batch = torch.stack((input_embeddings, input_embeddings), dim=0)

#batch = torch.stack((input_embeddings, input_embeddings, input_embeddings, input_embeddings, input_embeddings), dim=0)

batch = input_embeddings
print("batch shape: ", batch.shape, "\n")
print("batch: \n", batch)

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 768 

#mha = MhaAttn.MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads = 2)
mha = MhaAttn.MultiHeadAttention(d_in, d_out, 1024, 0.0, 12)

context_vec = mha(batch)  # calls the forward method of MultiHeadAttention class (MultiHeadAttention.py)
print("context vecs:\n", context_vec)
print("\nshape of context vec: ", context_vec.shape)

