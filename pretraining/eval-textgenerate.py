import torch

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

from GPTModel import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
print(model)

# add utility functions for text to tokenID conversion

import tiktoken
from generate_text_simple import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # unsqueeze(0) adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10,
    context_size = GPT_CONFIG_124M["context_length"]
)

print("token ids: ", token_ids)
print("Output text: \n", token_ids_to_text(token_ids, tokenizer))

'''
Let's setup framework for numerically assessing the text quality generated
during training by calculating the 'text generation loss'. (This loss metric
will serve as progress indicator of the training progress) 

First, let's map the text generation process explicity to the 5 step process
shown in textgenerate.png
'''

'''
step1: use vocabulary to map the input text to tokenIDs
Consider these 2 inputs, which have already been mapped to their token IDs from the vocab
'''

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

'''
matching to these inputs we define targets containing the tokenIDs that we want the model
to produce (after all a loss function must know what the goal is). We assume a sliding window
model where the next word to the right is the word we'd like the model to output (also see
DataLoader - this shifting strategy is crucial for teaching the model to predict the next token
in a sequence.
'''

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107, 588, 11311]])  #  " really like chocolate"]


