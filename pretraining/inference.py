'''
Use this as sample inference code to the demosthenes model
'''

import torch

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

import textgenerate
from textgenerate import text_to_token_ids, generate_text_simple, token_ids_to_text

# load previously saved instance of the model (to check inference) 
#(start with the simple model generated from a single file (the verdict))

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

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Comment out the following 2 lines when GPU works..
device = torch.device("cpu")
print("device override for my local ubuntu: ", device)

import platform
if (platform.system() != "Darwin"):
    print("compiling the model (non macOS")
    model = torch.compile(model)

model.to(device)

try:

    model.load_state_dict(torch.load("./model/model.pth", map_location=device))
    #model.load_state_dict(torch.load("./gutenberg/gutenberg/model_checkpoints/model_pg_275_interrupted.pth", map_location=device))

    print("\nmodel loaded .. <enter>")
    input()

except FileNotFoundError:
    print("model not found on disk. monitor as a one time thing, error out if repeats")
    exit

# benchmark a string (every effort moves you)
token_ids = generate_text_simple(
    model          = model,
    idx            = text_to_token_ids("Every effort moves you", tokenizer), # or the user's input (for a chat)
    max_new_tokens = 25,
    context_size   = GPT_CONFIG_124M["context_length"]     

)

print("\nchat output: ", token_ids_to_text(token_ids, tokenizer))

# chat version..
# now we ask user for their input and check how we do

while(True):
    user_input = input("\nchat with me: (and press enter) ")

    token_ids = generate_text_simple(
        model          = model,
        idx            = text_to_token_ids(user_input, tokenizer), # or the user's input (for a chat)
        max_new_tokens = 25,
        context_size   = GPT_CONFIG_124M["context_length"]     

    )
    print("\nchat output: ", token_ids_to_text(token_ids, tokenizer))


