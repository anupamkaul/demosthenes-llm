'''
Code to check general inference (via a chatbot) for the instruction-follow
demosthenes model that I created

1. Load the model
2. Load in all the dependencies
3. Run a loop, get user input, print out a response
4. Evaluate the goodness of the response manually
5. Augment this code with standardized LLM evaluation methodologies as the next 
   step (LLM as judget, IFCritic, etc)
'''

# load the model
import torch

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

from GPTModel import GPTModel
model = GPTModel(BASE_CONFIG)
model.eval()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
 
model.to(device)

print("loading model..")

import time
start_time = time.time()

try:
    model.load_state_dict(torch.load("./model/modelif.pth", map_location=device))

except FileNotFoundError:
    print("inference model not found on disk. please provide the correct path")
    exit()

end_time = time.time()
loading_time_minutes = (end_time - start_time) / 60
print(f"Loaded in {loading_time_minutes:.2f} minutes.")

'''
Do a chatbot that detects responds to instruction-follow prompts and/or normal text messages

It's a bummer that prompts need special processing vs normal text (i.e. one must know that 
the incoming text is an instruction vs normal text? to be checked later..) 
'''

from stylize_prompts import format_input

while(True):
    user_input = input("\nwrite up something (and press enter) ")
    input_text = format_input(user_input) # this does not include the output 
    print("formatted input: ", input_text)# doesn't work if input is not an "entry" (i.e. in a prompt format already)
    

