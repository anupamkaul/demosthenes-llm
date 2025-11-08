'''
Now I will use the "test" set of the data to check
how good the instruction fine tuning is (and general
methods of validating LLM accuracies)

First I will extract the model-generated outputs for
each input of the test dataset, collect it for manual
analysis
'''

import torch
torch.manual_seed(123)

from dataset_tuning import test_data
from stylize_prompts import format_input

'''
# check what is in test data
for entry in test_data:
    print(entry)
    input()
'''

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/preloaded_weights/openai/scripts') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

from textgenerate import generate, text_to_token_ids, token_ids_to_text
from GPTModel import GPTModel

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

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
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

'''
Load the instruction fine-tuned  model (from disk) for validating how good its inference is
'''

model = GPTModel(BASE_CONFIG)
model.eval()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
 
model.to(device)

try:
    import time
    start_time = time.time()

    model.load_state_dict(torch.load("./model/modelif.pth", map_location=device))

    end_time = time.time()
    load_time_minutes = (end_time - start_time) / 60
    print(f"loaded saved model (1.6G) in {load_time_minutes:.2f} minutes .. <enter>")
    input()

except FileNotFoundError:
    print("model not found on disk. please point to location, or train from scratch for instruction-follow")
    exit()

'''
Generate instruction-follow text and compare manually first with the 
supervised response, using first few samples of the test dataset
'''

#for entry in test_data[:5]:    # iterate over 1st 5 samples
for entry in test_data:    # iterate over all samples

    exec_start_time = time.time()
    input_text = format_input(entry) # this does not include the output 

    # use the model to infer (predict / write) out the response
    token_ids = generate(
        model = model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens = 256,
        context_size = BASE_CONFIG["context_length"],
        eos_id = 50256
    )

    model_generated_text = token_ids_to_text(token_ids, tokenizer)

    exec_end_time = time.time()
    exec_time = (exec_end_time - exec_start_time) / 60

    response_text = (
        model_generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    # actual response given during training is entry['output']
    # let's manually compare

    print(f"Response generated in {exec_time:.2f} minutes .. <enter>")
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
    input() #let it flow

