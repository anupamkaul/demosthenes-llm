import torch
torch.manual_seed(123)

from stylize_prompts import format_input
from dataset_tuning import val_data

input_text = format_input(val_data[0])
print(input_text)

'''
Next we generate the model’s response using 
the same generate function we used to pretrain 
the model in 'pretraining'
'''

import time
start_time = time.time()

# instantiate the pretrained model (from ./model.py) 
from model import model, BASE_CONFIG

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"model loaded (including finding on disk) in {execution_time_minutes:.2f} minutes.")

import os, sys

# the right textgenerate module that contains generate..
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '.././pretraining/preloaded_weights/openai/scripts/'))

from textgenerate import text_to_token_ids, token_ids_to_text, generate

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

'''
The generate function returns the combined input and output text. 
This behavior was previously convenient since pretrained LLMs are 
primarily designed as text-completion models, where the input and 
output are concatenated to create coherent and legible text. 

However, when evaluating the model’s performance on a specific task, 
we often want to focus solely on the model’s generated response.

To isolate the model’s response text, we need to subtract the length 
of the input instruction from the start of the generated_text:

'''

response_text = generated_text[len(input_text):].strip()
print(response_text)

