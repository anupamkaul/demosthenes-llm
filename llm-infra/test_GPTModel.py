'''
Let’s now initialize the 124-million-parameter GPT model using the GPT_CONFIG_ 124M dictionary 
we pass into the cfg parameter and feed it with the batch text input we previously created:
'''

'''
We prepare the input data and initialize a new GPT model to check its usage.
Lets tokenize a batch consisting of 2 text inputs for the GPT model, using the tiktoken tokenizer
(used in tokenizers folder)
'''

import torch
import tiktoken
import GPTModel as gpt
import GPT_CONFIG_124M as gpt2_cfg

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
#print(batch)

batch = torch.stack(batch, dim=0)
print(batch)

'''
tensor([[6109, 3626, 6100,  345],   # first text tokenized for gpt2
        [6109, 1110, 6622,  257]])  # second text tokenized for gpt2

Next initialize a new 124-million paramenter DummyGPTModel instance
and feed it this tokenized batch
'''

torch.manual_seed(123)
model = gpt.GPTModel(gpt2_cfg.get_GPT_CONFIG_124M())
logits = model(batch)

print("Output shape: ", logits.shape)
print(logits)
