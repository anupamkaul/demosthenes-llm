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

Next initialize a new 124-million paramenter GPTModel instance
and feed it this tokenized batch
'''

torch.manual_seed(123)
model = gpt.GPTModel(gpt2_cfg.get_GPT_CONFIG_GPT2_SMALL()) 
logits = model(batch)

print("Output shape: ", logits.shape)
print(logits)

'''
Output shape:  torch.Size([2, 4, 50257])
tensor([[[ 0.1381,  0.0077, -0.1963,  ..., -0.0222, -0.1060,  0.1717],
         [ 0.3865, -0.8408, -0.6564,  ..., -0.5163,  0.2369, -0.3357],
         [ 0.6989, -0.1829, -0.1631,  ...,  0.1472, -0.6504, -0.0056],
         [-0.4290,  0.1669, -0.1258,  ...,  1.1579,  0.5303, -0.5549]],

        [[ 0.1094, -0.2894, -0.1467,  ..., -0.0557,  0.2911, -0.2824],
         [ 0.0882, -0.3552, -0.3527,  ...,  1.2930,  0.0053,  0.1898],
         [ 0.6091,  0.4702, -0.4094,  ...,  0.7688,  0.3787, -0.1974],
         [-0.0612, -0.0737,  0.4751,  ...,  1.2463, -0.3834,  0.0609]]],
       grad_fn=<UnsafeViewBackward0>)

The shape of the output (logits) of the GPT shows the vocab size of the tokenizer.
Next we will convert the output back into text to get the context back in human form.

Let's now analyze the size of the model. This would be very useful for EdgeAI scenarios.

We can use the numel (number of elements) method to collect total number of params (wts)
in the model's parameter tensors so lets start with that

'''
print("\n\nModel Analysis (params and size)\n")

total_params = sum(p.numel() for p in model.parameters()) # we need to sum numel always..
print(f"Total number of parameters: {total_params:,}")

'''
Total number of parameters: 163,009,536 

This is greater than 124M that we started out with. Why?
This is related to "Weight Tying" which was used with GPT2.
Basically GPT2 re-uses the wts from the token embedding layer in its output layer.

'''

print("\tToken embedding layer shape:", model.tok_emb.weight.shape)
print("\tOutput layer shape:", model.out_head.weight.shape)

'''
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])

To test the weight typing theory let's remove weights of the output_layer (out_head)
from the total params and see what we land with:
'''

total_params_gpt2 = { total_params - sum(p.numel() for p in model.out_head.parameters()) }

print("Number of trainable parameters considering weight tying: ", total_params_gpt2)
#print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,"})

'''
Number of trainable parameters considering weight tying:  {124412160}

debug the next section hence commented: not printing out total_params_gpt2 as an f-string:

# for better formatting:
#print(f"Total number of parameters: {total_params:,}")
print(f"Total number of parameters: {total_params_gpt2:,}")

print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}"
      )

In Python, the print(f"...") syntax, available from Python 3.6 onwards, utilizes f-strings (formatted string literals) 
to embed expressions inside string literals for formatting. The f before the opening quote signifies that it's an f-string. 
Inside the string, expressions enclosed in curly braces {} are evaluated and their values are inserted into the string.
'''

'''
Continue: 

Let's also find number of params in the multihead attention module a
and in the feed forward network and compare/contrast them against the 124M params total
'''

params_out_head = sum(p.numel() for p in model.out_head.parameters()) 
#print("params of the out_head piece of the model: ", params_out_head)
print(f"params of the out_head piece of the model: {params_out_head:,}")

'''
params of the out_head piece of the model:  38597376 
'''

params_final_norm = sum(p.numel() for p in model.final_norm.parameters()) 
print("params of the final_norm piece of the model: ", params_final_norm)

'''
Too less so probably incorrect calculation:
params of the final_norm piece of the model:  1536 

(interestingly that is 768 * 2 (or the emb_dim value coming into the LayerNorm class, so these could
be the "scale" and the "shift" values of LayerNorm, each with 768 wts)

'''

'''
Now that I have been able to calculate parameter values of LayerNorm, MultiHeadAttention, FeedForward
and the TransformerBlock (single unit) -- see the associated class tests - let's figure out the memory
needed for the GPT module based on 163,009,536 params (including weight tying of token to output layer)
'''

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB") # and now I understand what the post :, was, it was the formating :) 

'''
Model Analysis (params and size)

Total number of parameters: 163,009,536
	Token embedding layer shape: torch.Size([50257, 768])
	Output layer shape: torch.Size([50257, 768])
Number of trainable parameters considering weight tying:  {124412160}
params of the out_head piece of the model: 38,597,376
params of the final_norm piece of the model:  1536
Total size of the model: 621.83 MB
'''

