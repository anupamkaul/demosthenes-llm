'''

prepare the actual openAI GPT2 wt trained demosthenes model now.
next, replace out_head to classify only 2 output variables that
are required for classification finetuning

First, some choice configurations

'''

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

'''
Next, we import the download_and_load_gpt2 function from gpt_download.py and 
we reuse the GPTModel class (demosthenes) and load_weights_into_gpt function
from pretraining to download the weights into the GPT model. A hackier way is
also to simply read the model that I stored to the disk but I will skip that
for now and re-create the pretrained demosthenes model now

Starting point is ../pretraining/preloaded_weights/openai/scripts
'''

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/preloaded_weights/openai/scripts') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

from gpt_download import download_and_load_gpt2
from load_wts_to_gpt import load_weights_into_gpt
from GPTModel import GPTModel

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
print("model with loaded weights is ready")

'''
After loading the model weights into the GPTModel, we reuse the text generation utility function 
from previous work to ensure that the model generates coherent text:
'''

from textgenerate import text_to_token_ids, generate_text_simple, token_ids_to_text, generate

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

# now check if classification prompts work (they won't as we
# haven't fine-tuned the model for classification prompts yet)

print("\n\nnow checking for classification prompts:")

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

'''
To prep the model for classification fine tuning, we replace the 
original output layer, which maps the hidden representation of 768
nodes to 50,257 (the token vocabulary) to 2 classes (0 - not spam, and
1 - spam). We use the same model as above which is pretrained, except
that we will replace the output layer and then train this model such
that only the edge most nodes of the outermost layer that we have replaced,
are tuned, thus achieving "fine tuning" of this model geared towards
classification.

Fine-tuning selected layers vs. all layers

Since we start with a pretrained model, it’s not necessary to fine-tune 
all model layers. In neural network-based language models, the lower layers 
generally capture basic language structures and semantics applicable across 
a wide range of tasks and datasets. So, fine-tuning only the last layers 
(i.e., layers near the output), which are more specific to nuanced linguistic 
patterns and task-specific features, is often sufficient to adapt the model to 
new tasks. A nice side effect is that it is computationally more efficient to 
fine-tune only a small number of layers.

'''

print("\n", model)

# first, freeze the model, meaning that we make all layers non-trainable

for param in model.parameters():
    param.requires_grad = False

# next, we replace the output layer (model.out_head -- see logs.txt) which
# originally maps to the size of the vocab, to 2

import torch
torch.manual_seed(123)
num_classes = 2

model.out_head = torch.nn.Linear(
    in_features  = BASE_CONFIG["emb_dim"],  # 768, as before
    out_features = num_classes              # 2 now
)

# note that this new model_out.head has requires_grad set to True by default
# so this will be the only layer in the model that will be updated during
# training. We also configure the last transformer block (accessed as -1) 
# and the final LayerNorm module, which connects this block to the output layer,
# to be trainable.

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

print(model)

'''
above shows that model.out_head maps to 2. At this time requires_grad field for
params is false for every layer except the last trf and layer_norm layers and the
final output block.
'''

'''
Lets now check that freezing the model doesn't impact the model's operations, i.e.
I should still be able to feed the model text, and grab out all of the output layers
with the difference being that the final tensor dims should be 2 instead of 50247
'''

print("\ncheck that the frozen model (except the final layers) is still able to operate on text as expected\n")

inputs = tokenizer.encode("Do you have time")	
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Input dims:", inputs.shape)    # shape(batch, size, num_tokens)

'''
The print output shows that the preceding code encodes the inputs into a tensor consisting of four input tokens:

Inputs: tensor([[5211,  345,  423,  640]])
Inputs dimensions: torch.Size([1, 4])

'''

# now we pass the encoded text into the frozen model and get the output tensors

with torch.no_grad():
    output = model(inputs)

print("Output: ", output)
print("Output dims", output.shape)    # shape(batch, size, num_tokens)

'''
The output tensor looks like the following:

Outputs:
 tensor([[[-1.5854,  0.9904],
          [-3.7235,  7.4548],
          [-2.2661,  6.6049],
          [-3.5983,  3.9902]]])
Outputs dimensions: torch.Size([1, 4, 2])

A similar input would have previously produced an output tensor of [1, 4, 50257], where 50257 
represents the vocabulary size. The number of output rows corresponds to the number of input tokens 
(in this case, four). However, each output’s embedding dimension (the number of columns) is now 2 
instead of 50,257 since we replaced the output layer of the model.
'''

