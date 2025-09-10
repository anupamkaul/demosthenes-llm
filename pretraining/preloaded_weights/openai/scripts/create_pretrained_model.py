from gpt_download import download_and_load_gpt2

# download_and_load_gpt2 loads all open weights from openAI/GPT2 into ram (python)
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# settings and params are dictionaries returned from the download_and_load_gpt2 function
# lets inspect settings and parameter dictionary keys

print("Settings: ", settings)
print("Params: ", params.keys()) # interesting insights to how weights are organized

#print(params) # all params (big data of all weights)

print("The weights of the token embedding layers are:", params["wte"])
print("The dims are:", params["wte"].shape)

# now that the params (weights) and settings are in RAM, I need to move
# them to my GPTModel instance for demosthenes. 

# Start by creating a config that shows differences between different 
# GPT model sizes (see ../gpt2-arch.png) (I can re-use this config to 
# load up weights of different versions of gpt2)

model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# now define and configure my previous GPTmodel class

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../../../../llm-infra/') )

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

# create a new map (NEW_CONFIG). These are the initialization parameters for my GPTModel class' architecture

# Note that copy() and update() are python primitives for dictionaries
# copy() creates a shallow copy without touching the previous one. update is used to merge key-value pairs
# from another dictionary (see dictionary_copy_update.py)

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

# we used a 256-token length earlier, but the original GPT2 models from openAI were trained
# with 1024-token length, so we update NEW_CONFIG accordingly

NEW_CONFIG.update({"context_length": 1024})

# Also, OpenAI used bias vectors in the multi-head attention module’s linear layers to implement 
# the query, key, and value matrix computations. Bias vectors are not commonly used in LLMs anymore 
# as they don’t improve the modeling performance and are thus unnecessary. However, since we are working 
# with pretrained weights, we need to match the settings for consistency and enable these bias vectors:

NEW_CONFIG.update({"qkv_bias": True})

print("\nNEW_CONFIG: ", NEW_CONFIG)

# instantiate my architecture with configs from the imported weights
gpt = GPTModel(NEW_CONFIG) 
gpt.eval()
print(gpt)

# now, the above model is initialized with random weights. We're now going to override those random
# weights with the weights loaded in the 'param' dictionary above that contains all the open weights
# in RAM

# reverse engineering : define a small assign utility function that checks whether two tensors or arrays
# (left and right) have the same dimensions or shape, and return the right tensor as trainable pytorch parameters

import torch

def assign(left, right):

    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

# next, define a load_weigths_into_gpt function that loads the weights from the params dictionary
# into my GPTModel instance (gpt)

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight  = assign(gpt.out_head.weight, params["wte"])

'''
In the load_weights_into_gpt function, we carefully match the weights from OpenAI’s implementation 
with our GPTModel implementation. To pick a specific example, OpenAI stored the weight tensor for 
the output projection layer for the first transformer block as params["blocks"][0]["attn"]["c_proj"]["w"]. 
In our implementation, this weight tensor corresponds to gpt.trf_blocks[b].att.out_proj .weight, where gpt 
is a GPTModel instance.

Developing the load_weights_into_gpt function took a lot of guesswork since OpenAI used a slightly different 
naming convention from ours. However, the assign function would alert us if we try to match two tensors with 
different dimensions. Also, if we made a mistake in this function, we would notice this, as the resulting GPT 
model would be unable to produce coherent text.

Let’s now try the load_weights_into_gpt out in practice and load the OpenAI model weights into our GPTModel 
instance gpt:
'''


load_weights_into_gpt(gpt, params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
gpt.to(device)

print("my model has been loaded with pretrained weights ! \n")

# save the model before we return
torch.save(gpt.state_dict(), "./model/model.pth")       
print("model saved\n")









