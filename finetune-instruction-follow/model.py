'''

This is where I prepare the model for instruction fine-tuning training
for demosthenes. I'll use all the data preparation from dataset_tuning.py
For the fine tuning process, I will reuse the loss calculation code that 
I wrote up in the "pretraining" folder. Heck, I will even use the same
train_model_simple training code that I used in pretraining demosthenes !
(But the train_loader and dataset_finetuning will be from here)

Before beginning instruction fine-tuning, we must first load a pretrained GPT 
model that we want to fine-tune (see figure 7.15), a process we have undertaken 
previously. However, instead of using the smallest 124-million-parameter model 
as before, we load the medium-sized model with 355 million parameters. 

The reason for this choice is that the 124-million-parameter model is too limited 
in capacity to achieve satisfactory results via instruction fine-tuning. Specifically, 
smaller models lack the necessary capacity to learn and retain the intricate patterns 
and nuanced behaviors required for high-quality instruction-following tasks.

'''

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/preloaded_weights/openai/scripts') )
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

from gpt_download import download_and_load_gpt2
from load_wts_to_gpt import load_weights_into_gpt

from GPTModel import GPTModel

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

settings, params = download_and_load_gpt2(
    model_size=model_size, 
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/') )

#from training import train_model_simple
from utils_loss import calc_loss_loader

# note: "training.py" in ../pretraining has not been well written
# pulls in garbage code from ltv that pulls in verdict.txt
# needs either a re-write or a re-import of train_model_simple into its own container

# note that train_loader etc now comes from ./dataset_tuning.py so all I need is a container
# that contains train_model_simple only (and minimal associated dependencies) 





