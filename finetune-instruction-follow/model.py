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
as before, we load the medium-sized model with 355-million-parameters. 

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
#CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size, 
    models_dir="gpt2"
)

'''
Load the existing model for continued training, or load the downloaded
pretrained 355M gpt2 model mapped to demosthenes and train it. Both options
train a pretrained model for instruction fine tuning
'''

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)



# an unfortunate hack for my local ubuntu 22.04 : even when cuda force it to CPU
# (this is because memory requests to my GPU0 exceed its total mem available, needs debugging)

device = torch.device("cpu")
print("device override (for my local ubuntu): ", device)

model = torch.compile(model)
model.to(device)

try:
    model.load_state_dict(torch.load("./model/modelif.pth", map_location=device))
    print("loaded previously saved model to continue training for instruction-follow..<enter>")
    input()

except FileNotFoundError:
    print("model not found on disk. train from scratch for instruction-follow..<enter>")
    input()


'''
Take the above model, take the instruction set data (dataset_finetune.py)
and now apply the train_model_simple framework to finetune/train it based
on the dataset that has been downloaded and formatted. 

We will take the above 355-M pretrained model above and further train it using 
the curently prepared instruction dataset dataset_finetune.py). 
We already did all the hard work when we implemented the instruction dataset 
processing. 

For the fine-tuning process itself, we can reuse the loss calculation and 
training functions implemented in 'pretraining' folder. (pretraining_container)
'''

sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../pretraining/') )

from utils_loss import calc_loss_loader
from training_container import train_model_simple, plot_losses

# (note that 'train_data' references to train_model_simple must flow from _this_ dataset_finetune.py
# hence I containerize train_model_simple)

'''
before we begin the instruction-follow training, let's calculate the 
initial loss of training and validation sets
'''

from dataset_tuning import train_loader, val_loader, train_data, val_data

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

'''
Need to optimize this:
Training loss: 3.9499454498291016
Validation loss: 3.89001145362854

trained for instruction fine-tuning will have lower values 
(0.3 and 0.6 ranges)
'''

'''
With the model and data loaders prepared, we can now proceed to train the model. 
The code below sets up the training process, including initializing the optimizer, 
setting the number of epochs, and defining the evaluation frequency and starting 
context to evaluate generated LLM responses during training based on the first 
validation set instruction (val_data[0]) 
'''

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)

num_epochs = 2
print("Now we fine tune (train) for instruction-follow ! ", num_epochs, " num_epochs ")

from stylize_prompts import format_input

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)



