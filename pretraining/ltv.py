'''
Lets now prep up the infra for writing up the loss function
and dividing the data set into training and validation sizes

previously in eval.. we looked at cross-entropy as the loss
function (negative average log probability difference between target and current iteration)

We apply the loss calculation to the entire dataset

The dataset size:
1. Start with "The Verdict"
2. Explore Project Gutenberg
3. Scale with Llama like LLMs (7B parameter), and go to 175B, 1T etc
'''

import torch
import tiktoken

#file_path = "the-verdict.txt"

#file_path = "./datasets/the-verdict.txt"
#file_path = "./datasets/wikipedia_corpus.txt"

#file_path = "../datasets/wikipedia_corpus.txt"
file_path = "../datasets/wikipedia_corpus_small.txt"

with open (file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

import tiktoken as tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

total_characters = len(text_data)
total_tokens     = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

'''
Characters: 20479
Tokens: 5145

5145 tokens : text may be too small to train an LLM but we could run the code in min
instead of weeks. Plus later we will shortcut with loading open pre-trained weights
'''

'''
Now we divide the dataset into training and validation set portions. 
(90% for training, 10% for validation)
'''

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
print("90% of the split is from index ", split_idx)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# comment this if we print wikipedia_corpus, just too big..
# print("raw training data: \n", train_data)

'''
Now we use dataloaders (from my code) and create 
a train_dataloader and a validation_loader..

See training-validation-dataset.png

When preparing the data loaders, we split the input text into training and validation set portions. 
Then we tokenize the text (only shown for the training set portion for simplicity) and divide the tokenized 
text into chunks of a user-specified length (here, 6). Finally, we shuffle the rows and organize the chunked 
text into batches (here, batch size 2), which we can use for model training.We are training the model with 
training data presented in similarly sized chunks for simplicity and efficiency. However, in practice, it can 
also be beneficial to train an LLM with variable-length inputs to help the LLM to better generalize across different 
types of inputs when it is being used.
'''

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

import sys, os 

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '../tokenizers/')

# add new path to sys.path (what python uses to search imported modules)
sys.path.append(module_path)

import dataloaderV1 as dataloader

# add flexibility for different batch size computes
import platform
if (platform.system() != "Darwin"):
    batch_size=2
else:
    batch_size=8

print("\nOS: ", platform.system(), " batch size: ", batch_size, " enter..")
input()

train_loader = dataloader.create_dataloader_v1(
    train_data,
    batch_size=batch_size,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = dataloader.create_dataloader_v1(
    val_data,
    batch_size=batch_size,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# train_data and val_data are not encoded..
#print("Validation data: \n", val_data)

'''

We can see the shapes are divided roughly into 9 chunks for the training
data and 1 chunk for validation as we want:

Train loader:
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])
torch.Size([2, 256]) torch.Size([2, 256])

Validation loader:
torch.Size([2, 256]) torch.Size([2, 256])

'''

'''
Implement a utility function to calculate the cross entropy loss
of a given batch returned via the training and validation loader
'''

def calc_loss_batch(input_batch, target_batch, model, device):

    input_batch =  input_batch.to(device) # optimize
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss

'''
We now use the above loss utility function and implement the following
calc_loss_loader function that computes the loss over all the batches
that are sampled by a given data loader
'''

def calc_loss_loader(data_loader, model, device, num_batches=None):

    print("calc_loss_loader..")
    total_loss = 0

    # sanity checks
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    print("num_batches: ", num_batches, "len of data_loader: ", len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):

        # single line printing to show progress
        print(f"\r{i} of {len(data_loader)}", end="", flush=True)

        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)    
            total_loss += loss.item() # sum the loss for each batch
        else:
            break

    print("finish calc_loss_loader..")
    # return the average loss over all batches
    return total_loss / num_batches 


# now check the calc_loss_loader function in action, applying it to the training
# and validation set loaders

import sys, os
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)),  '../llm-infra/') )

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
    
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss   = calc_loss_loader(val_loader,   model, device)  

print("Training loss: ", train_loss)
print("Validation loss: ", val_loss)

'''
Training loss:  10.987583584255642
Validation loss:  10.98110580444336

The loss is high (and not zero) since the model hasn't been trained yet

'''
