'''

Thus far, in spam-dataset.py we have downloaded the dataset, balanced it, and split it 
into training and evaluation subsets. Now we will set up the PyTorch data loaders that 
will be used to train the model.

Data loaders will be similar to what I used for text data. There I had used sliding window
concept to generate uniformly sized chunks and I had created batches for efficient model 
training. Each such chunk functioned as an individual training instance. 

In the SMS spam data set the messages are all of varying lengths. To get uniform size I can
either shorten to the shortest or pad to the longest. I will go with the latter option to not
lose accuracy. To pad, I will repeatedly use "<|endoftext|>" markers as padding tokens. 

The padding will be applied to the encoded text, like so:

'''

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# token is [50256]

import spam_datasetclass
from spam_datasetclass import SpamDataset

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

print(train_dataset.max_length)
#120 is the length, common for most sms

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
print(val_dataset.max_length)

test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
print(test_dataset.max_length)

# now that the classes are instantiated and the reformatting (padding) of the 
# 3 datasets has happened, we now create the training, validation and test
# data Loaders that load the text messages in batches of size 8

from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

import torch
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# to test the above code and check that batches of expected size are created,
# we can iterate over the training_loader and print the tensor dimensions of 
# the last batch (note that iterating will involve a for loop with a NOP)

for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

'''
Input batch dimensions: torch.Size([8, 120])
Label batch dimensions torch.Size([8])

As we can see, input batches (each batch) consists of 8 training
samples of size 120 each. The label tensor stores the class labels (0, 1 here : ham, spam)
to the 8 training samples (classifications for training)`

Lets also check how many batches are there in each dataset (training, validation and test)

'''

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

'''
130 training batches
19 validation batches
38 test batches
'''

# now that we have prepared the data, we need to prepare the model for fine tuning


