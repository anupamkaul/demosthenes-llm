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
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)


