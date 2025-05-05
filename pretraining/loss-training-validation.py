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

file_path = "the-verdict.txt"
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

print("raw training data: \n", train_data)

'''
Now we use dataloaders (from my code) and create 
a train_dataloader and a validation_loader..
'''








