'''
From sliding_window (see input-target pairs used for training LLMs): 

before we turn tokens into embeddings:
implement an efficient data loader that iterates over the input dataset
and returns the inputs and targets as PyTorch tensors (multi-dim arrays).

We return two tensors: an input tensor containing the  text that the LLM
sees and a target tensor that includes the target for the LLM to predict

The code will operatoe on TokenIDs instead of strings, using the encode/decode
methods of the BPE tokenizer.
'''


import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    
    # use sliding window to chunk the dataset into overlapping sequences of max_length
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire set

        token_ids = tokenizer.encode(txt)  

        # use a sliding window to chunk the text into overlapping 
        # sequences of max_length. Stride controls the overlap amount

        # the samples are of (max-length) where the next sequence moves
        # up by stride. If stride < max-length we have overlaps - this helps
        # control the prediction granularity

        for i in range(0, len(token_ids) - max_length, stride):

            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i+1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # return total number of rows in dataset
    def __len__(self):
        return len(self.input_ids)

    # return a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]          

'''

The following code uses GPTDatasetV1 to load the inputs in batches, via
Pytorch's DataLoader (this encompasses the DatasetV1 class above)

'''

def create_dataloader_v1(txt, batch_size = 4, max_length = 256, 
                         stride = 128, shuffle=True, drop_last=True,
                         num_workers=0):

   tokenizer = tiktoken.get_encoding("gpt2")
   dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=shuffle,
       drop_last=drop_last,
       num_workers=num_workers
   )

   return dataloader


'''
Test the above code. Let's use the dataloader with a batch size of 1
for an LLM with a context size of 4 to develop an intuition of how the
GPTDatasetV1 class and the create_dataloader functions work together
'''

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

# now we convert dataloader into a Python iterator to fetch
# the next entry via Python's built-in next() function

data_iter = iter(dataloader)

first_batch = next(data_iter)
print("batch 1: ", first_batch)

second_batch = next(data_iter)
print("batch 2: ", second_batch)

# let's increase the batch size to 8 and stride to 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)


