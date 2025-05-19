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

            # target_check is 1 element ahead..
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

