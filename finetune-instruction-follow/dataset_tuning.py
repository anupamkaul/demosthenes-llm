'''
Construct the training batches and prepare everything
with the data until the point we need to prepare the 
model and fine tune it (this comes after download_dataset.py
and stylize_prompts.py)

Before we move on to setting up the PyTorch data loaders 
let’s divide the dataset into training, validation, and test sets 
analogous to what we have done with the spam classification dataset. 

'''

from download_dataset import download_and_load_file

file_path = "instruction-data.json"
url =  (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Len of data: ", len(data))

# lets partition training 85%, test 10% and validation as 5%
train_portion = int(len(data) * 0.85)
test_portion  = int(len(data) * 0.1)
val_portion   = len(data) - train_portion - test_portion

train_data    = data[ : train_portion]
test_data     = data[train_portion : train_portion + test_portion]
val_data      = data[train_portion+test_portion : ]

print("Training set length:", len(train_data))
print("Test set length:", len(test_data))
print("Validation set length:", len(val_data))

'''
Next, we will create training batches. 

Previously the training batches were created automatically by the 
PyTorch DataLoader class, which employs a default collate function 
to combine lists of samples into batches. A collate function is 
responsible for taking a list of individual data samples and merging them 
into a single batch that can be processed efficiently by the model during training.

However, the batching process for instruction fine-tuning is a bit more involved 
and requires us to create our own custom collate function that we will later plug 
into the DataLoader. We implement this custom collate function to handle the specific 
requirements and formatting of our instruction fine-tuning dataset.

(see images/custom-collation-training-batches.png)

Let's do this in several steps. First, code an InstructionDataset class that 
applies 'format_input" (stylize_prompts) and "pretokenizes" all inputs in the
dataset, similar to SpamDataset. This is implemented in the __init__ constructor
of the following class:

'''

import torch
from torch.utils.data import Dataset
from stylize_prompts import format_input

class InstructionDataset(Dataset):

    def __init__(self, data, tokenizer):

        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __get_item__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


