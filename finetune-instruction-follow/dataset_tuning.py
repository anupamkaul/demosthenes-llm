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

