'''

The following code uses GPTDatasetV1 to load the inputs in batches, via
Pytorch's DataLoader (this encompasses the DatasetV1 class above)

'''

import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader
import tiktoken

def create_dataloader_v1(txt, batch_size = 4, max_length = 256, 
                         stride = 128, shuffle=True, drop_last=True,
                         num_workers=0):

   tokenizer = tiktoken.get_encoding("gpt2")
   dataset = GPTDatasetV1.GPTDatasetV1(txt, tokenizer, max_length, stride)
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


