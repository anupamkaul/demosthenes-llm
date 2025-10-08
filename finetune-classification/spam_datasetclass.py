import torch
from torch.utils.data import Dataset

import pandas as pd

class SpamDataset(Dataset):

   def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
       self.data = pd.read_csv(csv_file)
 
       # pretokenize the text
       self.encoded_texts = [
          tokenizer.encode(text) for text in self.data["Text"]
       ]

       # apply the padding strategy (as discussed in spam-dataloader.py)
       if max_length is None:
          self.max_length = self._longest_encoded_length() # from Dataset
       else:
          self.max_length = max_length # provide an override option to choose a max length

       #truncate sequences if they are longer than max_length
    
       self.encoded_texts = [
          encoded_text[:self.max_length]
          for encoded_text in self.encoded_texts
       ]

       self.encoded_texts = [
          encoded_text + [pad_token_id] * 
          (self.max_length - len(encoded_text))
          for encoded_text in self.encoded_texts
       ]

   # getter functions:

   def __getitem__(self, index):
      encoded = self.encoded_texts[index]
      label = self.data.iloc[index]["Label"]
      return (
         torch.tensor(encoded, dtype=torch.long),
         torch.tensor(label, dtype=torch.long)
      )

   def __len__(self):
      return len(self.data)

   def _longest_encoded_length(self):
      max_length = 0
      for encoded_text in self.encoded_texts:
          encoded_length = len(encoded_text)
          if encoded_length > max_length:
             max_length = encoded_length
      return max_length

    
