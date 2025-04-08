'''
We prepare the input data and initialize a new GPT model to check its usage.
Lets tokenize a batch consisting of 2 text inputs for the GPT model, using the tiktoken tokenizer
(used in tokenizers folder)
'''

import torch
import tiktoken
import DummyGPTModel as gpt
import GPT_CONFIG_124M as gpt2_cfg

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
#print(batch)

batch = torch.stack(batch, dim=0)
print(batch)

'''
tensor([[6109, 3626, 6100,  345],   # first text tokenized for gpt2
        [6109, 1110, 6622,  257]])  # second text tokenized for gpt2

Next initialize a new 124-million paramenter DummyGPTModel instance
and feed it this tokenized batch
'''

torch.manual_seed(123)
model = gpt.DummyGPTModel(gpt2_cfg.get_GPT_CONFIG_124M())
logits = model(batch)

print("Output share: ", logits.shape)
print(logits)

'''
Outputs (usually referred to as logits) are as follows:

Output share:  torch.Size([2, 4, 50257])

(2 rows, each for 1 text. 4 tokens per text (4 words in each text), and each token is a 50257 dim vector
which matches the size of the tokenizer's vocab)

tensor([[[-0.9289,  0.2748, -0.7557,  ..., -1.6070,  0.2702, -0.5888],
         [-0.4476,  0.1726,  0.5354,  ..., -0.3932,  1.5285,  0.8557],
         [ 0.5680,  1.6053, -0.2155,  ...,  1.1624,  0.1380,  0.7425],
         [ 0.0447,  2.4787, -0.8843,  ...,  1.3219, -0.0864, -0.5856]],

        [[-1.5474, -0.0542, -1.0571,  ..., -1.8061, -0.4494, -0.6747],
         [-0.8422,  0.8243, -0.1098,  ..., -0.1434,  0.2079,  1.2046],
         [ 0.1355,  1.1858, -0.1453,  ...,  0.0869, -0.1590,  0.1552],
         [ 0.1666, -0.8138,  0.2307,  ...,  2.5035, -0.3055, -0.3083]]],
       grad_fn=<UnsafeViewBackward0>)

'''


