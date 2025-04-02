import torch
import tiktoken
import LayerNorm as _ln

'''
# this version of batch_example has a d_input type error
tokenizer = tiktoken.get_encoding("gpt2")
batch_example = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch_example.append(torch.tensor(tokenizer.encode(txt1)))
batch_example.append(torch.tensor(tokenizer.encode(txt2)))
batch_example = torch.stack(batch_example, dim=0)
'''

torch.manual_seed(123)
batch_example = torch.randn(2, 5)

print("batch example: \n", batch_example)

ln = _ln.LayerNorm(emb_dim = 5)
out_ln = ln(batch_example)
print("out layer normalized:\n", out_ln)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
