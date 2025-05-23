import torch
import MultiHeadAttention as MhaAttn

inputs = torch.tensor(
    [ [0.43, 0.15, 0.89],    # Your      (x^1)
      [0.55, 0.87, 0.66],    # journey   (x^2) 
      [0.57, 0.85, 0.64],    # starts    (x^3)
      [0.22, 0.58, 0.33],    # with      (x^4) 
      [0.77, 0.25, 0.10],    # one       (x^5)
      [0.05, 0.80, 0.55]     # step      (x^6)
    ]
)

'''
for causal and multi headed self attention we assume code can handle batches of input data streams
'''

batch = torch.stack((inputs, inputs), dim=0)
print("batch shape: ", batch.shape, "\n")
print("batch: \n", batch)

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MhaAttn.MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads = 2)
context_vec = mha(batch)  # calls the forward method of MultiHeadAttention class (MultiHeadAttention.py)
print("context vecs:\n", context_vec)

