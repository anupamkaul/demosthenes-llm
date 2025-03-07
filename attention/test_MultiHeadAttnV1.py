import torch
import MultiHeadAttn_StackedCausalSA as MhaAttnWrapper

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
print(batch.shape)
print(batch)

torch.manual_seed(123)
context_length = batch.shape[1] # this is the number of tokens
d_in, d_out = 3, 2

mha = MhaAttnWrapper.MultiHeadAttentionWrapper(
    d_in, d_out, context_length, dropout=0.2, num_heads=2
)

context_vecs = mha(batch)

print("context vecs: \n", context_vecs)
print("context vecs shape: ", context_vecs.shape)
