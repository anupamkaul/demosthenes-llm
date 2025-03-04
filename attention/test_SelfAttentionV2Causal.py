import torch
import SelfAttentionV2Causal as SelfAttn

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

# above results in a 3-dim tensor consisting of 2 input texts with 6 tokens each, where
# each token is a 3-dim embedding vector
# torch.Size([2, 6, 3]) 

torch.manual_seed(123)
context_length = batch.shape[1]

d_in = 3
d_out = 3 # 3 works best here

ca = SelfAttn.SelfAttentionV2Causal(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
print("context vecs:\n", context_vecs)







