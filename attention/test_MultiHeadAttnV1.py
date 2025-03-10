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
    #d_in, d_out, context_length, dropout=0.2, num_heads=2
    #d_in, 1, context_length, dropout=0.2, num_heads=2
    #d_in, 3, context_length, dropout=0.2, num_heads=2
    #d_in, 3, context_length, dropout=0.2, num_heads=1
    #d_in, 10, context_length, dropout=0.2, num_heads=2
    #d_in, d_out, context_length, dropout=0.2, num_heads=12
    #d_in, d_out, context_length, dropout=0.2, num_heads=25
    d_in, 1, context_length, dropout=0.2, num_heads=25  
)

context_vecs = mha(batch)

print("context vecs: \n", context_vecs)
print("context vecs shape: ", context_vecs.shape)

'''

Note: true that the last dim of context_vec is num_heads * 2 (see num_heads=25 example) -- not entirely true, only because d_out=2
Also true that if I move d_out from 2 to 1, the context_vec dimension reduces by half and doesn't depend on num_heads anymore

There seems to be a relationship between the last dim of context_vector and its f(d_out, num_heads)
f(2, 2) = 4
f(1, 2) = 2
f(3, 2) = 6
f(10,2) = 20
f(2, 12) = 24 etc

correction to above : last dim of context_vector is NOT num_heads * 2, it is num_heads * d_out 
This explains the function relationship above

'''
