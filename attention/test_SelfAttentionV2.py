import torch
import SelfAttentionV2 as SelfAttn

inputs = torch.tensor(
    [ [0.43, 0.15, 0.89],    # Your      (x^1)
      [0.55, 0.87, 0.66],    # journey   (x^2) 
      [0.57, 0.85, 0.64],    # starts    (x^3)
      [0.22, 0.58, 0.33],    # with      (x^4) 
      [0.77, 0.25, 0.10],    # one       (x^5)
      [0.05, 0.80, 0.55]     # step      (x^6)
    ]
)

torch.manual_seed(789)

sa_v2 = SelfAttn.SelfAttention_v2(3, 2)
#sa_v2 = SelfAttn.SelfAttention_v2(3, 3)
#sa_v2 = SelfAttn.SelfAttention_v2(3, 10)

print(sa_v2(inputs)) # this should print out the 6-row tensor of context vectors of the 6 tokens described above






