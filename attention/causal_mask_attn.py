'''
Code to implement causal attention, i.e. mask out future tokens, which come after the current token
in the input text. We mask out the attention wts above the diagonal of a token*token 2 dim table, and we 
normalize the nonmasked attn weights such that the attn weights sum to 1 in each row. Later we will implement
this masking and normalization procedure in code.
'''

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

sa_v2 = SelfAttn.SelfAttention_v2(3, 2)

queries = sa_v2.W_query(inputs)
keys    = sa_v2.W_key(inputs)

# step 1 : apply softmax 
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

'''
tensor([[0.1770, 0.1558, 0.1569, 0.1666, 0.1896, 0.1541],
        [0.1772, 0.1554, 0.1566, 0.1666, 0.1902, 0.1538],
        [0.1771, 0.1556, 0.1568, 0.1666, 0.1900, 0.1539],
        [0.1720, 0.1609, 0.1615, 0.1669, 0.1784, 0.1603],
        [0.1725, 0.1617, 0.1623, 0.1659, 0.1782, 0.1594],
        [0.1733, 0.1589, 0.1597, 0.1673, 0.1820, 0.1587]],
       grad_fn=<SoftmaxBackward0>)
'''

# step 2: Mask with 0's above the diagonal
# let's use tril to create a mask where values above the diagonal are zero

context_length = attn_scores.shape[0]
print("context_length = ", context_length)
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

'''
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
'''

masked_simple = attn_weights * mask_simple
print(masked_simple)


'''
tensor([[0.1635, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1631, 0.1612, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1632, 0.1612, 0.1612, 0.0000, 0.0000, 0.0000],
        [0.1646, 0.1651, 0.1649, 0.1697, 0.0000, 0.0000],
        [0.1660, 0.1622, 0.1624, 0.1704, 0.1713, 0.0000],
        [0.1635, 0.1650, 0.1648, 0.1707, 0.1633, 0.1726]],
       grad_fn=<MulBackward0>)

'''

'''
Step 3 is to renormalize the attention weights to sum up to 1 again in each row.
We can achieve this by dividing each element in each row by the sum in each row:
'''

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

'''
Now all the non-zero elements of rows sum up to 1 for that row:
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4989, 0.5011, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3323, 0.3343, 0.3334, 0.0000, 0.0000, 0.0000],
        [0.2512, 0.2498, 0.2496, 0.2494, 0.0000, 0.0000],
        [0.2019, 0.2085, 0.2075, 0.1979, 0.1841, 0.0000],
        [0.1681, 0.1648, 0.1649, 0.1675, 0.1682, 0.1665]],
       grad_fn=<DivBackward0>)
'''


