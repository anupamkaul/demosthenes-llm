'''

dot products are the first line of calculations to get vector affinities (similarity)
The higher the dot product of 2 multi-dim tensors, the similar they are to each other

In the context of LLMs the "amount of (self) attention" is basically governed by the dot
product of 2 entities

A dot product is essentially a concise way of multiplying two vector element-wise and
then summing up the products

'''

import torch

# let the following tensor represet a 3-dim embedding
# of tokens of the sentence "Your journey starts with 1 step" :

inputs = torch.tensor(
    [ [0.43, 0.15, 0.89],    # Your      (x^1)
      [0.55, 0.87, 0.66],    # journey   (x^2) 
      [0.57, 0.85, 0.64],    # starts    (x^3)
      [0.22, 0.58, 0.33],    # with      (x^4) 
      [0.77, 0.25, 0.10],    # one       (x^5)
      [0.05, 0.80, 0.55]     # step      (x^6)
    ]
)

print("inputs : \n", inputs, "\n")

# normally we calculate intermediate attention scores between a query token (of the input sentence)
# with each input token. We determine these scores by computing the dot product of the query (x^2 here)
# with every other input token

query = inputs[1]  # first element is inputs[0]
#print(query)

# the second input token (represented as embeddings) serves as the query

# attention scores of query (x^2) would be:
attn_scores_2 = torch.empty(inputs.shape[0])  # initialize with the same shape as each element of inputs

print("calculate attn_scores of input token 2:")

for i,x_i in enumerate(inputs):
    print("i=", i, " x_i = ", x_i)
    attn_scores_2[i] = torch.dot(x_i, query)

print("\nattn score of query element 2 is the dot product of itself with each input token shown below : \n", attn_scores_2, "\n")

# here is how I validate what exactly does a dot product do:
# see if my hypothesis for a dot product's result matches tensor.dot's definition

print("validate what a dot product does - say dot multiply input[0] and input[1] (where input[1] is the query token):\n")

result = 0
for idx, element in enumerate(inputs[0]) :
    print("idx = ", idx, " element = ", element)
    print("mult (and keep adding) ", inputs[0][idx], " and ", query[idx], "\n")
    result += inputs[0] [idx] * query[idx]

print("my result is", result)
print("torch.dot is", torch.dot(inputs[0], query))
