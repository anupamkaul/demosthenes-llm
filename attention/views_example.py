'''

In PyTorch, the view() operator for tensors is used to reshape a tensor without changing its underlying data. 
It returns a new tensor with the specified shape, sharing the same data as the original tensor. 
This operation is memory-efficient as it avoids data copying, making it suitable for tasks requiring frequent reshaping.

The view() operator takes one or more integer arguments specifying the desired dimensions of the new tensor. 
The total number of elements in the new shape must match the original tensor's element count. 

A special value of -1 can be used for one dimension, allowing PyTorch to infer its size automatically based on the other dimensions and the total number of elements.

'''
import torch

# create a tensor
x = torch.arange(1, 17)
print(x)

# tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])

y = x.view(4, 4)
print(y)

'''
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
'''




