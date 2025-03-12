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

# we can use -1 to infer dimension like so:

z = x.view(2, -1, 2)
print(z)

'''
tensor([[[ 1,  2],
         [ 3,  4],
         [ 5,  6],
         [ 7,  8]],

        [[ 9, 10],
         [11, 12],
         [13, 14],
         [15, 16]]])
'''

# Incorrect usage (will raise an error)
#w = x.view(3, 3)
#print(w)

# RuntimeError: shape '[3, 3]' is invalid for input of size 16

# Modifying a view also modifies the original tensor as it is in-situ

b = x.view(4, 4)
print(b)

b[0, 0] = 101

print(x)
print(y) # was also a 4 * 4 view of x

'''
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]])
tensor([101,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
         15,  16])
tensor([[101,   2,   3,   4],
        [  5,   6,   7,   8],
        [  9,  10,  11,  12],
        [ 13,  14,  15,  16]])
'''





