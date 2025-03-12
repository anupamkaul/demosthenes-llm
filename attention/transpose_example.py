'''
In PyTorch, the transpose operator is used to swap or reverse the dimensions of a tensor. There are a few ways to achieve this:

torch.transpose(input, dim0, dim1):
This function takes a tensor and two dimensions as input and returns a new tensor with those dimensions swapped.

tensor.transpose(dim0, dim1):
This is a method of the torch.Tensor class that performs the same operation as torch.transpose().

tensor.T:
This method is a shorthand for transposing a 2D tensor (matrix). It's equivalent to tensor.transpose(0, 1). For tensors with more than two dimensions, using .T is deprecated and will raise an error in future versions.

torch.t(input):
This function is similar to tensor.T and is used for transposing 2D tensors. It also returns the tensor as is if it is 0D or 1D.

'''

import torch

# create a 2D tensor
x = torch.randn(2, 3)
print(x)

y = torch.transpose(x, 0, 1)
print(y)

'''
tensor([[ 0.7026,  0.0687,  2.0409],
        [ 0.8758, -1.7972, -0.9983]])
tensor([[ 0.7026,  0.8758],
        [ 0.0687, -1.7972],
        [ 2.0409, -0.9983]])
'''

# create a 3D tensor
a = torch.randn(2, 3, 4)
b = torch.transpose(a, 1, 2)
print("tensor a:\n", a, "\n")
print("transposed tensor b:\n", b, "\n")

print("shape of a: ", a.shape)
print("shape of b: ", b.shape)

'''
tensor a:
 tensor([[[ 0.7553,  0.0702, -0.3257,  1.1624],
         [-0.6228,  0.3602,  0.0982, -1.0431],
         [-1.0921,  2.1558, -2.6535,  0.3086]],

        [[ 1.7854,  0.3541,  0.2401,  0.5232],
         [-0.4436,  0.6643, -0.7571,  1.4012],
         [-0.5562, -0.3864, -1.6049,  1.2832]]]) 

transposed tensor b:
 tensor([[[ 0.7553, -0.6228, -1.0921],
         [ 0.0702,  0.3602,  2.1558],
         [-0.3257,  0.0982, -2.6535],
         [ 1.1624, -1.0431,  0.3086]],

        [[ 1.7854, -0.4436, -0.5562],
         [ 0.3541,  0.6643, -0.3864],
         [ 0.2401, -0.7571, -1.6049],
         [ 0.5232,  1.4012,  1.2832]]]) 

shape of a:  torch.Size([2, 3, 4])
shape of b:  torch.Size([2, 4, 3])
'''

# above showed torch.transpose. Now some examples with tensor.transpose
print(x, "\n")

z = x.transpose(0, 1)
print(z, "\n")

w = x.T
print(w, "\n")

v = torch.t(x)
print(v, "\n")

'''
tensor([[-0.5430,  1.4257,  0.8486],
        [ 0.1295, -0.6650, -0.0504]]) 

tensor([[-0.5430,  0.1295],
        [ 1.4257, -0.6650],
        [ 0.8486, -0.0504]]) 

tensor([[-0.5430,  0.1295],
        [ 1.4257, -0.6650],
        [ 0.8486, -0.0504]]) 

tensor([[-0.5430,  0.1295],
        [ 1.4257, -0.6650],
        [ 0.8486, -0.0504]]) 
'''
