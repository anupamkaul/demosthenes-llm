import torch

'''

Testing out some of the batch multiplication with transpose that MultiHeadAttention.py implements
(this is employed in MultiHeadAttention.py)

To illustrate batched matrix multiplication, suppose we have the following tensor:

'''
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],    # assume the 2, 3 cols (last 2) stand for num_tokens and head_dim
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print ("\n Testing batched multiplication with transpose\n")
print("tensor a : \n", a)
print("tensor a multiplied with its transpose by (2,3): \n", a @ a.transpose(2, 3))

'''
tensor([[[[1.3208, 1.1631, 1.2879],
          [1.1631, 2.2150, 1.8424],
          [1.2879, 1.8424, 2.0402]],

         [[0.4391, 0.7003, 0.5903],
          [0.7003, 1.3737, 1.0620],
          [0.5903, 1.0620, 0.9912]]]])

In this case, the matrix multiplication implementation in PyTorch handles the four-dimensional input tensor 
so that the matrix multiplication is carried out between the two last dimensions (num_tokens, head_dim)
and then repeated for the individual heads. Another way of doing the same thing is:

'''

first_head = a[0, 0, :, :] # the 2, 3 of the transpose
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_head = second_head @ second_head.T
print("\nSecond head:\n", second_head)

'''

First head:
 tensor([[1.3208, 1.1631, 1.2879],
        [1.1631, 2.2150, 1.8424],
        [1.2879, 1.8424, 2.0402]])

Second head:
 tensor([[0.4391, 0.7003, 0.5903],
        [0.7003, 1.3737, 1.0620],
        [0.5903, 1.0620, 0.9912]])

The results are exactly the same as before, so the transpose(2, 3) saves on the 
number of multiplications and achieves the same results, hence its called batched multiplication

'''

