'''
Training deep neural networks with many layers can sometimes prove challenging due to problems like Vanishing or Exploding Gradients. 

These problems lead to unstable training dynamics and make it difficult for the network to effectively adjust its weights, which means 
the learning process struggles to find a set of parameters (weights) for the neural network that minimizes the loss function. In other 
words, the network has difficulty learning the underlying patterns in the data to a degree that would allow it to make accurate 
predictions or decisions.

Let’s now implement layer normalization to improve the stability and efficiency of neural network training.
The main idea behind layer normalization is to adjust the activations (outputs) of a neural network layer 
to have a mean of 0 and a variance of 1, also known as unit variance. This adjustment speeds up the convergence 
to effective weights and ensures consistent, reliable training. 

In GPT-2 and modern transformer architectures, layer normalization is typically applied before and after the 
multi-head attention module, and, as we have seen with the DummyLayerNorm placeholder, before the final output layer.
'''

'''

Let's start by creating an example where we implement a neural network wth 5 inputs and 6 outputs, and we apply
2 input examples:
'''

import torch
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
print("batch example: \n", batch_example, "\n")

'''
batch example: 
 tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969],
        [ 0.2093, -0.9724, -0.7550,  0.3239, -0.1085]]) 

We implement a neural network layer with five inputs and six outputs that we apply to two input examples: 
'''

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

'''
The neural network layer we have coded consists of a Linear layer followed by a nonlinear activation function, ReLU 
(short for rectified linear unit), which is a standard activation function in neural networks. ReLU simply thresholds 
negative inputs to 0, ensuring that a layer outputs only positive values, which explains why the resulting layer output 
does not contain any negative values. Later, we will use another, more sophisticated activation function in GPT.
'''

out = layer(batch_example)
print("out: \n", out)

'''
out: 
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)

As we can see the out does not have a mean of 0 and variance of 1.
This will cause problems down the road when backprop is applied (ReLU)
and gradients (activations) will start diminishing or vanishing, meaning
that training will not converge.

Let's actually calculate the mean and variance and confirm:
'''

mean = out.mean(dim=-1, keepdim=True)
var  = out.var (dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)

'''
Mean:
 tensor([[0.1324],                          # mean for 1st row
        [0.2170]], grad_fn=<MeanBackward1>) # mean for 2nd row
Variance:
 tensor([[0.0231],                          # variance for 1st row
        [0.0398]], grad_fn=<VarBackward0>)  # variance for 2nd row
'''

'''
Next, let’s apply layer normalization to the layer outputs we obtained earlier.
The operation consists of subtracting the mean and dividing by the square root 
of the variance (also known as the standard deviation):
'''

print("out: \n", out)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var  = out_norm.var (dim=-1, keepdim=True)

print("out_norm: \n", out_norm)
print("Mean (normalized):\n", mean)
print("Variance (normalized):\n", var)

'''
out_norm: 
 tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
Mean (normalized):
 tensor([[9.9341e-09],
        [0.0000e+00]], grad_fn=<MeanBackward1>)
Variance (normalized):
 tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)

and these are the closest Mean = 0 and Variance = 1 values ! 

Notes on keepdim=True and dim param
-----------------------------------
Using keepdim=True in operations like mean or variance calculation ensures that the output tensor 
retains the same number of dimensions as the input tensor, even though the operation reduces the 
tensor along the dimension specified via dim. For instance, without keepdim=True, the returned mean tensor 
would be a two-dimensional vector [0.1324, 0.2170] instead of a 2 × 1–dimensional matrix [[0.1324], [0.2170]].

The dim parameter specifies the dimension along which the calculation of the statistic (here, mean or variance) 
should be performed in a tensor.For a two-dimensional tensor (like a matrix), using dim=-1 for operations such as 
mean or variance calculation is the same as using dim=1. This is because -1 refers to the tensor’s last dimension, 
which corresponds to the columns i a two-dimensional tensor. Later, when adding layer normalization to the GPT model, 
which produces three-dimensional tensors with the shape [batch_size, num_tokens, embedding_size], we can still use dim=-1 
for normalization across the last dimension, avoiding a change from dim=1 to dim=2

To clean up the output of the scientific precisions: 
'''

torch.set_printoptions(sci_mode=False)
print("Mean (normalized):\n", mean)
print("Variance (normalized):\n", var)
