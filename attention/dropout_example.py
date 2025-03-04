import torch

'''

Masking additional attention weights with dropout: Dropout in DL is a technique where randomly selected
hidden layer units are ignored during training effectively "dropping" them out. This method helps prevent
overfitting by ensuring that a model does not become overly reliant on any specific set of hidden layer units.
Dropout is only empolyed during training and is disabled afterwards.

In transformer archs like GPT, dropout i attention mechanusm is typically applied at 2 stages: after calculating
the atention weights, or after applying the attention weights to the value vectors. 

Here we use a dropout rate of 50%, which means masking out half of the attention rates. When we train GPT model, we 
will use a much lower dropout rate, like 0.1 or 0.2. We apply PyTorch's dropout implementation first to a 6*6 tensor
consisting of 1s for simplicity

'''

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)

example = torch.ones(6, 6)
print("A 6X6 tensor before dropout:\n", example, "\n")
'''
tensor([[1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.]]) 


'''

print("Post dropout (50%): \n", dropout(example))  

'''
tensor([[2., 2., 2., 2., 2., 2.],
        [0., 2., 0., 0., 0., 0.],
        [0., 0., 2., 0., 2., 0.],
        [2., 2., 0., 0., 0., 2.],
        [2., 0., 0., 0., 0., 2.],
        [0., 2., 0., 0., 0., 0.]])
'''

'''

When applying dropout to an attention wt matrix with a rate of 50%, half of the elements in the matrix are randomly
set to zdro. To compensate for the reduction in the active elements, the values of the remaining elements are scaled 
up by a factor of 1/0.5=2. This scaling is crucial to maintain the overall balance of the attention weights, ensuring 
that the average influence of the attention mechanism remains consistent during both the training and inference phases.

'''


