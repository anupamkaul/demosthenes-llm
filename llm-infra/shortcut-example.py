'''

Gradient optimizations using shortcut connections in layers:

Concept behind shortcut connections, also known as skip or residual connections: 
Originally, shortcut connections were proposed for deep networks in computer vision 
(specifically, in residual networks) to mitigate the challenge of vanishing gradients. 

The vanishing gradient problem refers to the issue where gradients (which guide weight updates 
during training) become progressively smaller as they propagate backward through the layers, 
making it difficult to effectively train earlier layers.

A shortcut connection creates an alternative, shorter path for the gradient to flow through the network 
by skipping one or more layers, which is achieved by adding the output of one layer to the output of a 
later layer. This is why these connections are also known as skip connections. They play a crucial role 
in preserving the flow of gradients during the backward pass in training.

Like Layer Normalization, Shortcut Connections are another method to optimize for vanishing gradients

'''

# Let's implement a neural network and then apply short cut connections to it, checking activation results:

import torch
import torch.nn as nn
import GELU as activation

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        self.layers = nn.ModuleList(
 
            # List of 5 sequences, each sequence has 2 linear networks of predefined sizes, 
            # with a non-linear GELU activation    
 
            [
                nn.Sequential( nn.Linear(layer_sizes[0], layer_sizes[1]), activation.GELU()),
                nn.Sequential( nn.Linear(layer_sizes[1], layer_sizes[2]), activation.GELU()),
                nn.Sequential( nn.Linear(layer_sizes[2], layer_sizes[3]), activation.GELU()),
                nn.Sequential( nn.Linear(layer_sizes[3], layer_sizes[4]), activation.GELU()),
                nn.Sequential( nn.Linear(layer_sizes[4], layer_sizes[5]), activation.GELU())
            ]
  
        )
        
    def forward(self, x):
  
        # data flow and processing
        # take every layer out and apply (or not) shortcut within output

        for layer in self.layers: 
            layer_output = layer(x)

            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output # add the input of the block (x) to its output (layer_output)
            else:
                x = layer_output # if no shortcut then simply grab the output
      
        return x


# test the above DNN class with shortcuts

layer_sizes = [3, 3, 3, 3, 3, 1]

sample_input = torch.tensor( [[1., 0., -1.]] )

# models without and with shortcut
torch.manual_seed(123)
model_1 = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

torch.manual_seed(123)
model_2 = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

def print_gradients(model, x):

    output = model(x)
    target = torch.tensor( [[ 0., ]]  )

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print_gradients(model_1, sample_input)
print("\n\n")
print_gradients(model_2, sample_input)


