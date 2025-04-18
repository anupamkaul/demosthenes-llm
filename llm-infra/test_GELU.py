# test the GELU class
# test and plot GELU activation function vs RELU activation function

import torch
import torch.nn as nn
import GELU as gelu


import matplotlib.pyplot as plt
gelu, relu = gelu.GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))

for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
    

plt.tight_layout()
plt.show()

