import sys
import torch
import numpy
import pandas
import matplotlib
import tiktoken
import tensorflow as tf

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("TensorFlow:", tf.__version__)
print("MPS available:", torch.backends.mps.is_available())
