import torch

# Check if Metal Performance Shaders (MPS) is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("Success! Sub-tensor initialized on GPU:", x)
else:
    print("MPS device not found. Check your environment.")
