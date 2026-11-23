import torch

assert torch.backends.mps.is_available(), "MPS unavailable"

device = torch.device("mps")

a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)

c = a @ b
torch.mps.synchronize()

print("Device:", c.device)
print("Shape:", tuple(c.shape))
print("Result finite:", torch.isfinite(c).all().item())
