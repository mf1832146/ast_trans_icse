
import torch

a = torch.ones((3, 2), device='cuda', dtype=torch.long)
b = torch.zeros((2,2), device='cpu', dtype=torch.float).long().to(a.device)

print(b)