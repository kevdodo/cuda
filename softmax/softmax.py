import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else cpu)
input = torch.randn(10_000_000, device=device)
m = nn.Softmax(dim=-1)
output = m(input)
print(torch.sum(output))
