import torch
import torch.nn as nn

input = torch.randn(1024)
m = nn.Softmax(dim=-1)
output = m(input)
print(output)
