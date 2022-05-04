import torch
import numpy as np
print(torch.version)

data = [[1,2],[3,4]]
print(type(data))

x_data = torch.tensor(data)
print(type(x_data))
print(x_data.device)

tensor = torch.rand((3,3))
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")
tensor[:,1] = 0

print(tensor)