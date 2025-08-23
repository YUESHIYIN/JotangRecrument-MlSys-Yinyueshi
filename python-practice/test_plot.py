import torch
import numpy as np

data = [[1,2],[3,4]]

x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

tensor = torch.rand(3,4)
#print(tensor.shape,tensor.dtype,tensor.device)

#print(torch.accelerator.is_available())
#print(torch.accelerator.current_accelerator())

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))