import torch


random_tensor = torch.rand(3, 4)
# print(random_tensor)
# print(random_tensor.shape)
# print(random_tensor.ndim)

# random_image_size_tensor = torch.rand(size= (3, 230, 200))
# print(random_image_size_tensor)
# print(random_image_size_tensor.shape)
# print(random_image_size_tensor.ndim)

# tensor_arange = torch.arange(start=1, end=11)
# tensor_zero = torch.zeros_like(tensor_rane)
# print(tensor_rane)
# print(tensor_zero)
float_32_tensor = torch.tensor([1.0, 2.0, 15.0],
                               dtype=None,
                               device=None,
                               requires_grad=False)
print(float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor.dtype)

multiple = float_16_tensor * float_32_tensor
print(multiple)