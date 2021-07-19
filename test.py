import torch

from module import build_relative_position

a = build_relative_position(10, 10, 2, 'cpu')
print(a.type())


