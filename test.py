import collections
import random
import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


# mu = torch.tensor([[3, 3], [5, 6], [6, 7]], dtype=torch.float)
#
# mu.requires_grad = True
# print (mu.shape)
#
#
# mx = torch.sum(mu, dim=-2)
# print (mx)
# print (mx.shape)
#
# my = torch.sum(mu, dim=-2, keepdim=True)
# print (my)
# print (my.shape)

np.random.seed(1)

L1 = np.random.rand(3, 3)
L2 = np.random.rand(3, 3)
print (L1)
print (L2)