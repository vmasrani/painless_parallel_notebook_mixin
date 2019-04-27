import numpy.random as npr
import torch
from torch.distributions.normal import Normal


def seed_all(seed):
    npr.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# linear regression data
def linear_regression_data(n, d, offset=0):
    X = torch.randn(n, d)
    X[:, 0] = 1.
    w_true = offset + torch.randn(d, 1)
    y = Normal(loc=X.mm(w_true), scale=1).sample()
    return X, y, w_true.squeeze()
