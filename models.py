import numpy as np
import numpy.random as npr
import pandas as pd
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn
from torch.distributions.normal import Normal


class LinRegGuide(nn.Module):
    """Mean field variational distribution using reparam trick"""
    def __init__(self, d, mu0=0, sig0=1, sig_lik=1):
        super().__init__()
        self.d         = d

        # For MCMC
        self.mu0       = mu0
        self.sig0      = sig0
        self.sig_lik   = sig_lik

        # For Variational Inference
        self.prior     = Normal(self.mu0, self.sig0)
        self.mu        = nn.Parameter(torch.randn((d, 1)))
        self._logsigma = nn.Parameter(torch.randn((d, 1)))

        self.sigma     = self._logsigma.exp()
        self.z         = torch.zeros_like(self.mu)

    def forward(self, input):
        self._logsigma.data.clamp_min_(-10)
        epsilon = torch.randn((self.mu.size()))
        self.sigma = self._logsigma.exp()

        self.z = self.mu + epsilon * self.sigma
        output = input.mm(self.z.view(self.d, 1))
        return output

    def log_lik(self, mu, y):
        return Normal(loc=mu, scale=1).log_prob(y).sum()

    def log_prior(self):
        return self.prior.log_prob(self.z).sum()

    def entropy(self):
        return Normal(self.mu, self.sigma).entropy().sum()

    def mh_target(self, X, y, w):
        mu = X.mm(w.view(self.d, 1))
        return Normal(mu, self.sig_lik).log_prob(y).sum() + \
               self.prior.log_prob(w).sum()



class Elbo(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, yhat, y):
        return self.model.entropy() - \
               self.model.log_lik(yhat, y) - \
               self.model.log_prior()
