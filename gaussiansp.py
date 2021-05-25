import torch
import torch.distributions as td
from mvnorm import multivariate_normal_cdf

import torch
import torch.distributions as td
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from itertools import product
import numpy as np
from torch.distributions.utils import _standard_normal, broadcast_all
from entmax import sparsemax


class GaussianSparsemax(td.Distribution):
    
    arg_constraints = {
        'loc': constraints.real, 
        'scale': constraints.positive
    }    
    support = td.constraints.simplex
    has_rsample = True
    
    @classmethod
    def all_faces(K):
        """Generate a list of 2**K - 1 bit vectors indicating all possible faces of a K-dimensional simplex."""
        return list(product([0, 1], repeat=K))[1:]

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianSparsemax, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape + self.event_shape)
        new.scale = self.scale.expand(batch_shape + self.event_shape)
        super(GaussianSparsemax, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # sample_shape + batch_shape + (K,)
        z = td.Normal(loc=self.loc, scale=self.scale).rsample(sample_shape)
        return sparsemax(z, dim=-1)
    
    def log_prob(self, y, pivot_alg='first', tiny=1e-12, huge=1e12):
        K = y.shape[-1]
        # [B, K]
        loc = self.loc
        scale = self.scale
        var = scale ** 2
        
        # The face contains the set of coordinates greater than zero
        # [B, K]
        face = y > 0 

        # Chose a pivot coordinate (a non-zero coordinate)
        # [B]
        if pivot_alg == 'first':
            ind_pivot = torch.argmax((face > 0).float(), -1)
        elif pivot_alg == 'random':
            ind_pivot = td.Categorical(
                probs=face.float()/(face.float().sum(-1, keepdims=True))
            ).sample()
        # Select a batch of pivots 
        # [B, K]
        pivot_indicator = torch.nn.functional.one_hot(ind_pivot, K).bool()
        # All non-zero coordinates but the pivot
        # [B, K]
        others = torch.logical_xor(face, pivot_indicator)
        # The value of the pivot coordinate
        # [B]
        t = (y * pivot_indicator.float()).sum(-1)
        # Pivot mean and variance
        # [B]
        t_mean = torch.where(pivot_indicator, loc, torch.zeros_like(loc)).sum(-1)
        t_var = torch.where(pivot_indicator, var, torch.zeros_like(var)).sum(-1)

        # Difference with respect to the pivot
        # [B, K]
        y_diff = torch.where(others, y - t.unsqueeze(-1), torch.zeros_like(y))
        # [B, K]
        mean_diff = torch.where(
            others, 
            loc - t_mean.unsqueeze(-1),
            torch.zeros_like(loc)
        )
        
        # Joint log pdf for the non-zeros
        # [B, K, K]    
        diag = torch.diag_embed(torch.where(others, var, torch.ones_like(var)))
        offset = t_var.unsqueeze(-1).unsqueeze(-1)
        # We need a multivariate normal for the non-zero coordinates in `other`
        # but to batch mvns we will need to use K-by-K covariances
        # we can do so by embedding the lower-dimensional mvn in a higher dimensional mvn
        # with cov=I.
        # [B, K, K]
        cov_mask = others.unsqueeze(-1) * others.unsqueeze(-2)
        cov = torch.where(cov_mask, diag + offset, diag)
        # This computes log prob of y[other] under  the lower dimensional mvn
        # times log N(0|0,1) for the other dimensions
        # [B]
        log_prob = td.MultivariateNormal(mean_diff, cov).log_prob(y_diff)
        # so we discount the contribution from the masked coordinates
        # [B, K]
        log_prob0 = td.Normal(torch.zeros_like(mean_diff), torch.ones_like(mean_diff)).log_prob(torch.zeros_like(y_diff)) 
        log_prob = log_prob - torch.where(others, torch.zeros_like(log_prob0), log_prob0).sum(-1)

        # Joint log prob for the zeros (needs the cdf)
        # [B]
        constant_term = 1. / torch.where(face, 1./var, torch.zeros_like(var)).sum(-1)
        # Again, we aim to reason with lower-dimensional mvns via 
        # the td.MultivariateNormal interface. For that, I will mask the coordinates in face.
        # The non-zeros get a tiny variance
        # [B, K, K]
        diag_corrected = torch.diag_embed(torch.where(face, torch.zeros_like(var) + tiny, var)) 
        # [B, 1, 1]
        offset_corrected = constant_term.unsqueeze(-1).unsqueeze(-1)
        # These are the zeros only.
        # [B, K, K]
        cov_corrected_mask = torch.logical_not(face).unsqueeze(-1) * torch.logical_not(face.unsqueeze(-2))    
        cov_corrected = torch.where(cov_corrected_mask, diag_corrected + offset_corrected, diag_corrected)    

        # The non-zeros get a large negative mean.
        # [B]
        mean_constant_term = constant_term * torch.where(face, (y - loc)/var, torch.zeros_like(y)).sum(-1)
        # [B, K]
        #  see that for non-zeros I move the location to something extremely negative
        #  in combination with tiny variace this makes the density of 0 evaluate to 0
        #  and the cdf of 0 evaluate to 1, for those coordinates
        mean_corrected = torch.where(face, torch.zeros_like(y) - huge, loc + mean_constant_term.unsqueeze(-1))

        # [B]
        cdf = multivariate_normal_cdf(
            torch.zeros_like(y),
            mean_corrected, cov_corrected
        )
        log_cdf = cdf.log()

        # [B]
        log_det = face.float().sum(-1).log()

        # [B]
        return log_prob + log_cdf + log_det


@td.register_kl(GaussianSparsemax, GaussianSparsemax)
def _kl_gaussiansparsemax_gaussiansparsemax(p, q):
    x = p.rsample()
    return p.log_prob(x) - q.log_prob(x)


import probabll.distributions as pd


class GaussianSparsemaxPrior(td.Distribution):
    
    def __init__(self, pF, alpha_net, validate_args=False):
        self.pF = pF
        self.alpha_net = alpha_net
        batch_shape, event_shape = pF.batch_shape, pF.event_shape        
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianSparsemaxPrior, _instance)
        new.pF = self.pF.expand(batch_shape)
        new.alpha_net = self.alpha_net
        super(GaussianSparsemaxPrior, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new       
        
    def sample(self, sample_shape=torch.Size()):
        f = self.pF.sample(sample_shape)
        Y = pd.MaskedDirichlet(f.bool(), self.alpha_net(f)) 
        return Y.sample()
        
    def log_prob(self, value):        
        f = (value > 0).float()       
        Y = pd.MaskedDirichlet(f.bool(), self.alpha_net(f)) 
        return self.pF.log_prob(f) + Y.log_prob(value)        
    
@td.register_kl(GaussianSparsemax, GaussianSparsemaxPrior)
def _kl_gaussiansparsemax_gaussiansparsemaxprior(p, q):
    x = p.rsample()
    return p.log_prob(x) - q.log_prob(x)
