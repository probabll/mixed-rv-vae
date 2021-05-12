import torch
import numpy as np
import torch
import torch.distributions as td
import probabll.distributions as pd
import torch.nn as nn
from collections import OrderedDict, deque


def assert_shape(t, shape, message):
    assert t.shape == shape, f"{message} has the wrong shape: got {t.shape}, expected {shape}"        


class GenerativeModel(nn.Module):
    """
    A joint distribution over         
        
        Y \in \Delta_{K-1}
            a sparse probability vector
        Z \in R^H         
            a latent embedding
        X in {0,1}^D
            an MNIST digit

    The prior over Y is prescribed hierarchically (as a fine mixture):
        * Let F take on one of the faces of the simplex. An outcome f in {0,1}^K
          is a bit vector where f_k indicates whether vertex e_k is in the face.
        * Y|F=f is distributed over the dim(f)-dimensional simplex.
        * The probability of y is given by
            p(Y=y) = \sum_f p(F=f)p(Y=y|F=f)
    """

    def __init__(self, y_dim, z_dim, data_dim, hidden_dec_size, 
                 p_drop=0.0, prior_scores=0.0, prior_location=0.0, prior_scale=1.0,
                 z_dist='gaussian'
            ):
        """        
        :param y_dim: dimensionality (K) of the mixed rv
        :param z_dim: dimensionality (H) of the Gaussian rv (use 0 to disable it)
        :param data_dim: dimensionality (D) of the observation
        :oaram hidden_dec_size: hidden size of the decoder that parameterises X|Z=z, Y=y
        :param p_drop: dropout probability
        :param prior_scores: \omega in F|\omega 
            (float or K-dimensional tensor)
        :param prior_location: location of the Gaussian prior
            (float or H-dimensional tensor)
        :param prior_scale: scale of the Gaussian prior
            (float or H-dimensional tensor)
        :param z_dist: 'gaussian' or 'dirichlet'
            if 'dirichlet' we use prior_scale as the Dirichlet concentration 
            and ignore prior_location
        """
        assert z_dim + y_dim > 0
        assert z_dist in ['gaussian', 'dirichlet'], "Unknown choice of distribution for Z"
        super().__init__()
        self._z_dim = z_dim
        self._y_dim = y_dim
        self._data_dim = data_dim   
        self._z_dist = z_dist
        # TODO: support TransposedCNN?
        self._decoder = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(z_dim + y_dim, hidden_dec_size),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dec_size, hidden_dec_size),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dec_size, data_dim),
        )        
        self.register_buffer("_prior_scores", (torch.zeros(y_dim, requires_grad=False) + prior_scores).detach())
        self.register_buffer("_prior_location", (torch.zeros(z_dim, requires_grad=False) + prior_location).detach())
        self.register_buffer("_prior_scale", (torch.zeros(z_dim, requires_grad=False) + prior_scale).detach())

    @property
    def data_dim(self):
        return self._data_dim
    
    @property
    def latent_dim(self):
        return self._z_dim + self._y_dim
    
    @property
    def z_dim(self):
        return self._z_dim
    
    @property
    def y_dim(self):
        return self._y_dim
    
    def Z(self, predictors=None):
        """Return a Normal distribution over latent space"""
        if self._z_dim:            
            if self._z_dist == 'gaussian':
                Z = td.Independent(
                        td.Normal(loc=self._prior_location, scale=self._prior_scale),
                        1
                )
            else:
                Z = td.Dirichlet(self._prior_scale)
        else:
            Z = td.Independent(pd.Delta(self._prior_location), 1)
        return Z

    def F(self, predictors=None):
        """
        Return a distribution over the non-empty faces of the simplex
        :param predictors: input predictors, this is reserved for future use
        """
        if self._y_dim:
            return pd.NonEmptyBitVector(scores=self._prior_scores)
        else:
            return td.Independent(pd.Delta(self._prior_scores), 1)

    def Y(self, f, predictors=None):
        """
        Return a batch of masked Dirichlet distributions Y|F=f
        :param f: face-encoding [batch_size, K]
        :param predictors: input predictors, this is reserved for future use
        """
        if self._y_dim:
            return pd.MaskedDirichlet(f.bool(), torch.ones_like(f))
        else:
            return td.Independent(pd.Delta(torch.zeros_like(f)), 1)

    def X(self, y, z, predictors=None):
        """Return a product of D Bernoulli distributions"""
        if z.shape[:-1] != y.shape[:-1]:
            raise ValueError("z and y must have the same batch_shape")        
        inputs = torch.cat([y, z], -1)
        logits = self._decoder(inputs)
        return td.Independent(td.Bernoulli(logits=logits), 1)
    
    def sample(self, sample_shape=torch.Size([])):
        """Return (f, y, z, x)"""        
        # [sample_shape, K]
        f = self.F().sample(sample_shape)
        # [sample_shape, K]
        y = self.Y(f=f).sample()
        # [sample_shape, H]
        z = self.Z().sample(sample_shape)
        # [sample_shape, D]
        x = self.X(z=z, y=y).sample()
        return f, y, z, x
    
    def log_prob(self, f, y, z, x, per_bit=False, reduce=True):
        """
        Return the log probability of each one of the variables in order
        or the sum of their log probabilities.
        """
        if not reduce:
            if per_bit:
                return self.F().log_prob(f), self.Y(f).log_prob(y), self.Z().log_prob(z), self.X(z=z, y=y).base_dist.log_prob(x)
            else:
                return self.F().log_prob(f), self.Y(f).log_prob(y), self.Z().log_prob(z), self.X(z=z, y=y).log_prob(x)
        if reduce:
            if per_bit:
                return self.F().log_prob(f).unsqueeze(-1) + self.Y(f).log_prob(y).unsqueeze(-1) + self.Z().log_prob(z).unsqueeze(-1) + self.X(z=z, y=y).base_dist.log_prob(x)
            else:
                return self.F().log_prob(f) + self.Y(f).log_prob(y) + self.Z().log_prob(z) + self.X(z=z, y=y).log_prob(x) 
        
        
class InferenceModel(nn.Module):
    """
    A joint distribution over         
        
        Y \in \Delta_{K-1}
            a sparse probability vector
        Z \in R^H         
            a latent embedding
            
            
    q(Y=y,Z=z|X=x) = q(Y=y|X=x)q(Z=z|Y=y, X=x)
    and 
    q(Y=y|X=x) = \sum_f q(F=f|X=x)q(Y=y|F=f,X=x)
    
    Optionally we make a mean field assumption, then q(Z=z|Y=y, X=x)=q(Z=z|X=x),
    and optionally we predict a shared set of concentrations for all faces given x.

    """

    def __init__(self, y_dim, z_dim, data_dim, hidden_enc_size, 
                 p_drop=0.0, z_dist='gaussian', mean_field=True, shared_concentrations=True):
        assert z_dim + y_dim > 0
        assert z_dist in ['gaussian', 'dirichlet'], "Unknown choice of distribution for Z|x"
        super().__init__()
                
        self._y_dim = y_dim
        self._z_dim = z_dim
        self._z_dist = z_dist
        self._mean_field = mean_field
        self._shared_concentrations = shared_concentrations
        
        if z_dim: 
            z_num_params = 2 * z_dim if z_dist == 'gaussian' else z_dim
            self._zparams_net = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(data_dim if mean_field else y_dim + data_dim, hidden_enc_size),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, hidden_enc_size),
                nn.ReLU(),
                nn.Linear(hidden_enc_size, z_num_params)
            )
        
        if y_dim:
            self._scores_net = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(data_dim, hidden_enc_size),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, hidden_enc_size),
                nn.ReLU(),
                nn.Linear(hidden_enc_size, y_dim)
            )

            self._concentrations_net = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(data_dim if shared_concentrations else y_dim + data_dim, hidden_enc_size),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, hidden_enc_size),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, y_dim),
                nn.Softplus()
            )
                
    @property
    def mean_field(self):
        return self._mean_field
    
    def _match_sample_shape(self, x, y):
        if len(x.shape) == len(y.shape):
            return x, y        
        if len(y.shape) > len(x.shape):
            sample_dims = len(y.shape) - len(x.shape)
            sample_shape = y.shape[:sample_dims] 
            x = x.view((1,) * sample_dims + x.shape).expand(sample_shape + (-1,) * len(x.shape))
        else:
            y, x  = self._match_sample_shape(y, x)
        return x, y
            
    def Z(self, x, y, predictors=None):  
        x, y = self._match_sample_shape(x, y)
        if self._z_dim:
            inputs = x if self._mean_field else torch.cat([y, x], -1)
            params = self._zparams_net(inputs)
            if self._z_dist == 'gaussian':                
                Z = td.Normal(
                    loc=params[...,:self._z_dim], 
                    scale=nn.functional.softplus(params[...,self._z_dim:]))
                Z = td.Independent(Z, 1)
            else:
                Z = td.Dirichlet(nn.functional.softplus(params))
        else:
            Z = pd.Delta(torch.zeros(x.shape[:-1] + (0,), device=x.device))
            Z = td.Independent(Z, 1)
        return Z
            
    def F(self, x, predictors=None):
        if not self._y_dim:
            return td.Independent(pd.Delta(torch.zeros(x.shape[:-1] + (0,), device=x.device)), 1)
        # [B, K]
        scores = self._scores_net(x) 
        # constrain scores?
        # e.g., by clipping?
        # 2.5 + tanh(NN(f,x)) * 2.5 + eps
        return pd.NonEmptyBitVector(scores)


    def Y(self, x, f, predictors=None):
        x, f = self._match_sample_shape(x, f)
        if not self._y_dim:
            return td.Independent(pd.Delta(torch.zeros_like(f)), 1)
        if self._shared_concentrations:
            inputs = x  # [...,D]
        else:
            inputs = torch.cat([f, x], -1)  # [...,K+D]
        # [...,K]
        concentration = self._concentrations_net(inputs) 
        # constrain concentration?
        # e.g., by clipping?
        # 2.5 + tanh(NN(f,x)) * 2.5 + eps
        return pd.MaskedDirichlet(f.bool(), concentration)
    
    def sample(self, x, sample_shape=torch.Size([])):
        """Return (f, y, z), No gradients through this."""
        with torch.no_grad():            
            # [sample_shape, B, K]
            f = self.F(x).sample(sample_shape)
            # [sample_shape, B, K]
            y = self.Y(f=f, x=x).sample()
            # [sample_shape, B, H]
            z = self.Z(x=x, y=y).sample()
            return f, y, z
    
    def log_prob(self, x, f, y, z, reduce=True):
        """log q(f|x), log q(y|f, x), log q(z|y, x)"""
        if reduce:
            return self.F(x).log_prob(f) + self.Y(x=x, f=f).log_prob(y) + self.Z(x=x, y=y).log_prob(z)
        else:
            return self.F(x).log_prob(f), self.Y(x=x, f=f).log_prob(y), self.Z(x=x, y=y).log_prob(z)
    

class VAE:
    """
    Helper class to compute quantities related to VI.
    """

    def __init__(self, p: GenerativeModel, q: InferenceModel, 
                 use_self_critic=False, use_reward_standardisation=True):        
        self.p = p
        self.q = q
        self.use_self_critic = use_self_critic
        self.use_reward_standardisation = use_reward_standardisation
        self._rewards = deque([])

    def train(self):
        self.p.train()
        self.q.train()

    def eval(self):
        self.p.eval()
        self.q.eval()

    def gen_parameters(self):
        return self.p.parameters()

    def inf_parameters(self):
        return self.q.parameters()   
    
    def critic(self, x_obs, z, q_F):
        """This estimates reward (w.r.t sampling of F) on a single sample for variance reduction"""
        B, H, K, D = x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim
        with torch.no_grad():            
            # Approximate posteriors and samples
            # [B, K]
            f = q_F.sample()  # we resample f
            assert_shape(f, (B, K), "f ~ F|X=x, \lambda")
            q_Y = self.q.Y(x=x_obs, f=f)  # and thus also resample y
            # [B, K]
            y = q_Y.sample() 
            assert_shape(y, (B, K), "y ~ Y|X=x, F=f, \lambda")
            if not self.q.mean_field:
                q_Z = self.q.Z(x=x_obs, y=y)
                # [B, H]
                z = q_Z.sample()  # and thus also resample z
                assert_shape(z, (B, H), "z ~ Z|X=x, Y=y, \lambda")
            else:
                q_Z = None
            
            # Priors
            p_F = self.p.F()
            if p_F.batch_shape != x_obs.shape[:1]:
                p_F = p_F.expand(x_obs.shape[:1] + p_F.batch_shape)

            p_Y = self.p.Y(f)  # we condition on f ~ q_F                     

            # Sampling distribution
            p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y and z ~ q_Z

            # [B]
            ll = p_X.log_prob(x_obs)
            # [B]
            kl_Y = td.kl_divergence(q_Y, p_Y)
            # [B]
            critic = ll - kl_Y
            
            if not self.q.mean_field:
                p_Z = self.p.Z()
                if p_Z.batch_shape != x_obs.shape[:1]:
                    p_Z = p_Z.expand(x_obs.shape[:1] + p_Z.batch_shape)
                # [B]
                kl_Z = td.kl_divergence(q_Z, p_Z)
                # [B]
                critic -= kl_Z
            
            return critic
        
    def update_reward_stats(self, reward):
        """Return the current statistics and update the vector"""
        if len(self._rewards) > 1:
            avg = np.mean(self._rewards)
            std = np.std(self._rewards)
        else:
            avg = 0.0
            std = 1.0
        if len(self._rewards) == 100:
            self._rewards.popleft()
        self._rewards.append(reward.mean(0).item())
        return avg, std
    
    def DR(self, x_obs):
        with torch.no_grad():
            B, H, K, D = x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim            

            # Posterior approximations and samples
            q_F = self.q.F(x_obs)
            # [B, K]
            f = q_F.sample() # not rsample
            assert_shape(f, (B, K), "f ~ F|X=x, \lambda")

            q_Y = self.q.Y(x=x_obs, f=f)
            y = q_Y.rsample()
            assert_shape(y, (B, K), "y ~ Y|X=x, F=f, \lambda")
            
            q_Z = self.q.Z(x=x_obs, y=y)
            # [B, H]
            z = q_Z.rsample()
            assert_shape(z, (B, H), "z ~ Z|X=x, Y=y, \lambda")
            
            # Priors            
            p_F = self.p.F()
            if p_F.batch_shape != x_obs.shape[:1]:
                p_F = p_F.expand(x_obs.shape[:1] + p_F.batch_shape)

            p_Y = self.p.Y(f)  # we condition on f ~ q_F thus batch_shape is already correct
            
            p_Z = self.p.Z()
            if p_Z.batch_shape != x_obs.shape[:1]:
                p_Z = p_Z.expand(x_obs.shape[:1] + p_Z.batch_shape)
                
            # Sampling distribution
            p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y

            # Return type
            ret = OrderedDict(
                D=0.,
                R=0.,
            )

            # ELBO: the first term is an MC estimate (we sampled (f,y))
            # the second term is exact 
            # the third tuse_self_criticis an MC estimate (we sampled f)
            D = -p_X.log_prob(x_obs)            
            kl_Y = td.kl_divergence(q_Y, p_Y)
            kl_F = td.kl_divergence(q_F, p_F)
            kl_Z = td.kl_divergence(q_Z, p_Z)

            ret['D'] = D
            ret['R'] = kl_F + kl_Y + kl_Z
            ret['R_F'] = kl_F
            ret['R_Y'] = kl_Y
            ret['R_Z'] = kl_Z
        return ret

    def loss(self, x_obs):
        """
        :param x_obs: [B, D]
        """
        B, H, K, D = x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim        
                
        # Approximate posteriors and samples
        q_F = self.q.F(x_obs)
        # [B, K]
        f = q_F.sample() # not rsample
        assert_shape(f, (B, K), "f ~ F|X=x, \lambda")
        
        q_Y = self.q.Y(x=x_obs, f=f)
        y = q_Y.rsample()  # with reparameterisation! (important)
        assert_shape(y, (B, K), "y ~ Y|F=f, \lambda")
        
        q_Z = self.q.Z(x=x_obs, y=y)
        # [B, H]
        z = q_Z.rsample()  # with reparameterisation
        assert_shape(z, (B, H), "z ~ Z|X=x, \lambda")
        
        # Priors
        p_F = self.p.F()
        if p_F.batch_shape != x_obs.shape[:1]:
            p_F = p_F.expand(x_obs.shape[:1] + p_F.batch_shape)
        
        p_Y = self.p.Y(f)  # we condition on f ~ q_F  thus batch_shape is already correct
        
        p_Z = self.p.Z()
        if p_Z.batch_shape != x_obs.shape[:1]:
            p_Z = p_Z.expand(x_obs.shape[:1] + p_Z.batch_shape)
            
        # Sampling distribution
        p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y
        
        # Return type
        ret = OrderedDict(
            loss=0.,
        )
        
        # ELBO: the first term is an MC estimate (we sampled (f,y))
        # the second term is exact 
        # the third tuse_self_criticis an MC estimate (we sampled f)
        ll = p_X.log_prob(x_obs)        
        kl_Z = td.kl_divergence(q_Z, p_Z)
        kl_Y = td.kl_divergence(q_Y, p_Y)
        kl_F = td.kl_divergence(q_F, p_F)
        
        # Logging ELBO terms
        ret['D'] = -ll.mean(0).item()        
        if self.p.y_dim:
            ret['R_F'] = kl_F.mean(0).item()
            ret['R_Y'] = kl_Y.mean(0).item()
        if self.p.z_dim:
            ret['R_Z'] = kl_Z.mean(0).item()        
            
        # Gradient surrogates and loss
        
        # i) reparameterised gradient (g_rep)
        grep_surrogate = ll - kl_Z - kl_F - kl_Y

        # ii) score function estimator (g_SFE)
        if self.p.y_dim:            
            # E_ZFY[ log p(x|z,f,y)] - -KL(Z) - KL(F) - E_F[ KL(Y) ]
            # E_F[ E_Y[ E_Z[ log p(x|z,f,y) ] - KL(Y) ] ] -KL(Z) - KL(F)
            # E_F[ r(F) ] for r(f) = log p(x|z,f,y)
            # r(f).detach() * log q(f)            
            reward = (ll - kl_Y).detach() if self.q.mean_field else (ll - kl_Y - kl_Z).detach()
            # Variance reduction tricks
            if self.use_self_critic:
                criticised_reward = reward - self.critic(x_obs, z=z, q_F=q_F).detach()
            else:
                criticised_reward = reward        
            if self.use_reward_standardisation:
                reward_avg, reward_std = self.update_reward_stats(criticised_reward)
                standardised_reward = (criticised_reward - reward_avg) / np.minimum(reward_std, 1.0)
            else:
                standardised_reward = criticised_reward

            sfe_surrogate = standardised_reward * q_F.log_prob(f)
            
            # Loggin SFE variants
            ret['SFE_reward'] = reward.mean(0).item()
            if self.use_self_critic:
                ret['SFE_criticised_reward'] = criticised_reward.mean(0).item()
            if self.use_reward_standardisation:
                ret['SFE_standardised_reward'] = standardised_reward.mean(0).item()
        else:
            sfe_surrogate = torch.zeros_like(grep_surrogate)
        
        # []
        loss = -(grep_surrogate + sfe_surrogate).mean(0)
        ret['loss'] = loss.item()

        return loss, ret

    def estimate_ll(self, x_obs, num_samples):     
        with torch.no_grad():
            self.eval()
            # log 1/N \sum_{n} p(x, z_n)/q(z_n|x)
            # [N, B, K], [N, B, K], [N, B, H]
            f, y, z = self.q.sample(x_obs, (num_samples,))
            # Here I compute: log p(f) + log p(y|f) + log p(z) + log p(x|y,z)
            # [N, B]
            log_p = self.p.log_prob(f=f, y=y, z=z, x=x_obs)
            # Here I compute: log q(f|x) + log q(y|x,f) + log q(z|x,y) 
            # [N, B]
            log_q = self.q.log_prob(x=x_obs, f=f, y=y, z=z)
            # [B]
            ll = torch.logsumexp(log_p - log_q, 0) - np.log(num_samples)                    
        return ll

    def estimate_ll_per_bit(self, x_obs, num_samples):             
        with torch.no_grad():
            # log 1/N \sum_{n} p(x, z_n)/q(z_n|x)
            # [N, B, K], [N, B, K], [N, B, H]
            f, y, z = self.q.sample(x_obs, (num_samples,))        
            # [N, B, D]
            log_p = self.p.log_prob(f=f, y=y, z=z, x=x_obs, per_bit=True)
            # [N, B]
            log_q = self.q.log_prob(z=z, f=f, y=y, x=x_obs)
            # [B, D]
            ll = torch.logsumexp(log_p - log_q.unsqueeze(-1), 0) - np.log(num_samples)                   
        return ll    
    
