# Gaussian VAE


```python
# Basic FF encoder and decoder
cfg = dict(
    # Data
    batch_size=200,
    data_dir='tmp',
    height=28,
    width=28, 
    # CUDA
    device='cuda:0',
    # Joint distribution
    z_dim=64, 
    y_dim=0,
    cond='fx',
    prior_scores=0.0,
    prior_location=0.0, 
    prior_scale=1.0,
    # Architecture
    hidden_enc_size=500,
    hidden_dec_size=500,
    # Training
    epochs=200,    
    # Evaluation
    num_samples=100,    
    # Optimisation & regularisation
    gen_lr=1e-4,
    inf_lr=1e-4,  
    gen_l2=1e-4,
    inf_l2=1e-4,  
    gen_p_drop=0.1,  
    inf_p_drop=0.0,  # dropout for inference model is not well understood    
    grad_clip=5.0,
    # Variance reduction
    use_self_critic=False,
    use_reward_standardisation=True,
)

# Training KL ~ 25 
# 
metric        mean       min
--------  --------  --------
val_nll   97.8193   96.7285
val_bpd    2.20505   2.18046

# With z_dim=10 we get
metric        mean       std
--------  --------  --------
IS-NLL    102.367
IS-BPD     14.7685
D          88.5055  29.1935
R          17.4076   1.83976
R_Z        17.4076   1.83976
R_F         0        0
R_Y         0        0

```

# Dirichlet VAE

with 10 units

```python
metric        mean       std
--------  --------  --------
IS-NLL    118.015
IS-BPD     17.0259
D         106.71    34.2814
R          14.7882   1.44848
R_Z        14.7882   1.44848
R_F         0        0
R_Y         0        0
```

# Mixed RV VAE 

This employs a mixtured of masked Dirichlet distributions as prior over a K-dimensional latent code (a sparse probability vector)

```python
cfg = dict(
    # Data
    batch_size=200,
    data_dir='tmp',
    height=28,
    width=28, 
    # CUDA
    device='cuda:0',
    # Joint distribution
    z_dim=0, 
    y_dim=10,
    cond='fx',
    prior_scores=0.0,
    prior_location=0.0, 
    prior_scale=1.0,
    # Architecture
    hidden_enc_size=500,
    hidden_dec_size=500,
    # Training
    epochs=200,    
    # Evaluation
    num_samples=100,    
    # Optimisation & regularisation
    gen_lr=1e-4,
    inf_lr=1e-4,  
    gen_l2=1e-4,
    inf_l2=1e-4,  
    gen_p_drop=0.1,  
    inf_p_drop=0.0,  # dropout for inference model is not well understood    
    grad_clip=5.0,
    # Variance reduction
    use_self_critic=False,
    use_reward_standardisation=True,
)
``` 

```python
metric         mean        std
--------  ---------  ---------
IS-NLL    120.99
IS-BPD     17.4551
D         112.068    33.8009
R          12.9761    2.52693
R_Z         0         0
R_F         5.07505   0.763042
R_Y         7.90104   2.47224
```


# Mixed (NLL=120 -> lower)
Y ~ mixed (Simplex)
   sampling/marginal: \sum_f p(f)p(y|f)
   Prior:
    F ~ Gibbs(w_1, ..., w_K) properly normalised over constrained support
    Y|F=f ~ Dir(1_{dim(f)+1})
    
   Posterior:
    F|x ~ Gibbs(NN1(x))  from image x to K scores with an NN1
        * no rparam (SFE)
    Y|x,f ~ Dir(NN2(x))  from image x to dim(f)+1 concentrations with an NN2
        * rparam (grep)
     
X|y ~ dec(y)


# VAE-Dir 
Z ~ Dirichlet (Simplex)  # latent code is a dense prob vector
X|z ~ dec(z)

Pr(full simple) = Dir(1)

# samples
# efficiency code! H N-precision bit


# VAE (NLL=90)
Z ~ Gaussian (R^D)      # latent code is an embedding 
X|z ~ dec(z)

# Mixture model with K components 

C ~ Categorical()    
X|c ~ dec(c)
