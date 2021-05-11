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
# metric        mean       min
# --------  --------  --------
# val_nll   97.8193   96.7285
# val_bpd    2.20505   2.18046
```
