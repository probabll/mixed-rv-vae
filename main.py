import sys
import json
import pathlib
import random
from shutil import copyfile
import torch
import numpy as np
import torch
import torch.distributions as td
import probabll.distributions as pd
import torch.nn as nn
from collections import namedtuple, defaultdict, OrderedDict, deque
from tabulate import tabulate
from tqdm.auto import tqdm
from itertools import chain
from components import GenerativeModel, InferenceModel, VAE
from data import load_mnist
from data import Batcher


def default_cfg():
    cfg = OrderedDict(        
        # Data        
        batch_size=200,
        data_dir='tmp',
        height=28,
        width=28, 
        output_dir='.',
        # CUDA
        seed=42,
        device='cuda:0',
        # Joint distribution    
        y_dim=10,    
        prior_scores=0.0,
        z_dim=32,     
        z_dist='gaussian',    
        prior_location=0.0, 
        prior_scale=1.0,
        hidden_dec_size=500,
        # Approximate posteriors
        shared_concentrations=True,
        mean_field=False,
        hidden_enc_size=500,    
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
    return cfg


def parse_args(path, break_if_missing=True, break_if_unknown=True):
    with open(path) as f:
        d = json.load(open(path), object_hook=OrderedDict)
    known = default_cfg()
    for k in known.keys():
        if k not in d:
            print(f"Are you sure you do not want to specify {k}?")
            if break_if_missing:
                raise ValueError(f'Missing cfg for {k}')
    for k, v in d.items():
        if k not in known:
            print(f"Key {k} with value {v} is unknown")
            if break_if_unknown:
                raise ValueError(f'Unknown cfg: {k}')
    args = namedtuple('Config', d.keys())(*d.values())
    return args 


def get_batcher(data_loader, args, binarize=True, num_classes=10, onehot=True):
    batcher = Batcher(
        data_loader, 
        height=args.height, 
        width=args.width, 
        device=torch.device(args.device), 
        binarize=binarize, 
        num_classes=num_classes,
        onehot=onehot
    )
    return batcher


def validate(vae: VAE, batcher: Batcher, num_samples: int, compute_DR=False):
    """
    Return average NLL
        average number of bits per dimension
        and a dictionary with distortion and rate estimates
    """
    with torch.no_grad():
        vae.eval()
        
        nb_obs = 0
        nb_bits = 0.
        ll = 0.
        DR = OrderedDict()
        for x_obs, y_obs in batcher:
            # [B, H*W]
            x_obs = x_obs.reshape(-1, vae.p.data_dim)     
            # [B]
            ll = ll + vae.estimate_ll(x_obs, num_samples).sum(0)
            nb_bits += np.prod(x_obs.shape)
            nb_obs += x_obs.shape[0]
            # []
            if compute_DR:
                ret = vae.DR(x_obs)
                for k, v in ret.items():
                    if k not in DR:
                        DR[k] = []
                    DR[k].append(v.cpu().numpy())

    nll = - (ll / nb_obs).cpu()
    if compute_DR:
        DR = OrderedDict((k, np.concatenate(v, 0)) for k, v in DR.items())    
        return nll, nll / np.log(2) / vae.p.latent_dim, DR
    else:
        return nll, nll / np.log(2) / vae.p.latent_dim


def main(cfg_file, load_ckpt=False, reset_opt=False):
    
    args = parse_args(cfg_file)
    
    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)    
    
    # Make dirs
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    copyfile(cfg_file, f"{args.output_dir}/cfg.json")
    
    # Preparing data
    print("# Preparing MNIST (may take some time the first time)", file=sys.stderr)
    train_loader, valid_loader, test_loader = load_mnist(
        args.batch_size, 
        save_to=args.data_dir, 
        height=args.height, 
        width=args.width
    )
    
    # Make model components
    print("# Building model components", file=sys.stderr)
    p = GenerativeModel(    
        y_dim=args.y_dim,
        z_dim=args.z_dim, 
        data_dim=args.height * args.width, 
        hidden_dec_size=args.hidden_dec_size,
        p_drop=args.gen_p_drop,
        prior_scores=args.prior_scores,
        prior_location=args.prior_location,
        prior_scale=args.prior_scale,
        z_dist=args.z_dist,
    )
    print(f"Generative model\n{p}", file=sys.stderr)
    
    q = InferenceModel(    
        y_dim=args.y_dim,
        z_dim=args.z_dim, 
        data_dim=args.height * args.width, 
        hidden_enc_size=args.hidden_enc_size,
        shared_concentrations=args.shared_concentrations,
        p_drop=args.inf_p_drop,
        z_dist=args.z_dist,
        mean_field=args.mean_field,
    )
    print(f"Inference model\n{q}", file=sys.stderr)    
    
    # Checkpoints
    
    if load_ckpt and pathlib.Path(f"{args.output_dir}/training.ckpt").exists():
        print("Loading model parameters")
        ckpt = torch.load(f"{args.output_dir}/training.ckpt")
        p.load_state_dict(ckpt['p_state_dict'])            
        p = p.to(torch.device(args.device))
        q.load_state_dict(ckpt['q_state_dict'])            
        q = q.to(torch.device(args.device))
        # Get optimisers
        p_opt = torch.optim.Adam(p.parameters(), lr=args.gen_lr, weight_decay=args.gen_l2)
        q_opt = torch.optim.Adam(q.parameters(), lr=args.inf_lr, weight_decay=args.inf_l2)
        if not reset_opt:
            print("Loading optimiser state")
            p_opt.load_state_dict(ckpt['p_opt_state_dict'])
            q_opt.load_state_dict(ckpt['q_opt_state_dict'])
        stats_tr = ckpt['stats_tr']
        stats_val = ckpt['stats_val']
    else:
        p = p.to(torch.device(args.device))
        q = q.to(torch.device(args.device))
        # Get optimisers
        p_opt = torch.optim.Adam(p.parameters(), lr=args.gen_lr, weight_decay=args.gen_l2)
        q_opt = torch.optim.Adam(q.parameters(), lr=args.inf_lr, weight_decay=args.inf_l2)
        stats_tr = defaultdict(list)
        stats_val = defaultdict(list)

    # Training 
    
    vae = VAE(p, q,
        use_self_critic=args.use_self_critic, 
        use_reward_standardisation=args.use_reward_standardisation
    )
    
    # Demo 
    print("Example:\n 2 samples from inference model", file=sys.stderr)
    print(q.sample(p.sample((2,))[-1]), file=sys.stderr)
    print(" their log probability density under q", file=sys.stderr)
    print(q.log_prob(
        torch.zeros(args.height * args.width, device=torch.device(args.device)),
        *q.sample(p.sample((2,))[-1]), 
    ), file=sys.stderr)    
    
    
    print("# Training", file=sys.stderr)
    val_metrics = validate(vae, get_batcher(valid_loader, args), args.num_samples)
    print(f'Validation {0:3d}: nll={val_metrics[0]:.2f} bpd={val_metrics[1]:.2f}', file=sys.stderr)
    for epoch in range(args.epochs):

        iterator = tqdm(get_batcher(train_loader, args))

        for x_obs, y_obs in iterator:        
            # [B, H*W]
            x_obs = x_obs.reshape(-1, args.height * args.width)
            
            vae.train()      
            loss, ret = vae.loss(x_obs)

            for k, v in ret.items():
                stats_tr[k].append(v)

            p_opt.zero_grad()
            q_opt.zero_grad()        
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(vae.gen_parameters(), vae.inf_parameters()), 
                args.grad_clip
            )        
            p_opt.step()
            q_opt.step()

            iterator.set_description(f'Epoch {epoch+1:3d}/{args.epochs}')
            iterator.set_postfix(ret)

        val_metrics = validate(vae, get_batcher(valid_loader, args), args.num_samples)
        stats_val['val_nll'].append(val_metrics[0])
        stats_val['val_bpd'].append(val_metrics[1])
        print(f'Validation {epoch+1:3d}: nll={val_metrics[0]:.2f} bpd={val_metrics[1]:.2f}', file=sys.stderr)
        
        torch.save({
            'p_state_dict': p.state_dict(),
            'q_state_dict': q.state_dict(),
            'p_opt_state_dict': p_opt.state_dict(),
            'q_opt_state_dict': q_opt.state_dict(),
            'stats_tr': stats_tr,
            'stats_val': stats_val,
        }, f"{args.output_dir}/training.ckpt")
    
    
    np_stats_tr = {k: np.array(v) for k, v in stats_tr.items()}
    np_stats_val = {k: np.array(v) for k, v in stats_val.items()}
    
#     torch.save({
#         'epoch': epoch + 1,
#         'p_state_dict': p.state_dict(),
#         'q_state_dict': p.state_dict(),
#         'p_opt_state_dict': p_opt.state_dict(),
#         'q_opt_state_dict': p_opt.state_dict(),
#         'stats_tr': stats_tr,
#         'stats_val': stats_val,
#     }, f"{args.output_dir}/training.ckpt")
    

    #print(tabulate(
    #    [(k, np.mean(v[-100:]), np.min(v[-100:])) for k, v in np_stats_val.items()],
    #    headers=['metric', 'mean', 'min']
    #))
    
    print("# Final validation run")
    val_nll, val_bpd, val_DR = validate(
        vae, get_batcher(valid_loader, args), args.num_samples, compute_DR=True)
    rows = [('NLL', val_nll, None), ('BPD', val_bpd, None)]
    for k, v in val_DR.items():
        rows.append((k, v.mean(), v.std()))
    print(tabulate(rows, headers=['metric', 'mean', 'std']))    
    
#     print("Final test run")
#     test_nll, test_bpd, test_DR = validate(
#         vae, get_batcher(test_loader, args), args.num_samples, compute_DR=True)
#     rows = [('IS-NLL', test_nll, None), ('IS-BPD', test_bpd, None)]
#     for k, v in test_DR.items():
#         rows.append((k, v.mean(), v.std()))
#     print(tabulate(rows, headers=['metric', 'mean', 'std']))    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} cfg (loadckpt) (resetopt)")
        print(" after the cfg file, you can use some optional commands:")
        print(" * loadckpt")
        print(" * resetopt")
        
        sys.exit()
    main(sys.argv[1], 'loadckpt' in sys.argv[2:], 'resetopt' in sys.argv[2:])