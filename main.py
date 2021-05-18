import wandb
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


def get_optimiser(choice, params, lr, weight_decay, momentum=0.0):    
    if choice == 'adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif choice == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError("Unknown optimiser")
    return opt


def make_state(args, device: str, ckpt_path=None, load_opt=True):
    """
    :param ckpt_path: use None to get an untrained model
    """

    if ckpt_path and not pathlib.Path(ckpt_path).exists():
        raise ValueError(f"I cannot find {ckpt_path}. Use None to get an untrained model.")

    p = GenerativeModel(    
        y_dim=args.y_dim,
        z_dim=args.z_dim, 
        data_dim=args.height * args.width, 
        hidden_dec_size=args.hidden_dec_size,
        p_drop=args.gen_p_drop,
        prior_f=args.prior_f,
        prior_y=args.prior_y,
        prior_z=args.prior_z,
    )
    
    q = InferenceModel(    
        y_dim=args.y_dim,
        z_dim=args.z_dim, 
        data_dim=args.height * args.width, 
        hidden_enc_size=args.hidden_enc_size,
        shared_concentrations=args.shared_concentrations,
        p_drop=args.inf_p_drop,
        posterior_f=args.posterior_f,
        posterior_y=args.posterior_y,
        posterior_z=args.posterior_z,
        mean_field=args.mean_field,
    )
    
    # Checkpoints
    
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        p.load_state_dict(ckpt['p_state_dict'])            
        p = p.to(torch.device(device))
        q.load_state_dict(ckpt['q_state_dict'])            
        q = q.to(torch.device(device))
        # Get optimisers
        p_opt = get_optimiser(args.gen_opt, p.parameters(), args.gen_lr, args.gen_l2)
        q_opt = get_optimiser(args.inf_opt, q.parameters(), args.inf_lr, args.inf_l2)
        if load_opt:
            p_opt.load_state_dict(ckpt['p_opt_state_dict'])
            q_opt.load_state_dict(ckpt['q_opt_state_dict'])
        stats_tr = ckpt['stats_tr']
        stats_val = ckpt['stats_val']
    else:
        p = p.to(torch.device(device))
        q = q.to(torch.device(device))
        # Get optimisers
        p_opt = get_optimiser(args.gen_opt, p.parameters(), args.gen_lr, args.gen_l2)
        q_opt = get_optimiser(args.inf_opt, q.parameters(), args.inf_lr, args.inf_l2)
        stats_tr = defaultdict(list)
        stats_val = defaultdict(list)

    # Training 
    
    vae = VAE(p, q,
        use_self_critic=args.use_self_critic, 
        use_reward_standardisation=args.use_reward_standardisation
    )


    state = OrderedDict(vae=vae, p=p, q=q, p_opt=p_opt, q_opt=q_opt, stats_tr=stats_tr, stats_val=stats_val, args=args)
    return namedtuple("State", state.keys())(*state.values())


def default_cfg():
    cfg = OrderedDict(        
        # Data        
        batch_size=200,
        data_dir='tmp',
        height=28,
        width=28, 
        output_dir='.',
        wandb=False,
        wandb_watch=False,
        # CUDA
        seed=42,
        device='cuda:0',
        # Joint distribution    
        z_dim=32,     
        prior_z='gaussian 0.0 1.0',
        y_dim=10,    
        prior_f='gibbs 0.0',
        prior_y='dirichlet 1.0',
        hidden_dec_size=500,
        # Approximate posteriors
        posterior_z='gaussian',
        posterior_f='gibbs -10 10',
        posterior_y='dirichlet 1e-3 1e3',
        shared_concentrations=True,
        mean_field=False,
        hidden_enc_size=500,    
        # Training
        epochs=200,    
        # Evaluation
        num_samples=100,    
        # Optimisation & regularisation
        gen_opt="adam",
        gen_lr=1e-3,
        gen_l2=0.0,
        gen_p_drop=0.0,
        inf_opt="adam",
        inf_lr=1e-3,  
        inf_l2=0.0,  
        inf_p_drop=0.0,  # dropout for inference model is not well understood    
        grad_clip=5.0,
        load_ckpt=False,
        reset_opt=False,
        # Variance reduction
        use_self_critic=False,
        use_reward_standardisation=True,
    )
    return cfg


def load_cfg(path, **kwargs):
    with open(path) as f:
        cfg = json.load(open(path), object_hook=OrderedDict)
    known = default_cfg()
    for k, v in known.items():
        if k not in cfg:
            cfg[k] = v
            print(f"Setting {k} to default {v}", file=sys.stderr)
    for k, v in cfg.items():
        if k not in known:
            raise ValueError(f'Unknown cfg: {k}')
    for k, v in kwargs.items():
        if k in known:
            cfg[k] = v
            print(f"Overriding {k} to user choice {v}", file=sys.stderr)
        else:
            raise ValueError(f"Unknown hparam {k}")
    return cfg

def make_args(cfg: dict):
    return namedtuple("Config", cfg.keys())(*cfg.values())


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


def main(cfg_file, **kwargs):
    
    cfg = load_cfg(cfg_file, **kwargs)
    if cfg.get('wandb', False):
        wandb.init(project='neurips21', config=cfg)
        cfg['output_dir'] = f"{cfg['output_dir']}/{wandb.run.name}"
        print(f"Output directory: {cfg['output_dir']}", file=sys.stderr)
    args = namedtuple('Config', cfg.keys())(*cfg.values())
    
    # Reproducibility
    print(f"# Reproducibility\nSetting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)    

    # Make dirs
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(f"{args.output_dir}/cfg.json", "w"), indent=4)
    #copyfile(cfg_file, f"{args.output_dir}/cfg.json")
    
    # Preparing data
    print("# Preparing MNIST (may take some time the first time)", file=sys.stderr)
    train_loader, valid_loader, test_loader = load_mnist(
        args.batch_size, 
        save_to=args.data_dir, 
        height=args.height, 
        width=args.width
    )

    print("# Building model", file=sys.stderr)
    state = make_state(
        args, 
        device=args.device, 
        ckpt_path=f"{args.output_dir}/training.ckpt" if args.load_ckpt else None, 
        load_opt=not args.reset_opt
    )
    #print(f"Generative model\n{state.p}", file=sys.stderr)
    #print(f"Inference model\n{state.q}", file=sys.stderr)

    # Training 
    
    # Demo 
    print("Example:\n 2 samples from inference model", file=sys.stderr)
    print(state.q.sample(state.p.sample((2,))[-1]), file=sys.stderr)
    print(" their log probability density under q", file=sys.stderr)
    print(state.q.log_prob(
        torch.zeros(args.height * args.width, device=torch.device(args.device)),
        *state.q.sample(state.p.sample((2,))[-1]), 
    ), file=sys.stderr)    
   
    
    print("# Training", file=sys.stderr)
    val_metrics = validate(state.vae, get_batcher(valid_loader, args), args.num_samples)
    print(f'Validation {0:3d}: nll={val_metrics[0]:.2f} bpd={val_metrics[1]:.2f}', file=sys.stderr)

    if args.wandb and args.wandb_watch:
        wandb.watch(state.p)
        wandb.watch(state.q)

    for epoch in range(args.epochs):

        iterator = tqdm(get_batcher(train_loader, args))

        for i, (x_obs, c_obs) in enumerate(iterator):
            # [B, H*W]
            x_obs = x_obs.reshape(-1, args.height * args.width)
            
            state.vae.train()      
            #if i % 20 == 0:
            #    samples, images = dict(), dict()
            #else:
            #    samples, images = None, None
            samples, images = None, None

            loss, ret = state.vae.loss(x_obs, c_obs, samples=samples, images=images)

            for k, v in ret.items():
                state.stats_tr[k].append(v)

            state.p_opt.zero_grad()
            state.q_opt.zero_grad()        
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                chain(state.vae.gen_parameters(), state.vae.inf_parameters()), 
                args.grad_clip
            )        
            state.p_opt.step()
            state.q_opt.step()

            iterator.set_description(f'Epoch {epoch+1:3d}/{args.epochs}')
            iterator.set_postfix(ret)

            if args.wandb and i % 50 == 0:
                wandb.log({f"training.{k}": v for k, v in ret.items()}, commit=False)
                if samples:
                    wandb.log({f"training.{k}": v for k, v in samples.items()}, commit=False)
                if images:
                    wandb.log({f"training.{k}": wandb.Image(v) for k, v in images.items()})


        val_metrics = validate(state.vae, get_batcher(valid_loader, args), args.num_samples, compute_DR=True)
        state.stats_val['val_nll'].append(val_metrics[0])
        state.stats_val['val_bpd'].append(val_metrics[1])
        print(f'Validation {epoch+1:3d}: nll={val_metrics[0]:.2f} bpd={val_metrics[1]:.2f}', file=sys.stderr)
        if args.wandb:
            wandb.log({'val.nll': val_metrics[0], 'val.bpd': val_metrics[1]}, commit=False)
            wandb.log({f"val.{k}": v for k, v in val_metrics[2].items()})
        
        torch.save({
            'p_state_dict': state.p.state_dict(),
            'q_state_dict': state.q.state_dict(),
            'p_opt_state_dict': state.p_opt.state_dict(),
            'q_opt_state_dict': state.q_opt.state_dict(),
            'stats_tr': state.stats_tr,
            'stats_val': state.stats_val,
        }, f"{args.output_dir}/training.ckpt")
    
    
    #np_stats_tr = {k: np.array(v) for k, v in stats_tr.items()}
    #np_stats_val = {k: np.array(v) for k, v in stats_val.items()}
    
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
        state.vae, get_batcher(valid_loader, args), args.num_samples, compute_DR=True)
    rows = [('NLL', val_nll, None), ('BPD', val_bpd, None)]
    for k, v in val_DR.items():
        rows.append((k, v.mean(), v.std()))
    print(tabulate(rows, headers=['metric', 'mean', 'std']))    
    return rows, ['metric', 'mean', 'std']
    
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
    main(sys.argv[1], load_ckpt='loadckpt' in sys.argv[2:], reset_opt='resetopt' in sys.argv[2:], seed=22)
