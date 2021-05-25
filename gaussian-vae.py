from main import main as train
from main import load_cfg
from tabulate import tabulate

name = 'gaussian-vae'

rows, headers = train(
    load_cfg(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}/debug-test', 
        device='cuda:0', 
        wandb=False, 
        epochs=4,
        prior_z="gaussian 0.0 1.0",
        posterior_z="gaussian",
        gen_opt="adam",
        gen_lr=1e-3,
        inf_opt="adam",
        inf_lr=1e-3,
        inf_l2=1e-6,
    )
)
