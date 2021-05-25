from main import main as train
from main import load_cfg
from tabulate import tabulate

name = 'gaussian-vae'

rows, headers = train(
    load_cfg(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}/debug-gsp', 
        device='cuda:0', 
        wandb=False, 
        epochs=4,
        batch_size=10,
        num_samples=10,
        #prior_z="gaussian 0.0 1.0",
        prior_z="gaussian-sparsemax-max-ent 1",
        posterior_z="gaussian-sparsemax",
        gen_opt="adam",
        gen_lr=1e-4,
        inf_opt="adam",
        inf_lr=1e-4,
        inf_l2=1e-6,
    )
)
