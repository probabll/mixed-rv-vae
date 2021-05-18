from main import main as train
from tabulate import tabulate

name = 'dirichlet-vae'

rows, headers = train(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}', 
        device='cuda:1', 
        wandb=True, 
        epochs=200,
        prior_z="dirichlet 0.1",
        posterior_z="dirichlet 1e-3 1e3",
        gen_opt="adam",
        gen_lr=1e-3,
        inf_opt="adam",
        inf_lr=1e-3,
        inf_l2=1e-6,
)
