from main import main as train
from tabulate import tabulate

name = 'onehotcat-vae'

rows, headers = train(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}', 
        device='cuda:1', 
        wandb=True, 
        epochs=200,
        prior_z="onehotcat 1.0 0.0",
        posterior_z="onehotcat 1.0 -10 10",
        gen_opt="adam",
        gen_lr=5e-4,
        inf_opt="adam",
        inf_lr=5e-4,
        inf_l2=1e-6,
)
