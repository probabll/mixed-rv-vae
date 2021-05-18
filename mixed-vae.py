from main import main as train
from tabulate import tabulate

name = 'mixed-vae'

rows, headers = train(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}/', 
        device='cuda:0', 
        wandb=True, 
        epochs=200,
        prior_f="gibbs -1.0",
        prior_y="dirichlet 1.0",
        gen_opt="adam",
        gen_lr=5e-4,
        inf_opt="adam",
        inf_lr=5e-4,
        inf_l2=1e-6,
        use_self_critic=False,
        use_reward_standardisation=True,
)
