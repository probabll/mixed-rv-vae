from main import main as train
from tabulate import tabulate

name = 'mixed-vae'

rows, headers = train(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}/debugz2', 
        device='cuda:1', 
        wandb=False, 
        epochs=200,
        batch_size=100,
        prior_f="gibbs-max-ent 0",
        prior_y="dirichlet 1.0",
        posterior_f="gibbs -10 10",
        gen_opt="adam",
        gen_lr=1e-5,
        inf_opt="adam",
        inf_lr=1e-5,
        inf_l2=1e-6,
        use_self_critic=False,
        use_reward_standardisation=False,
        shared_concentrations=True,
        training_samples=1, 
        exact_marginal=True,
        gen_p_drop=0.1,
        inf_p_drop=0.1,
        y_dim=5,
)
