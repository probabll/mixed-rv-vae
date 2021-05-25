from main import main as train
from main import load_cfg
from tabulate import tabulate

name = 'mixed-vae'

rows, headers = train(
    load_cfg(
        f'cfg/{name}.json', 
        seed=10, 
        output_dir=f'neurips-mixed-rv/cmp/{name}', 
        device='cuda:1', 
        wandb=True, 
        epochs=200,
        batch_size=200,
        prior_f="gibbs-max-ent 1",
        prior_y="dirichlet 1.0",
        posterior_f="gibbs -10 10",
        gen_opt="adam",
        gen_lr=1e-5,
        gen_l2=0.,
        gen_p_drop=0.1,
        inf_opt="adam",
        inf_lr=1e-5,
        inf_l2=1e-7,
        inf_p_drop=0.2,
        use_self_critic=False,
        use_reward_standardisation=True,
        shared_concentrations=True,
        training_samples=1, 
        exact_marginal=True,
        y_dim=10,
    )
)
