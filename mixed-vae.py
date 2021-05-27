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
        epochs=500,
        batch_size=200,
        prior_f="gibbs-max-ent 0",
        prior_y="dirichlet 1.0",
        posterior_f="gibbs -10 10",
        gen_opt="rmsprop",
        gen_lr=5e-4,
        gen_l2=0.,
        gen_p_drop=0.1,
        inf_opt="adam",
        inf_lr=5e-4,
        inf_l2=1e-7,
        inf_p_drop=0.1,
        use_self_critic=True,
        use_reward_standardisation=False,
        shared_concentrations=True,
        training_samples=1, 
        exact_marginal=False,
        y_dim=10,
    )
)
