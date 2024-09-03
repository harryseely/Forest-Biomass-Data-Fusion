
from optuna.samplers import TPESampler
from ray.tune.search.optuna import OptunaSearch
import torch
import yaml

from hp_tuning_utils import tune_model
from ray import tune

if __name__ == "__main__":


    search_space = {

        'lr': tune.loguniform(1e-5, 1e-1),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'dropout_final': tune.quniform(0.1, 0.5, 0.1),
        'ocnn_dropout': tune.quniform(0.1, 0.5, 0.1),
    }

    search_alg = OptunaSearch(sampler=TPESampler())
    resources_per_trial = {"cpu": 1, "gpu": 1}

    # Check the number of GPUs available
    n_gpus = torch.cuda.device_count()

    #Load the static cfg
    with open(r"config.yaml", 'r') as file:
        static_cfg = yaml.safe_load(file)

    #Update cfg so we are tuning the correct model
    static_cfg['ocnn_lenet'] = True
    static_cfg['spec_cnn'] = False
    static_cfg['terrain_cnn'] = False

    #Implement tuning
    tune_model(search_space,
               resources_per_trial,
               n_concurrent_trials=n_gpus,
               wandb_project="base ocnn tuning",
               search_alg=search_alg,
               num_samples=100,
               time_budget_s=60 * 60 * 24 * 7,
               cfg=static_cfg,
               test=True,
               )
