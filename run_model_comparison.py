import torch
import yaml
from pathlib import Path
from ray import tune

from hp_tuning.hp_tuning_utils import tune_model as run_trials

if __name__ == "__main__":

    n_data_fold_iters = 5

    search_space = {

        'data_fold': tune.grid_search([1, 2, 3, 4, 5]),
    }

    search_alg = None
    resources_per_trial = {"cpu": 1, "gpu": 1}

    # Check the number of GPUs available
    n_gpus = torch.cuda.device_count()

    #Loop through each model config file
    config_dir = r"optimal_hyperparameters"
    model_configs = list(Path(config_dir).glob("*" + 'yaml'))

    #Print filepaths
    print("Looping through the following config files:\n",
          "\n".join([str(model_config) for model_config in model_configs]))

    #Load base config for general configurations
    with open(r"config.yaml", 'r') as file:
        base_cfg = yaml.safe_load(file)

    #Iterate through each model variant
    for model_config in model_configs:
        with open(model_config, 'r') as file:
            tuned_cfg = yaml.safe_load(file)

        #Update the main config with the model specific config
        base_cfg.update(tuned_cfg)

        run_trials(search_space,
                   resources_per_trial,
                   n_concurrent_trials=n_gpus,
                   wandb_project="Compare fusion modules",
                   search_alg=search_alg,
                   num_samples=n_data_fold_iters,
                   time_budget_s=60 * 60 * 24 * 7,
                   cfg=base_cfg,
                   test=True,
                   )
