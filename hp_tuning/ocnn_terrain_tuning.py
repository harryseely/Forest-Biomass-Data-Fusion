
from optuna.samplers import TPESampler
from ray.tune.search.optuna import OptunaSearch
import torch
import yaml

from hp_tuning_utils import tune_model
from ray import tune

if __name__ == "__main__":

    search_space = {

        #General HPs
        'lr': tune.loguniform(1e-5, 1e-1),
        'batch_size': tune.choice([16, 32, 64, 128]),
        'dropout_final': tune.quniform(0.1, 0.5, 0.1),

        #Terrain CNN
        'terrain_cnn_n_neurons_1': tune.choice([16, 32, 64, 128, 256, 512]),
        'terrain_cnn_n_neurons_2': tune.choice([16, 32, 64, 128, 256, 512]),
        'terrain_cnn_neuron_mult': tune.randint(1, 3),
        'terrain_cnn_n_layers': tune.randint(1, 5),
        'terrain_cnn_conv_kernel_size': tune.randint(2, 3),
        'terrain_cnn_pool_kernel_size': tune.randint(2, 3),
        'terrain_cnn_stride_1': tune.randint(1, 3),
        'terrain_cnn_stride': tune.randint(1, 2),
        'terrain_cnn_dilation_rate': tune.choice([1, 2, 3]),
        'terrain_cnn_pooling_type': tune.choice(['max', 'avg']),
        'terrain_cnn_activation_fn': tune.choice(['relu', 'leaky_relu', 'elu']),
        'terrain_cnn_dropout': tune.quniform(0.1, 0.5, 0.1),
    }

    search_alg = OptunaSearch(sampler=TPESampler())

    resources_per_trial = {"cpu": 1, "gpu": 1}

    # Check the number of GPUs available
    n_gpus = torch.cuda.device_count()

    # Load the static cfg
    with open(r'config.yaml', "r") as yamlfile:
        static_cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Update cfg so we are tuning the correct model
    static_cfg['ocnn_lenet'] = True
    static_cfg['spec_cnn'] = False
    static_cfg['terrain_cnn'] = True

    #Implement tuning
    tune_model(search_space,
               resources_per_trial,
               n_concurrent_trials=n_gpus,
               wandb_project="ocnn + terrain tuning",
               search_alg=search_alg,
               num_samples=100,
               time_budget_s=60 * 60 * 24 * 7,
               cfg=static_cfg,
               test=True,
               )
