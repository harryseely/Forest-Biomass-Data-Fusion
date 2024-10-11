import pytorch_lightning as pl
import os
import wandb
import yaml
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.loggers.wandb import WandbLogger

from utils.trainer import LitModel
from utils.dataset import BiomassDataModule
from utils.training_utils import get_callbacks, name_run
from utils.test_model import test_model


def train_fn(tune_config, wandb_project="misc", static_cfg=None, test=True):
    """
    :param tune_config: ray tune config object
    :param wandb_project: wandb project name
    :param static_cfg: dictionary containing hyperparameters that are not being tuned (i.e., static config)
    :param test: boolean, whether to test the model after training
    :return:
    """

    if static_cfg is None:
        # Load hps from yaml file and update with ray tune config
        with open(r'config.yaml', "r") as yamlfile:
            static_cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    #Ensure all keys in the tune_config are in the static cfg
    static_keys = static_cfg.keys()
    tune_keys = tune_config.keys()
    assert set(tune_keys).issubset(
        static_keys), f"Keys in tune_config are not in cfg. tune_keys: {tune_keys}, static_keys: {static_keys}"

    #Update cfg with tune_config parameters
    static_cfg.update(tune_config)
    cfg = static_cfg

    # Create a Lightning model
    model = LitModel(cfg)

    # Add run name and save dir to cfg
    cfg['run_name'] = name_run(cfg)
    cfg['save_dir'] = rf"{cfg['run_name']}"

    # set logger
    with open(cfg['wandb_key']) as f:
        wandb_key = f.readlines()[0]

    wandb.login(key=wandb_key)

    wandb_logger = WandbLogger(project=wandb_project, config=cfg, save_dir=os.getcwd())

    # Enable gradient clipping if specified
    if cfg['gradient_clip']:
        gradient_clip_val = 0.5
        gradient_clip_algorithm = "norm"
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None

    callback_list = get_callbacks(cfg, model_ckpt=True, progress_bar=False, lr_rate_monitor=False)

    callback_list.append(TuneReportCallback({"val_loss": "val_loss"}, on="validation_end"))

    # Create a Lighting Trainer
    trainer = pl.Trainer(
        max_epochs=cfg['num_epochs'],
        logger=wandb_logger,
        enable_progress_bar=False,
        # * * * * Gradient Clipping
        gradient_clip_val=gradient_clip_val,  # https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html
        gradient_clip_algorithm=gradient_clip_algorithm,
        # * * * * Callbacks
        callbacks=callback_list
    )

    # Build your datasets on each worker
    data_module = BiomassDataModule(cfg=cfg, logger=wandb_logger)

    #Implement training loop
    trainer.fit(model, datamodule=data_module)

    # Write the config dictionary to a YAML file
    with open(os.path.join(cfg['save_dir'], f"{cfg['run_name']}_config.yaml"), 'x') as file:
        yaml.dump(wandb.config._items, file)

    #If specified, test model
    if test:
        test_model(save_dir=cfg['save_dir'], logger=wandb_logger, fig_out_dir=cfg['save_dir'],
                   save_plots_to_wandb=False, view_plots=False)

    wandb.finish()


def tune_model(search_space, resources_per_trial=None, n_concurrent_trials=1, wandb_project="misc",
               search_alg=None, num_samples=-1, time_budget_s=60 * 60, cfg=None, test=True):
    """
    Function that tunes hyperparameters using ray tune.

    :param search_space:
    :param resources_per_trial:
    :param n_concurrent_trials:
    :param wandb_project:
    :param search_alg:
    :param num_samples:
    :param time_budget_s: -1 for infinite
    :param cfg: dictionary containing hyperparameters that are not being tuned (i.e., static config)
    :param test: boolean, whether to test the model after training
    :return:
    """
    if resources_per_trial is None:
        resources_per_trial = {"cpu": 1, "gpu": 1}

    reporter = CLIReporter(
        parameter_columns=list(search_space.keys()),
        metric_columns=["val_loss", "train_loss", "training_iteration"],
    )

    # Insert parameters for trainable function
    trainable_with_params = tune.with_parameters(train_fn, wandb_project=wandb_project, static_cfg=cfg, test=test)

    tuner = tune.Tuner(
        tune.with_resources(trainable_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            max_concurrent_trials=n_concurrent_trials,
            metric="val_loss",
            mode="min",
            scheduler=None,
            num_samples=num_samples,
            time_budget_s=time_budget_s,
        ),
        run_config=train.RunConfig(
            progress_reporter=reporter,

        ),
        param_space=search_space,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)