import yaml
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import wandb
from pprint import pprint
import os

from utils.trainer import LitModel
from utils.dataset import BiomassDataModule
from utils.test_model import test_model
from utils.training_utils import get_callbacks, run_lr_finder, name_run

def main(cfg):

    # Set run name and ensure all devices use same name
    run_name = name_run(cfg)
    print(f"Current training run is called: {run_name}")

    # Set up save directory where config, checkpoints, and other things will be saved
    save_dir = os.path.join(os.getcwd(), os.path.join("checkpoints_and_config", run_name))
    model_ckpt = True

    # Set matrix multiplication precision https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision('medium')

    # Set up wandb logger
    logging = True if cfg['num_epochs'] > 2 else False
    if logging:

        # Add save_dir for run to cfg before it is logged in wandb
        cfg["save_dir"] = save_dir

        # More info about logging: https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.loggers.wandb.html
        # Best practices for wandb: https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1
        with open(cfg['wandb_key']) as f:
            wandb_key = f.readlines()[0]

        wandb.login(key=wandb_key)

        # Integrate wandb logger with lightning
        wandb_logger = WandbLogger(log_model="all",
                                   checkpoint_name=f'ckpt-{run_name}',
                                   name=run_name,
                                   config=cfg,
                                   id=run_name,
                                   project="RQ2",
                                   save_code=True,
                                   mode="online",
                                   allow_val_change=True,
                                   job_type='training',  # for organizing runs (e.g. preprocessing vs. training)
                                   resume="allow"
                                   )
    else:
        wandb_logger = None
        save_dir = None
        model_ckpt = False

    # Enable gradient clipping if specified
    if cfg['gradient_clip']:
        gradient_clip_val = 0.5
        gradient_clip_algorithm = "norm"
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None

    callback_list = get_callbacks(cfg, model_ckpt=model_ckpt, progress_bar=True, lr_rate_monitor=logging)

    # Set up gpu strategy
    if cfg['ddp']:

        # Explicitly specify the process group backend if you choose to
        gpu_strategy = DDPStrategy(process_group_backend="gloo")

        #Overide specified gpu ids and get IDs for all available GPUs
        gpu_ids = list(range(0, torch.cuda.device_count()))

    else:
        gpu_ids = [0]
        gpu_strategy = "auto"

    # Set up trainer
    # Ref docs for lightning https://lightning.ai/docs/pytorch/stable/common/trainer.html
    trainer = pl.Trainer(

        # * * * * Parallelization and resources
        strategy=gpu_strategy,
        devices=gpu_ids,
        accelerator="gpu",
        benchmark=True,  # torch.backends.cudnn.benchmark
        # Implementing FP16 Mixed Precision Training https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
        precision=cfg['precision'],

        # * * * * Gradient Clipping
        gradient_clip_val=gradient_clip_val,  # https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html
        gradient_clip_algorithm=gradient_clip_algorithm,
        # * * * * Duration Params
        max_epochs=cfg['num_epochs'],
        max_time=None,

        # * * * * Logging
        enable_progress_bar=True,
        logger=wandb_logger,
        default_root_dir="lr_finder_logs",

        # * * * * Debugging
        fast_dev_run=False,  # Run 1 batch for debugging
        profiler=None,  # Profiles the train epoch base to identify bottlenecks set to None to turn off
        detect_anomaly=False,  # Enable anomaly detection for the autograd engine

        # * * * * Callbacks
        callbacks=callback_list,
    )

    print(f"Config:\n")
    pprint(cfg)

    # Load data
    data_module = BiomassDataModule(cfg=cfg, logger=wandb_logger)

    # Wrap in lightning module
    model = LitModel(cfg)

    if cfg['lr_finder']:
        run_lr_finder(cfg, trainer, model, data_module, wandb_logger)

    # Implement training
    trainer.fit(model, datamodule=data_module)

    # Stop DDP
    if cfg['ddp']:
        torch.distributed.destroy_process_group()

    # Save the config as a yaml (before model fit, in case of error)
    if save_dir is not None and torch.cuda.current_device() == 0:
        # Write the config dictionary to a YAML file
        with open(os.path.join(save_dir, f"{run_name}_config.yaml"), 'x') as file:
            yaml.dump(wandb.config._items, file)

    # Test model
    if torch.cuda.current_device() == 0 and logging:
        test_model(save_dir=save_dir, logger=wandb_logger)

    # Finish
    if logging:
        wandb.finish()

    return


if __name__ == "__main__":
    # Read config
    with open("config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    main(cfg=cfg)
