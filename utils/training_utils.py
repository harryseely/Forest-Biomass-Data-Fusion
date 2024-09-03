from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner import Tuner
import torch.nn as nn
import sys
import wandb
import matplotlib.pyplot as plt
import torch
from datetime import datetime as dt
from itertools import compress
import random
import string


def name_run(cfg):
    nms_list = []

    if cfg['ocnn_lenet']:
        nms_list.append('ocnn_lenet')

    if cfg['spec_cnn']:
        nms_list.append('spectral_cnn')

    if cfg['terrain_cnn']:
        nms_list.append("terrain_cnn")

    nms_list = list(compress(nms_list, [x is not None for x in nms_list]))

    model_name = '_'.join(nms_list)

    fold = cfg['data_fold']

    # Add a random 4 character alphanumeric string to the run name
    rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

    # Generate run name
    run_name = f"{model_name}_fold{fold}_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}_id{rand_id}"

    return run_name


def get_callbacks(cfg, lr_rate_monitor, model_ckpt, progress_bar):
    callback_list = list()

    if model_ckpt:
        checkpoint_callback = ModelCheckpoint(dirpath=cfg['save_dir'],
                                              filename='{epoch}-{val_loss:.2f}-{val_r2:.2f}',
                                              monitor="val_loss",
                                              mode="min",
                                              save_top_k=1,
                                              every_n_epochs=1,
                                              )

        # Add to callback list
        callback_list.append(checkpoint_callback)

    if lr_rate_monitor:
        # Set up lr logging
        callback_list.append(LearningRateMonitor(logging_interval='step'))

    if progress_bar:
        # Add custom progress bar to callbacks list
        class MyProgressBar(TQDMProgressBar):
            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                if not sys.stdout.isatty():
                    bar.disable = True
                return bar

            def init_predict_tqdm(self):
                bar = super().init_predict_tqdm()
                if not sys.stdout.isatty():
                    bar.disable = True
                return bar

            def init_test_tqdm(self):
                bar = super().init_test_tqdm()
                if not sys.stdout.isatty():
                    bar.disable = True
                return bar

        callback_list.append(MyProgressBar())

    if cfg['early_stopping']:
        # Add early stopping to callback list
        callback_list.append(EarlyStopping(monitor="val_loss",
                                           mode="min",
                                           min_delta=0.01,
                                           patience=cfg['patience']))

    # Set callback list to None if it contains zero elements
    if len(callback_list) == 0:
        callback_list = None

    return callback_list


def run_lr_finder(cfg, trainer, model, data_module, wandb_logger):
    # Implement auto lr finder
    tuner = Tuner(trainer)
    model.hparams['lr'] = cfg['lr']  # Ensure model has lr hyperparameter so it can be updated
    lr_finder = tuner.lr_find(model, datamodule=data_module)
    # Only log new lr on main device (0)
    if torch.cuda.current_device() == 0:
        new_lr = lr_finder.suggestion()
        print(f"LR finder identified:\n{new_lr}\nas candidate learning rate")
        fig = lr_finder.plot(suggest=True)
        plot_fpath = "temp/lr_finder_plot.jpg"
        plt.savefig(plot_fpath)
        fig.show()

        # Save plot as a artifact in wandb
        if wandb_logger is None:
            pass
        else:
            wandb_logger.log_image(key="lr_finder_plot", images=[plot_fpath])

            # Set new lr hp in wandb config
            wandb.config.update({"lr": new_lr}, allow_val_change=True)


class LossFunction:
    """
    Class to handle different loss functions
    """

    def __init__(self, loss_fn_name):
        """

        :param loss_fn_name:
        """
        super(LossFunction, self).__init__()

        self.loss_fn_name = loss_fn_name

        if loss_fn_name == "smooth_l1":
            self.loss_fn = torch.nn.SmoothL1Loss()

        elif loss_fn_name == "mse":
            self.loss_fn = torch.nn.MSELoss()

        else:
            raise Exception(f"{loss_fn_name} loss function is not supported")

    def __call__(self, pred, y):
        """
        :param pred:
        :param y:
        :return:
        """


        loss_bark = self.loss_fn(pred[:, 0], y[:, 0])
        loss_branch = self.loss_fn(pred[:, 1], y[:, 1])
        loss_foliage = self.loss_fn(pred[:, 2], y[:, 2])
        loss_wood = self.loss_fn(pred[:, 3], y[:, 3])

        # Take the mean of the four components loss
        loss = torch.mean(torch.stack([loss_bark, loss_branch, loss_foliage, loss_wood]))

        return loss


def forward_pass(model, batch):
    """

    :param model:
    :param batch:
    :return:
    """

    pred = model(batch)


    return pred


def get_activation(fn_name):
    if fn_name == 'relu':
        return nn.ReLU(inplace=True)
    elif fn_name == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif fn_name == 'elu':
        return nn.ELU(inplace=True)
    else:
        raise ValueError(f'Invalid activation function: {fn_name}')
