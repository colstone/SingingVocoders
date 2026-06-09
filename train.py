import importlib
import logging
import os
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')  # headless: set before any pyplot import -> no Tk backend (avoids Tcl_AsyncDelete crash)

import click
import lightning.pytorch as pl
import torch.utils.data
import yaml
from lightning.pytorch.loggers import TensorBoardLogger


from utils.config_utils import read_full_config, print_config
from utils.training_utils import (
    DsModelCheckpoint, DsTQDMProgressBar,
    get_latest_checkpoint_path, get_strategy
)

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def _prepare_audio_list(audio):
    import numpy as np

    if hasattr(audio, 'detach'):
        audio = audio.detach()
    if hasattr(audio, 'cpu'):
        audio = audio.cpu()
    audio_np = np.array(audio)
    if audio_np.ndim == 3 and audio_np.shape[1] == 1:
        audio_np = audio_np[:, 0, :]
    if audio_np.ndim == 1:
        audio_np = audio_np[None, :]
    elif audio_np.ndim < 2:
        audio_np = np.array([audio_np]).reshape(1, -1)
    return [audio_np[i] for i in range(audio_np.shape[0])]


def build_logger(logger_type, work_dir: pathlib.Path, exp_name: str):
    logger_type = (logger_type or 'tensorboard').lower()
    if logger_type == 'tensorboard':
        return TensorBoardLogger(
            save_dir=str(work_dir),
            name='lightning_logs',
            version='lastest'
        )

    if logger_type == 'wandb':
        try:
            from lightning.pytorch.loggers import WandbLogger
            from lightning.pytorch.loggers.logger import rank_zero_experiment
        except Exception as exc:
            raise ImportError("Please install wandb to enable the wandb logger.") from exc

        class WandbLoggerWithMedia(WandbLogger):
            @property
            @rank_zero_experiment
            def experiment(self):
                base_run = super().experiment

                class _WandbExperimentAdapter:
                    def __init__(self, run):
                        self._run = run

                    def __getattr__(self, item):
                        return getattr(self._run, item)

                    def add_audio(self, tag, audio, sample_rate, global_step=None):
                        import wandb

                        audio_list = _prepare_audio_list(audio)
                        payload = [wandb.Audio(a, sample_rate=sample_rate) for a in audio_list]
                        self._run.log({tag: payload if len(payload) > 1 else payload[0]}, step=global_step)

                    def add_figure(self, name, fig, global_step=None):
                        import wandb

                        self._run.log({name: wandb.Image(fig)}, step=global_step)

                return _WandbExperimentAdapter(base_run)

        return WandbLoggerWithMedia(
            project=exp_name,
            name=exp_name,
            save_dir=str(work_dir),
            log_model=False,
        )

    if logger_type == 'swanlab':
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger
            from lightning.pytorch.loggers.logger import rank_zero_experiment
        except Exception as exc:
            raise ImportError("Please install swanlab to enable the swanlab logger.") from exc

        class SwanLabLoggerWithMedia(SwanLabLogger):
            @property
            @rank_zero_experiment
            def experiment(self):
                run = super().experiment

                class _SwanLabExperimentAdapter:
                    def __init__(self, inner_run, logger_obj):
                        self._run = inner_run
                        self._logger = logger_obj

                    def __getattr__(self, item):
                        return getattr(self._run, item)

                    def add_audio(self, tag, audio, sample_rate, global_step=None):
                        audio_list = _prepare_audio_list(audio)
                        sr_list = [sample_rate] * len(audio_list)
                        self._logger.log_audio(tag, audio_list, step=global_step, sample_rate=sr_list)

                    def add_figure(self, name, fig, global_step=None):
                        self._logger.log_image(name, [fig], step=global_step)

                return _SwanLabExperimentAdapter(run, self)

        return SwanLabLoggerWithMedia(
            project=exp_name,
            experiment_name=exp_name,
            save_dir=str(work_dir),
        )

    raise ValueError(f"Unsupported logger type '{logger_type}'. Use 'tensorboard', 'wandb' or 'swanlab'.")


@click.command(help='')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
@click.option('--exp_name', required=True, metavar='EXP', help='Name of the experiment')
@click.option('--work_dir', required=False, metavar='DIR', help='Directory to save the experiment')
def train(config, exp_name, work_dir):
    config = pathlib.Path(config)
    config = read_full_config(config)
    print_config(config)
    if work_dir is None:
        work_dir = pathlib.Path(__file__).parent / 'experiments'
    else:
        work_dir = pathlib.Path(work_dir)
    work_dir = work_dir / exp_name
    assert not work_dir.exists() or work_dir.is_dir(), f'Path \'{work_dir}\' is not a directory.'
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / 'config.yaml', 'w', encoding='utf8') as f:
        yaml.safe_dump(config, f)
    config.update({'work_dir':str(work_dir)})

    if not config['nccl_p2p']:
        print("Disabling NCCL P2P")
        os.environ['NCCL_P2P_DISABLE'] = '1'

    pl.seed_everything(config['seed'], workers=True)
    assert config['task_cls'] != ''
    pkg = ".".join(config["task_cls"].split(".")[:-1])
    cls_name = config["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    # assert issubclass(task_cls, training.BaseTask), f'Task class {task_cls} is not a subclass of {training.BaseTask}.'

    task = task_cls(config=config)

    # work_dir = pathlib.Path(config['work_dir'])
    callbacks = [
        DsModelCheckpoint(
            dirpath=work_dir,
            filename='model_ckpt_steps_{step}',
            auto_insert_metric_name=False,
            monitor='step',
            mode='max',
            save_last=False,
            # every_n_train_steps=config['val_check_interval'],
            save_top_k=config['num_ckpt_keep'],
            permanent_ckpt_start=config['permanent_ckpt_start'],
            permanent_ckpt_interval=config['permanent_ckpt_interval'],
            verbose=True
        ),
        DsTQDMProgressBar(),
    ]
    # Optional: keep the BEST-val checkpoint (overfit guard) + early stopping.
    # Config-gated so non-FCPE tasks (which may not log the metric) are unaffected.
    best_monitor = config.get('best_ckpt_monitor')  # e.g. 'val/committed_accuracy'
    if best_monitor:
        from lightning.pytorch.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            dirpath=work_dir,
            filename='best_{step}',
            auto_insert_metric_name=False,
            monitor=best_monitor,
            mode=config.get('best_ckpt_mode', 'max'),
            save_top_k=1,
            save_last=False,
            verbose=True,
        ))
    es_monitor = config.get('early_stopping_monitor')  # e.g. 'val/committed_accuracy'
    if es_monitor:
        from lightning.pytorch.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(
            monitor=es_monitor,
            mode=config.get('early_stopping_mode', 'max'),
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 0.0),
            verbose=True,
        ))
    trainer = pl.Trainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        num_nodes=config['pl_trainer_num_nodes'],
        strategy=get_strategy(
            config['pl_trainer_devices'],
            config['pl_trainer_num_nodes'],
            config['pl_trainer_accelerator'],
            config['pl_trainer_strategy'],
            config['pl_trainer_precision'],
        ),
        precision=config['pl_trainer_precision'],
        callbacks=callbacks,
        logger=build_logger(config.get('logger', 'tensorboard'), work_dir, exp_name),
        # gradient_clip_val=config['clip_grad_norm'],
        val_check_interval=config['val_check_interval'] ,#* config['accumulate_grad_batches'],
        # so this is global_steps
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        max_steps=config['max_updates'],
        use_distributed_sampler=True,
        num_sanity_val_steps=config['num_sanity_val_steps'],
        # accumulate_grad_batches=config['accumulate_grad_batches']
    )
    trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))


os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision


if __name__ == '__main__':
    train()
