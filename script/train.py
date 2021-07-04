import torch
import torch.nn as nn
import ignite.distributed as idist
from ignite.contrib.handlers import tensorboard_logger, clearml_logger
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.utils import setup_logger, convert_tensor
from ignite.contrib.engines import common
import ignite

from py_config_runner.utils import set_seed
from py_config_runner.config_utils import get_params, Schema
from pathlib import Path
from clearml import Task
import torch.utils.data as data
from torch.utils.data import DataLoader

from dataset import get_data_set
from utils import exp_tracking


def initialize(config):
    model = config.model.to(config.device)
    optimizer = config.optimizer
    if config.multi_gpu:
        model = idist.auto_model(model)
        optimizer = idist.auto_optim(optimizer)
    criterion = config.criterion
    return model, optimizer, criterion


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed} - {tag} metrics:\n {metrics_output}")


def get_dataflow(config):
    if idist.get_local_rank() > 0:
        idist.barrier()

    train_dataset, valid_dataset = get_data_set(config)

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    return train_dataset, valid_dataset


def get_data_loader(config, is_train, data_set):
    batch_size = config.batch_size
    is_shuffle = True if is_train else False
    num_workers = config.num_threads if is_train else 0
    if config.multi_gpu:
        data_loader = idist.auto_dataloader(dataset=data_set,
                                            batch_size=batch_size,
                                            shuffle=is_shuffle,
                                            collate_fn=data_set.collect_fn,
                                            pin_memory='cuda' in idist.device().type,
                                            num_workers=num_workers)
    else:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=batch_size,
                                 shuffle=is_shuffle,
                                 collate_fn=data_set.collect_fn,
                                 num_workers=num_workers)
    return data_loader


def training(local_rank, logger=None, config=None):
    if idist.get_rank() == 0:
        task = Task.init(project_name=config.project_name, task_name=config.task_name)
        task.connect_configuration(config.config_filepath.as_posix())
        exp_tracking.log_params(get_params(config, config.schema))

    set_seed(config.seed + local_rank)
    train_data_set, eval_data_set = get_dataflow(config)
    train_loader = get_data_loader(config, is_train=True, data_set=train_data_set)
    valid_loader = get_data_loader(config, is_train=True, data_set=eval_data_set)

    # Setup model, optimizer, criterion
    model, optimizer, criterion = initialize(config)
    trainer = create_supervised_trainer(model, optimizer, criterion, config.device)
    greedy_generator = GreedyGenerator(model, len(config.tgt_vocab.w2i), config.max_tgt_len,
                                       multi_gpu=config.multi_gpu)

    metrics_valid = {}







