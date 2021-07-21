import ignite.distributed as idist
import math
import torch.optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.utils import setup_logger, convert_tensor
from ignite.contrib.engines import common
import json
from py_config_runner.utils import set_seed
from pathlib import Path
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from config import get_model
from module import GreedyGenerator
from utils import load_vocab
from valid_metrices.bleu_metrice import BLEU4, bleu_output_transform

__all__ = ['run']

from valid_metrices.compute_scores import eval_accuracies

valid_bleu = None


def _graph_prepare_batch(batch, device=None, non_blocking: bool = False):
    x, y = batch
    return (
        x.to(device),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def params2str(params):
    if params is None:
        return ''
    return '|'.join([' ' + str(key) + ': ' + str(value) for key, value in params.items()])


def initialize(config, train_data_set_len):
    model = get_model(config)
    model = model.to(config.device)
    t_total = math.ceil(train_data_set_len / config.batch_size) * config.num_epochs
    optimizer = BertAdam(model.parameters(), lr=config.learning_rate, warmup=config.warmup, t_total=t_total)
    if config.multi_gpu:
        model = idist.auto_model(model)
        optimizer = idist.auto_optim(optimizer)
    criterion = config.criterion
    return model, optimizer, criterion


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed} - {tag} metrics:\n {metrics_output}")


def get_dataflow(config):
    train_data_set = config.data_set(config, 'train')
    eval_data_set = config.data_set(config, 'dev')
    return train_data_set, eval_data_set


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


def training(local_rank, config=None, **kwargs):
    torch.cuda.empty_cache()
    logger = kwargs['logger']
    hype_params = kwargs['hype_params']
    if idist.get_rank() == 0:
        if config.use_clearml:
            from clearml import Task
            from utils import exp_tracking
            task = Task.init(project_name=config.project_name,
                             task_name=config.task_name + params2str(hype_params))
            task.connect_configuration(config.config_filepath.as_posix())
            if hype_params is not None:
                exp_tracking.log_params(hype_params)

    set_seed(config.seed + local_rank)
    train_data_set, eval_data_set = get_dataflow(config)
    train_loader = get_data_loader(config, is_train=True, data_set=train_data_set)
    valid_loader = get_data_loader(config, is_train=False, data_set=eval_data_set)
    config.checkpoint = None
    # Setup model, optimizer, criterion
    model, optimizer, criterion = initialize(config, train_data_set.__len__())

    trainer = create_supervised_trainer(model, optimizer, criterion, config.device,
                                        prepare_batch=_graph_prepare_batch)
    metrics_valid = {'bleu': BLEU4()}
    greedy_generator = GreedyGenerator(model, config.max_tgt_len, multi_gpu=config.multi_gpu)
    evaluator = create_supervised_evaluator(greedy_generator, metrics=metrics_valid, device=config.device,
                                            prepare_batch=_graph_prepare_batch,
                                            output_transform=lambda x, y, y_pred:
                                            bleu_output_transform((y_pred, y), config.tgt_vocab.i2w))

    # timer = Timer(average=True)
    # timer.attach(trainer,
    #              start=Events.EPOCH_STARTED,
    #              resume=Events.EPOCH_COMPLETED,
    #              pause=Events.EPOCH_COMPLETED,
    #              step=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED(every=getattr(config, 'val_interval', 1)) | Events.COMPLETED)
    def run_validation():
        epoch = trainer.state.epoch
        states = evaluator.run(valid_loader)
        log_metrics(logger, epoch, states.times['COMPLETED'], 'Test', states.metrics)

    common.save_best_model_by_val_score(
        config.output_path.as_posix(),
        evaluator,
        model=model,
        metric_name='bleu',
        n_saved=1,
        trainer=trainer,
        tag='val'
    )

    common.add_early_stopping_by_val_score(patience=4,
                                           evaluator=evaluator,
                                           trainer=trainer,
                                           metric_name='bleu')

    if idist.get_rank() == 0:
        ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})
        if not config.fast_mod:
            if 'tensorboard' in config.logger:
                from ignite.contrib.handlers import tensorboard_logger
                tb_logger = common.setup_tb_logging(
                    config.output_path.as_posix(),
                    trainer,
                    optimizer,
                    evaluators={'validation': evaluator}
                )

                tb_logger.attach(
                    trainer,
                    log_handler=tensorboard_logger.OutputHandler(
                        tag="training", output_transform=lambda loss: {"loss": loss}, metric_names="all"
                    ),
                    event_name=Events.ITERATION_COMPLETED(every=50)
                )

            if 'clear_ml' in config.logger:
                from ignite.contrib.handlers import clearml_logger
                exp_tracking_logger = exp_tracking.setup_logging(
                    trainer, optimizer, evaluators={"validation": evaluator}
                )

                exp_tracking_logger.attach(
                    trainer,
                    log_handler=clearml_logger.OutputHandler(
                        tag="training", output_transform=lambda loss: {"loss": loss}, metric_names="all"
                    ),
                    event_name=Events.ITERATION_COMPLETED(every=50)
                )

    trainer.run(train_loader, max_epochs=config.num_epochs)

    global valid_bleu
    valid_bleu = evaluator.state.metrics['bleu']

    test(local_rank, config, logger)

    if idist.get_rank() == 0 and not config.fast_mod:
        if config.use_clearml:
            task.close()
        if not config.fast_mod:
            if 'tensorboard' in config.logger:
                tb_logger.writer.add_hparams(hype_params, {'hparam/test_accuracy': valid_bleu})
                tb_logger.close()
            if 'clear_ml' in config.logger:
                exp_tracking_logger.writer.add_hparams(hype_params, {'hparam/test_accuracy': valid_bleu})
                exp_tracking_logger.close()


def test(local_rank, config, logger):
    if local_rank == 0:
        torch.cuda.empty_cache()
        output_path = config.output_path.as_posix()
        load_epoch_path = ''
        for file in os.listdir(output_path):
            if file.endswith('.pt'):
                logger.info('load ' + file)
                sub_dir = os.path.join(output_path, file)
                load_epoch_path = sub_dir

        if load_epoch_path == '':
            raise Exception('Can not find the save model. ')

        logger.info('*' * 5 + 'Start TEST' + '*'*5)

        if torch.cuda.is_available():
            checkpoint = torch.load(load_epoch_path)
        else:
            checkpoint = torch.load(load_epoch_path, map_location='cpu')
        config.checkpoint = checkpoint
        model = get_model(config)
        model.eval()
        model = model.to(config.device)
        greedy_generator = GreedyGenerator(model, config.max_tgt_len, multi_gpu=False)
        # metrics_test = {'total': TotalMetric(output_path=config.output_path.as_posix())}

        # evaluator = create_supervised_evaluator(greedy_generator, metrics=metrics_test, device=config.device,
        #                                         prepare_batch=_graph_prepare_batch,
        #                                         output_transform=lambda x, y, y_pred:
        #                                         bleu_output_transform((y_pred, y), config.tgt_vocab.i2w))

        test_data_set = config.data_set(config, 'test')
        test_loader = DataLoader(dataset=test_data_set,
                                 batch_size=config.batch_size // len(config.g.split(',')),
                                 shuffle=False,
                                 collate_fn=test_data_set.collect_fn)

        # states = evaluator.run(test_loader)
        # bleu, rougle_l, meteor = states.metrics['total']
        _hypothesises = []
        _references = []

        for batch in tqdm(test_loader):
            x, y = _graph_prepare_batch(batch, device=config.device)
            y_pred = greedy_generator(x)
            hypothesises, references = bleu_output_transform((y_pred, y), config.tgt_vocab.i2w)
            _hypothesises.extend(hypothesises)
            _references.extend(references)

        hypothesises = {index: [' '.join(value)] for index, value in enumerate(_hypothesises)}
        references = {index: [' '.join(value)] for index, value in enumerate(_references)}
        bleu, rougle_l, meteor, ind_bleu, ind_rouge = eval_accuracies(hypothesises, references)

        outputs = []
        for i in hypothesises.keys():
            outputs.append({
                'predict': hypothesises[i][0],
                'true': references[i][0],
                'bleu': ind_bleu[i],
                'rouge': ind_rouge[i]
            })

        file_name = "/predict_results_bleu_{:.2f}_rouge_{:.2f}_meteor_{:.2f}.json".format(bleu, rougle_l, meteor)
        with open(config.output_path.as_posix() + file_name, 'w') as f:
            json.dump(outputs, f)
        logger.info(f"bleu: {bleu}, rouge: {rougle_l} meteor: {meteor}")


def run(config, hype_params=None):
    if hype_params is not None:
        config.__internal_config_object_data_dict__.update(hype_params)
        config.max_rel_pos = max(config.max_par_rel_pos, config.max_bro_rel_pos)

    config.src_vocab, config.tgt_vocab = load_vocab(config.data_dir, config.is_split, config.data_type)

    logger = setup_logger(name='AST Transformer Training', distributed_rank=idist.get_rank())
    logger.info('Hype-Params: ' + params2str(hype_params))

    config.output_path = Path('./outputs/' + config.project_name + '/' + config.task_name + params2str(hype_params))
    config.output_path_str = config.output_path.as_posix()
    if not config.is_test:
        if config.multi_gpu:
            with idist.Parallel(backend="nccl", master_port=2224) as parallel:
                try:
                    parallel.run(training, config, logger=logger, hype_params=hype_params)
                except KeyboardInterrupt:
                    logger.info("Catched KeyboardInterrupt -> exit")
                except Exception as e:  # noqa
                    logger.exception("")
                    raise e
        else:
            training(0, config, logger=logger, hype_params=hype_params)
    else:
        test(0, config, logger=logger)

    global valid_bleu
    return valid_bleu







