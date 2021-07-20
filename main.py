import argparse
import os
from pathlib import Path

from py_config_runner import ConfigObject
from ax.service.managed_loop import optimize
from script import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Example application")
    parser.add_argument("--config", type=Path, help="Input configuration file")
    parser.add_argument('--use_hype_params', action='store_true')
    parser.add_argument('--data_type', type=str, default='')
    parser.add_argument('--g', type=str, default='')
    args = parser.parse_args()

    assert args.config is not None
    assert args.config.exists()

    config = ConfigObject(args.config)

    if args.g != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.g
        config.device = "cuda"
        config.g = args.g
        if len(args.g.split(',')) > 1:
            config.multi_gpu = True
            config.batch_size = config.batch_size * len(args.g.split(','))
        else:
            config.multi_gpu = False
    else:
        config.device = 'cpu'
        config.multi_gpu = False

    if args.use_hype_params:
        config.hype_parameters = [
            {
                "name": "max_rel_pos",
                "type": "choice",
                "values": [1, 3, 5, 7],
                "value_type": "int"
            },
            {
                "name": "par_heads",
                "type": "choice",
                "values": [0, 2, 4, 6, 8],
                "value_type": "int"
            },
            {
                "name": "pos_type",
                "type": "choice",
                "values": ['', 'p2q_p2k', 'p2q_p2k_p2v'],
                "value_type": "str"
            }
        ]
        optimize(parameters=config.hype_parameters,
                 evaluation_function=lambda params: run(config, params),
                 objective_name='bleu')
        #run(config, config.hype_parameters)
    else:
        if args.data_type != '':
            config.data_type = args.data_type
            config.task_name += args.data_type
        run(config)


