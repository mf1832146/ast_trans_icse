import argparse
from pathlib import Path

from py_config_runner import ConfigObject
from ax.service.managed_loop import optimize
from script import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Example application")
    parser.add_argument("--config", type=Path, help="Input configuration file")
    parser.add_argument('--use_hype_params', action='store_true')
    args = parser.parse_args()

    assert args.config is not None
    assert args.config.exists()

    config = ConfigObject(args.config)
    if args.use_hype_params:
        config.hype_parameters = [
            {
                "name": "max_rel_pos",
                "type": "choice",
                "values": [1, 3, 5, 7]
            },
            {
                "name": "par_heads",
                "type": "range",
                "bounds": [0, 8],
                "value_type": "int"
            },
            {
                "name": "pos_type",
                "type": "choice",
                "values": ['', 'p2q_p2k', 'p2q_p2k_p2v']
            }
        ]
        optimize(parameters=config.hype_parameters,
                 evaluation_function=lambda params: run(config, params),
                 objective_name='bleu')
        #run(config, config.hype_parameters)
    else:
        run(config)


