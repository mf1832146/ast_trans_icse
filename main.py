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
        # optimize(parameters=config.hype_parameters,
        #          evaluation_function=lambda params: run(config, params),
        #          objective_name='bleu')
        run(config, config.hype_parameters)
    else:
        run(config)

