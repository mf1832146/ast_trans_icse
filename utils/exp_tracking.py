import os
from pathlib import Path
from ignite.contrib.engines import common
import ignite.distributed as idist
from clearml import Task

if "CLEARML_OUTPUT_PATH" not in os.environ:
    os.environ["CLEARML_OUTPUT_PATH"] = './clearml/'


has_clearml = True
clearml_output_path = None


def _clearml_get_output_path():
    global clearml_output_path

    if clearml_output_path is None:
        from datetime import datetime

        output_path = Path(os.environ["CLEARML_OUTPUT_PATH"])
        output_path = output_path / "clearml" / datetime.now().strftime("%Y%m%d-%H%M%S")
        clearml_output_path = output_path

    return clearml_output_path.as_posix()


@idist.one_rank_only()
def _clearml_log_artifact(fp):
    task = Task.current_task()
    task.upload_artifact(Path(fp).name, fp)


@idist.one_rank_only()
def _clearml_log_params(params_dict):
    task = Task.current_task()
    task.connect(params_dict)


if has_clearml:
    get_output_path = _clearml_get_output_path
    log_params = _clearml_log_params
    setup_logging = common.setup_clearml_logging
    log_artifact = _clearml_log_artifact
else:
    raise RuntimeError(
        "No experiment tracking system is setup. "
        "Please, setup either ClearML. "
    )
