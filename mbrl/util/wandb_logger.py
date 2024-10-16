import pathlib
from typing import Mapping, Union

import wandb
from mbrl.util.logger import Logger
from mbrl.util.logger import LogTypes

class WANDBLogger(Logger):
    """extends the Logger class to log data to Weights & Biases.

    Args:
        log_dir, enable_back_compatible: see Logger class
        cfg: the configuration object to log to Weights & Biases
    """

    def __init__(
        self, log_dir: Union[str, pathlib.Path], enable_back_compatible: bool = False
    ):
        super().__init__(log_dir=log_dir, enable_back_compatible=enable_back_compatible)

    def log_data(self, group_name: str, data: Mapping[str, LogTypes]):
        super().log_data(group_name, data, dump=False)
        wb_data = {f"{group_name}/{key}": value for key, value in data.items()}
        wandb.log(wb_data)