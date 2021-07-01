import math
from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from src.utils import relative_to_file


def create_tb_and_csv_logger(run_name: str, log_path: Optional[Path] = None):
    if log_path is None:
        log_path = relative_to_file(__file__, "../../lightning_logs").resolve()

    tensorboard_logger = TensorBoardLogger(
        str(log_path),
        name=run_name,
    )

    version_name = f"/version_{tensorboard_logger.version}"

    csv_logger = CSVLogger(str(log_path), name=run_name + version_name, version="")

    return [tensorboard_logger, csv_logger]


def create_return_based_model_checkpoint(
    log_path,
    period=1,
    save_top_k: int = 1,
    save_last: bool = True,
) -> ModelCheckpoint:
    dirpath = Path(log_path) / "checkpoints"

    return ModelCheckpoint(
        dirpath=str(dirpath),
        save_top_k=save_top_k,
        mode="max",
        filename="{epoch}-{return:.2f}",
        monitor="return",
        period=period,
        save_last=save_last,
    )


def max_epochs(max_steps: int, batches_per_epoch: int) -> int:
    return math.ceil(max_steps / batches_per_epoch)
