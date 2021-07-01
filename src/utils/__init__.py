import os
from pathlib import Path

from src.utils.types import StrOrPath


def relative_to_file(file: StrOrPath, relative: str) -> Path:
    dir_name = os.path.dirname(file)

    return Path(os.path.join(dir_name, relative))
