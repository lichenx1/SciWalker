import os
from importlib.resources import files
from pathlib import Path


def data_root() -> Path:
    """Resolve artifacts relative to the working directory, or an explicit override."""
    return Path(os.environ.get("SCIDATA_DATA_DIR", "outputs")).expanduser().resolve()


def catalog_root() -> Path:
    return data_root() / "catalog"


def resource_path(name: str) -> Path:
    return Path(str(files("scidata.resources").joinpath(name)))
