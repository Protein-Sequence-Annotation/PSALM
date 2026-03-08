"""PSALM package."""

from importlib.metadata import PackageNotFoundError, version

from psalm.psalm_model import PSALM

try:
    __version__ = version("protein-sequence-annotation")
except PackageNotFoundError:
    __version__ = "0+local"

__all__ = ["PSALM", "__version__"]
