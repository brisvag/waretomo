"""Batch processing for tomography data with Warp and aretomo."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waretomo")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lorenzo Gaifas"
__email__ = "brisvag@gmail.com"
