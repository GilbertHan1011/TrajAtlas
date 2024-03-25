"""Top-level package for TrajAtlas."""

__author__ = "GilbertHan"
__email__ = "GilbertHan1011@gmail.com"
__version__ = "0.1.0"

import warnings

from TrajAtlas import TrajDiff
from . import TrajDiff as diff
from . import model, utils,TRAVMap

__all__ = ["TrajDiff", "model", "utils", "diff"]
