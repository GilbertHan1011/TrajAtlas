"""Top-level package for TrajAtlas."""


import warnings
from importlib import metadata
from TrajAtlas import TrajDiff,logging
from . import TrajDiff as diff
from . import model, utils,TRAVMap
from TrajAtlas.settings import settings

try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None

__all__ = ["TrajDiff", "model", "utils", "diff","settings","logging"]
