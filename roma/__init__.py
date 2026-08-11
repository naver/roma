# RoMa
# Copyright (c) 2020 NAVER Corp.
# 3-Clause BSD License.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("roma")
except PackageNotFoundError:
    # Package is not installed (e.g. used as a git submodule).
    __version__ = "unknown"

from .mappings import *
from .utils import *
from .transforms import *
from .euler import *
