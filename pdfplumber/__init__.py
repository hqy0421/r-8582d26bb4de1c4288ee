__all__ = [
    "__version__",
    "utils",
    "regions",
    "pdfminer",
    "open",
    "repair",
    "set_debug",
]

import pdfminer
import pdfminer.pdftypes

from . import regions, utils
from ._version import __version__
from .pdf import PDF
from .repair import repair

open = PDF.open
