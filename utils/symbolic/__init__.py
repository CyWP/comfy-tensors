"""Symbolic expression parsing utilities.

Exposes:
- Parser: Base recursive-descent expression parser.
- TorchParser: PyTorch-specific parser with tensor functions.
"""

from .parser import Parser
from .torch_parser import TorchParser
