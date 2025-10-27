"""This module contains the blocks implemented in Linax."""

from linax.blocks.base import Block, BlockConfig
from linax.blocks.linoss import LinOSSBlock, LinOSSBlockConfig
from linax.blocks.lru import LRUBlock, LRUBlockConfig
from linax.blocks.s5 import S5Block, S5BlockConfig

__all__ = [
    "BlockConfig",
    "Block",
    "LinOSSBlockConfig",
    "LinOSSBlock",
    "LRUBlockConfig",
    "LRUBlock",
    "S5BlockConfig",
    "S5Block",
]
