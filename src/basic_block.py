"""
Deprecated compatibility shim.

`basic_block.py` previously contained model + training code. The model
definitions have been moved to `src/model.py` and the training script is
now in `src/train.py`. This module re-exports the model classes for
backwards compatibility with older imports.
"""

from .model import BasicBlock, ResNet18, build_model

__all__ = ['BasicBlock', 'ResNet18', 'build_model']