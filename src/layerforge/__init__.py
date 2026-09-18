"""LayerForge v0: still image / Imagine parts → completed layer pack."""

import os

# Conda+PyTorch on Windows ships two OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from layerforge.contracts import SCHEMA

__all__ = ["SCHEMA"]
