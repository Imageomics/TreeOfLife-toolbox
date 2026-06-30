"""
Tooling for converting HDF5-backed Tree of Life data into WebDataset shards.

Importing this module registers filter/scheduler/runner implementations with
the toolbox registry so they can be scheduled through the generic CLI.
"""

from TreeOfLife_toolbox.tol_hybrid_to_wds import classes  # noqa: F401
