"""Restricted unpickling for model files.

``pickle.load`` on a file is ``eval`` on that file: a ``model.pkl`` that someone
swapped can run arbitrary code the moment it is opened.  ATDRPS ships into
defence networks where model files are copied between machines, so the loader
only reconstructs the exact classes a saved model contains and refuses every
other global -- ``os.system``, ``builtins.eval`` and friends never resolve.

The allowlist was taken from the globals actually present in saved models
(``pickletools`` scan of a trained ``NumpyDynamicsWorldModel`` and both
``LogisticBaseline`` modes), not guessed.  If a future scikit-learn adds a
class to a saved model, loading fails loudly with the name to review, which is
the correct failure for a security control.
"""

from __future__ import annotations

import pickle
from typing import Any, BinaryIO

__all__ = ["UnsafePickleError", "safe_load"]

_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    # numpy arrays and scalars (numpy 1.x spells the module ``numpy.core``)
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    # the estimators the world model and baselines are built from
    ("sklearn.linear_model._ridge", "Ridge"),
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    # ATDRPS's own containers
    ("atdrps.models.baseline", "LogisticBaseline"),
    ("atdrps.models.base", "Standardiser"),
})


class UnsafePickleError(pickle.UnpicklingError):
    """The file references something a model file has no business containing."""


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED:
            raise UnsafePickleError(
                f"refusing to load {module}.{name}: it is not part of a saved ATDRPS "
                f"model. The file may be corrupt or tampered with.")
        return super().find_class(module, name)


def safe_load(fh: BinaryIO) -> Any:
    """Load a pickle from ``fh``, resolving only allowlisted globals."""
    return _RestrictedUnpickler(fh).load()
