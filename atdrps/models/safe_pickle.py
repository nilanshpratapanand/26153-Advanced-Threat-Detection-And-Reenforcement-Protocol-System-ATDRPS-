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

__all__ = ["UnsafePickleError", "safe_load", "HAZARD_ALLOWED"]

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


# Recorded from the globals a real pickle of a fitted HazardModel (gradient-boosted trees +
# isotonic calibrator) asks the unpickler for -- not guessed.  Opt-in per call so that the
# baseline/world-model loaders keep their tighter list.
HAZARD_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.random._pickle", "__generator_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._pcg64", "PCG64"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
    ("numpy.random.bit_generator", "SeedSequence"),
    ("sklearn.ensemble._hist_gradient_boosting.gradient_boosting", "HistGradientBoostingClassifier"),
    ("sklearn.ensemble._hist_gradient_boosting.binning", "_BinMapper"),
    ("sklearn.ensemble._hist_gradient_boosting.predictor", "TreePredictor"),
    ("sklearn.preprocessing._label", "LabelEncoder"),
    ("sklearn._loss.loss", "HalfBinomialLoss"), ("sklearn._loss._loss", "CyHalfBinomialLoss"),
    ("sklearn._loss.link", "LogitLink"), ("sklearn._loss.link", "Interval"),
    ("sklearn.isotonic", "IsotonicRegression"),
})


class UnsafePickleError(pickle.UnpicklingError):
    """The file references something a model file has no business containing."""


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, fh, extra=frozenset()):
        super().__init__(fh)
        self._allowed = _ALLOWED | frozenset(extra)

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._allowed:
            raise UnsafePickleError(
                f"refusing to load {module}.{name}: it is not part of a saved ATDRPS "
                f"model. The file may be corrupt or tampered with.")
        return super().find_class(module, name)


def safe_load(fh: BinaryIO, extra=frozenset()) -> Any:
    """Load a pickle from ``fh``, resolving only allowlisted globals (plus ``extra``)."""
    return _RestrictedUnpickler(fh, extra).load()
