"""Modules to analyze and compare functions (e.g. DOS or spectra)."""

from aim2dat.fct.discretization import DiscretizedAxis, DiscretizedGrid
from aim2dat.fct.function_comparison import FunctionAnalysis
from aim2dat.fct.fingerprint import FunctionDiscretizationFingerprint

__all__ = [
    "DiscretizedAxis",
    "DiscretizedGrid",
    "FunctionAnalysis",
    "FunctionDiscretizationFingerprint",
]
