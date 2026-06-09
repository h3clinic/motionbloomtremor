"""
hand_landscape – 3D Hand State Landscape for Residual Tremor Extraction.

Public API
----------
    from hand_landscape import (
        HandState,
        HandStateLandscape,
        Observation,
        TrackingQualityMonitor,
        TremorAnalyzer,
        DotFieldAnalyzer,
        MicroOscillator,
        compute_residuals,
        run_tremor_analysis,
        assign_regions,
        build_landscape_diagnostics,
        build_waiting_diagnostics,
    )
"""

from .state import HandState
from .observation import Observation
from .landscape import HandStateLandscape
from .tracking import TrackingQualityMonitor
from .residuals import compute_residuals
from .tremor import run_tremor_analysis, TremorAnalyzer
from .dots import DotFieldAnalyzer, MicroOscillator
from .regions import assign_regions, REGIONS
from .diagnostics import build_landscape_diagnostics, build_waiting_diagnostics

__all__ = [
    "HandState",
    "HandStateLandscape",
    "Observation",
    "TrackingQualityMonitor",
    "TremorAnalyzer",
    "DotFieldAnalyzer",
    "MicroOscillator",
    "compute_residuals",
    "run_tremor_analysis",
    "assign_regions",
    "REGIONS",
    "build_landscape_diagnostics",
    "build_waiting_diagnostics",
]
