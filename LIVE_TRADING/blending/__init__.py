"""
Blending Module
===============

Ridge risk-parity blending with temperature compression.
"""

from .ridge_weights import RidgeWeightCalculator, calculate_ridge_weights
from .temperature import TemperatureCompressor
from .horizon_blender import HorizonBlender, BlendedAlpha

__all__ = [
    "RidgeWeightCalculator",
    "calculate_ridge_weights",
    "TemperatureCompressor",
    "HorizonBlender",
    "BlendedAlpha",
]
