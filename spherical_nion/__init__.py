"""
Spherical Nion Library

A high-performance library for spherical aberration correction 
simulation in electron microscopy using laser techniques.

This library provides:
- Electron beam tracing and simulation  
- Laser wave propagation modeling
- Spherical aberration correction analysis
- Fast implementation with optimized performance
"""

__version__ = "1.0.0" 
__author__ = "Spherical Nion Team"

# Main imports for easy access
from .config import SimulationConfig
from .core import SphericalSimulation

# Main exports
__all__ = ['SimulationConfig', 'SphericalSimulation']
