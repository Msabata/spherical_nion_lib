"""
Configuration management for spherical nion simulations.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration class for spherical aberration correction simulations."""
    
    # Electron beam parameters
    acc_voltage_nominal: float = 200000  # [V]
    acc_voltage_FWHM: float = 0.6
    beam_r: float = 5e-6
    n_electrons: int = 10000
    electron_initialcomposition: str = "tophat"
    
    # Lens parameters
    lens1_f: float = 37.5e-3
    lens2_f: float = 1.5e-3
    lens1_Cs: float = 0
    lens2_Cs: float = 1e-3
    lens1_Cc: float = 0
    lens2_Cc: float = 1.1e-3
    demagnification: float = 6
    
    # Laser parameters
    laser_wavelength: float = 1035e-9
    laser_pulse_timelength: float = 200e-15
    laser_direction: int = -1
    laser_thick_pulse_energy: float = 4.05e-6  # [J]
    
    # Simulation parameters
    dt: float = 1e-14
    tracing_maxiter: int = 3000
    defocus: float = 0.148e-6  # For aberrated simulation
    defocus_ideal: float = 0.381e-6  # For ideal correction
    
    # Analysis parameters
    nbins: int = 402
    nlayers: int = 201
    depth_range: tuple = (-1e-7, 1.5e-7)
    
    # Laser setup parameters
    plane0_width: float = 50.0e-3
    plane_points: int = 1002
    laser_beam_width: float = 2.5e-3
    SLM_npix: list = None
    SLM_size: list = None
    detectplanesz: np.ndarray = None
    laser_zpos: float = 0
    laser_xcenter: float = 0
    SLM_zpos: float = 0.0001e-3
    SLM_xcenter: float = 0
    SLM_dist: float = 0.2
    lens1_f_laser: float = 200e-3
    lens2_f_laser: float = 370e-3
    lens3_f_laser: float = 40e-3
    beamblock_r: float = 100e-6
    lens3_zdist: float = 150e-3
    lens1_r: float = 12.5e-3
    lens2_r: float = 12.5e-3
    lens3_r: float = 12.5e-3
    rescaling_f: float = 0.01
    
    # SLM parameters
    SLM_doughnut_turns: int = 1
    SLM_lens_f_inverse: float = -0.000001
    SLM_sphericalaberrationcoef: float = 10e10
    SLM_lens_f_inverse_2: float = 0.35
    
    def __post_init__(self):
        """Initialize default values that depend on other parameters."""
        if self.SLM_npix is None:
            self.SLM_npix = [1200, 1200]
        if self.SLM_size is None:
            self.SLM_size = [9.60e-3, 9.60e-3]
        if self.detectplanesz is None:
            self.detectplanesz = np.linspace(-0.1, 0.1, 5) * 1e-3
    
    @classmethod
    def default(cls) -> 'SimulationConfig':
        """Create a default configuration for standard UHR STEM."""
        return cls()
    
    @classmethod
    def fast_test(cls) -> 'SimulationConfig':
        """Create a fast configuration for testing (fewer electrons, iterations)."""
        config = cls()
        config.n_electrons = 1000
        config.tracing_maxiter = 500
        config.nbins = 100
        config.nlayers = 50
        return config
    
    @classmethod
    def no_aberration(cls) -> 'SimulationConfig':
        """Create configuration with no spherical aberration."""
        config = cls()
        config.lens2_Cs = 0
        config.lens2_Cc = 0
        return config
