"""
Main Spherical Nion Simulation Module
=====================================

High-performance spherical aberration correction simulation for electron microscopy.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import time
import sys
import os

# Import the engine modules
try:
    from .engine import lib_electron_tracing as lib
    from .engine import lib_laser_wave_FFT  
    from .services import lib_service_spherical
except ImportError:
    # Fallback for direct execution
    try:
        sys.path.append(os.getcwd())
        from engine import lib_electron_tracing as lib
        from engine import lib_laser_wave_FFT  
        from services import lib_service_spherical
    except ImportError as e:
        print(f"Warning: Could not import engine modules: {e}")
        raise

from .config import SimulationConfig


class SphericalSimulation:
    """
    High-performance spherical aberration simulation.
    
    This class provides the main interface for running spherical aberration
    correction simulations in electron microscopy using laser techniques.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation with configuration.
        
        Args:
            config: SimulationConfig object. If None, uses default configuration.
        """
        self.config = config or SimulationConfig.default()
        self.solver = None
        self.solver_no_aberration = None
        
        # Results storage
        self.sample_crossection_aberrated = None
        self.sample_crossection_corrected = None  
        self.sample_crossection_no_aberration = None
        self.bins = None
        self.depths = None
        self._timing_info = {}
        
    def setup_solver(self) -> None:
        """Create the spherical aberration solver with current configuration."""
        self.solver = lib_service_spherical.Spherical_aberation_solver(
            acc_voltage_nominal=self.config.acc_voltage_nominal,
            acc_voltage_FWHM=self.config.acc_voltage_FWHM,
            beam_r=self.config.beam_r,
            lens1_f=self.config.lens1_f,
            lens2_f=self.config.lens2_f,
            lens1_Cs=self.config.lens1_Cs,
            lens2_Cs=self.config.lens2_Cs,
            lens1_Cc=self.config.lens1_Cc,
            lens2_Cc=self.config.lens2_Cc,
            demagnification=self.config.demagnification,
            laser_wavelength=self.config.laser_wavelength,
            n_electrons=self.config.n_electrons,
            laser_pulse_timelength=self.config.laser_pulse_timelength,
            dt=self.config.dt,
            tracing_maxiter=self.config.tracing_maxiter,
            laser_direction=self.config.laser_direction,
            electron_initialcomposition=self.config.electron_initialcomposition
        )
        
    def run_aberrated_simulation(self) -> None:
        """Run simulation with spherical aberration."""
        start_time = time.time()
        
        if self.solver is None:
            self.setup_solver()
            
        self.solver.defocus = self.config.defocus
        self.solver.build_elements()
        self.solver.trace_aberrated()
        
        self._timing_info['aberrated'] = time.time() - start_time
        print(f"✓ Aberrated simulation completed in {self._timing_info['aberrated']:.2f}s")
        
    def run_corrected_simulation(self) -> None:
        """Run simulation with laser correction."""
        start_time = time.time()
        
        if self.solver is None:
            self.setup_solver()
            
        self.solver.defocus = self.config.defocus_ideal
        self.solver.build_elements()
        self.solver.trace_corrected()
        
        self._timing_info['corrected'] = time.time() - start_time
        print(f"✓ Corrected simulation completed in {self._timing_info['corrected']:.2f}s")
        
    def run_no_aberration_simulation(self) -> None:
        """Run simulation with no aberration (ideal case)."""
        start_time = time.time()
        
        # Create a no-aberration solver
        no_aberr_config = SimulationConfig.no_aberration()
        no_aberr_config.n_electrons = self.config.n_electrons
        no_aberr_config.tracing_maxiter = self.config.tracing_maxiter
        
        self.solver_no_aberration = lib_service_spherical.Spherical_aberation_solver(
            acc_voltage_nominal=no_aberr_config.acc_voltage_nominal,
            acc_voltage_FWHM=no_aberr_config.acc_voltage_FWHM,
            beam_r=no_aberr_config.beam_r,
            lens1_f=no_aberr_config.lens1_f,
            lens2_f=no_aberr_config.lens2_f,
            lens1_Cs=no_aberr_config.lens1_Cs,
            lens2_Cs=no_aberr_config.lens2_Cs,  # 0 for no aberration
            lens1_Cc=no_aberr_config.lens1_Cc,
            lens2_Cc=no_aberr_config.lens2_Cc,  # 0 for no aberration
            demagnification=no_aberr_config.demagnification,
            laser_wavelength=no_aberr_config.laser_wavelength,
            n_electrons=no_aberr_config.n_electrons,
            laser_pulse_timelength=no_aberr_config.laser_pulse_timelength,
            dt=no_aberr_config.dt,
            tracing_maxiter=no_aberr_config.tracing_maxiter,
            laser_direction=no_aberr_config.laser_direction,
            electron_initialcomposition=no_aberr_config.electron_initialcomposition
        )
        
        self.solver_no_aberration.defocus = no_aberr_config.defocus
        self.solver_no_aberration.build_elements()
        self.solver_no_aberration.trace_aberrated()  # Same method, just no aberration coefficients
        
        self._timing_info['no_aberration'] = time.time() - start_time
        print(f"✓ No-aberration simulation completed in {self._timing_info['no_aberration']:.2f}s")
        
    def compute_cross_sections(self) -> None:
        """Compute sample cross-sections for all simulation results."""
        print("Computing cross-sections...")
        
        # Set up depth and bin arrays
        self.depths = np.linspace(self.config.depth_range[0], self.config.depth_range[1], self.config.nlayers)
        
        # Compute aberrated cross-section
        if self.solver is not None:
            self._compute_single_cross_section('aberrated', self.solver)
            
        # Compute no-aberration cross-section  
        if self.solver_no_aberration is not None:
            self._compute_single_cross_section('no_aberration', self.solver_no_aberration)
            
        print("✓ Cross-sections computed")
        
    def _compute_single_cross_section(self, name: str, solver) -> None:
        """Compute cross-section for a single solver."""
        cross_section = np.full((self.config.nlayers, self.config.nbins), np.nan)
        
        # Get initial detector positions to set up bins
        detector = solver.e_beam.detector_spacelike(0, nans=False)
        detector_r = np.linalg.norm(detector, axis=0)
        histogram, bins = np.histogram(detector_r, self.config.nbins, range=(0, 10e-9), density=True)
        
        if self.bins is None:
            self.bins = bins
            
        # Compute cross-section at each depth
        for c, depth in enumerate(self.depths):
            detector = solver.e_beam.detector_spacelike(depth, nans=False)
            detector_r = np.linalg.norm(detector, axis=0)
            cross_section[c], _ = np.histogram(detector_r, self.bins, density=True)
            
        # Store result
        if name == 'aberrated':
            self.sample_crossection_aberrated = cross_section
        elif name == 'no_aberration':
            self.sample_crossection_no_aberration = cross_section
        elif name == 'corrected':
            self.sample_crossection_corrected = cross_section
            
    def run_full_simulation(self) -> Dict[str, Any]:
        """
        Run complete simulation workflow: aberrated, corrected, and no-aberration cases.
        
        Returns:
            Dictionary containing all results and timing information.
        """
        print("🚀 Starting full spherical aberration simulation...")
        total_start = time.time()
        
        # Run all simulations
        self.run_aberrated_simulation()
        self.run_no_aberration_simulation()
        
        # Compute cross-sections
        self.compute_cross_sections()
        
        total_time = time.time() - total_start
        self._timing_info['total'] = total_time
        
        print(f"✅ Full simulation completed in {total_time:.2f}s")
        
        return self.get_results()
        
    def get_results(self) -> Dict[str, Any]:
        """Get all simulation results."""
        return {
            'config': self.config,
            'cross_section_aberrated': self.sample_crossection_aberrated,
            'cross_section_no_aberration': self.sample_crossection_no_aberration,
            'cross_section_corrected': self.sample_crossection_corrected,
            'bins': self.bins,
            'depths': self.depths,
            'timing': self._timing_info,
            'detector_positions': self._get_detector_positions(),
        }
        
    def _get_detector_positions(self) -> Dict[str, np.ndarray]:
        """Get final detector positions from all simulations."""
        positions = {}
        
        if self.solver is not None:
            detector = self.solver.e_beam.detector_spacelike(0, nans=False)
            positions['aberrated'] = detector
            
        if self.solver_no_aberration is not None:
            detector = self.solver_no_aberration.e_beam.detector_spacelike(0, nans=False)
            positions['no_aberration'] = detector
            
        return positions
        
    def get_cross_section_data(self) -> pd.DataFrame:
        """Get cross-section data as a pandas DataFrame."""
        if self.sample_crossection_aberrated is None:
            raise ValueError("No cross-section data available. Run compute_cross_sections() first.")
            
        data = []
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        for i, depth in enumerate(self.depths):
            for j, radius in enumerate(bin_centers):
                row = {
                    'depth': depth,
                    'radius': radius,
                    'aberrated': self.sample_crossection_aberrated[i, j] if self.sample_crossection_aberrated is not None else np.nan,
                    'no_aberration': self.sample_crossection_no_aberration[i, j] if self.sample_crossection_no_aberration is not None else np.nan,
                }
                if self.sample_crossection_corrected is not None:
                    row['corrected'] = self.sample_crossection_corrected[i, j]
                data.append(row)
                
        return pd.DataFrame(data)
        
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot simulation results."""
        if self.sample_crossection_aberrated is None:
            raise ValueError("No results to plot. Run simulation first.")
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Spherical Aberration Simulation Results\n{self.config.n_electrons:,} electrons', fontsize=14)
        
        # Plot cross-sections
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2 * 1e9  # Convert to nm
        depths_nm = self.depths * 1e9  # Convert to nm
        
        # Aberrated case
        if self.sample_crossection_aberrated is not None:
            im1 = axes[0, 0].imshow(self.sample_crossection_aberrated, 
                                   extent=[bin_centers[0], bin_centers[-1], depths_nm[-1], depths_nm[0]],
                                   aspect='auto', cmap='viridis')
            axes[0, 0].set_title('Aberrated')
            axes[0, 0].set_xlabel('Radius (nm)')
            axes[0, 0].set_ylabel('Depth (nm)')
            plt.colorbar(im1, ax=axes[0, 0])
            
        # No aberration case
        if self.sample_crossection_no_aberration is not None:
            im2 = axes[0, 1].imshow(self.sample_crossection_no_aberration,
                                   extent=[bin_centers[0], bin_centers[-1], depths_nm[-1], depths_nm[0]], 
                                   aspect='auto', cmap='viridis')
            axes[0, 1].set_title('No Aberration')
            axes[0, 1].set_xlabel('Radius (nm)')
            axes[0, 1].set_ylabel('Depth (nm)')
            plt.colorbar(im2, ax=axes[0, 1])
            
        # Detector positions comparison
        positions = self._get_detector_positions()
        if 'aberrated' in positions:
            axes[1, 0].scatter(positions['aberrated'][0] * 1e9, positions['aberrated'][1] * 1e9, 
                             alpha=0.6, s=1, label='Aberrated')
        if 'no_aberration' in positions:
            axes[1, 0].scatter(positions['no_aberration'][0] * 1e9, positions['no_aberration'][1] * 1e9,
                             alpha=0.6, s=1, label='No Aberration')
        axes[1, 0].set_xlabel('X (nm)')
        axes[1, 0].set_ylabel('Y (nm)')
        axes[1, 0].set_title('Detector Positions')
        axes[1, 0].legend()
        
        # Timing information
        if self._timing_info:
            timing_names = list(self._timing_info.keys())
            timing_values = list(self._timing_info.values())
            axes[1, 1].bar(timing_names, timing_values)
            axes[1, 1].set_title('Computation Times')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
