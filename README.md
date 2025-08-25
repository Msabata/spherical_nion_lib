# Spherical Nion Library

A high-performance library for spherical aberration correction simulation in electron microscopy using laser techniques.

## Features

- **High Performance**: Optimized with Numba JIT compilation
- **Easy to Use**: Simple, clean API
- **Scientific Computing**: Built on NumPy, SciPy, and Matplotlib
- **Electron Beam Simulation**: Advanced electron tracing capabilities
- **Laser Wave Modeling**: FFT-based laser propagation
- **Spherical Aberration Correction**: Advanced correction algorithms

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from spherical_nion import SphericalSimulation, SimulationConfig

# Create configuration
config = SimulationConfig.default()

# Run simulation
sim = SphericalSimulation(config)
sim.run_full_simulation()

# Get results
results = sim.get_results()
print(f"Simulation completed with {len(results['detector_positions'])} electrons")
```

### Example Script

```python
from spherical_nion import SphericalSimulation, SimulationConfig

# Quick simulation
config = SimulationConfig.fast_test()  # Small configuration for testing
sim = SphericalSimulation(config)
sim.run_aberrated_simulation()

# Access results
cross_section = sim.get_cross_section_data()
sim.plot_results()
```

## Testing

Run the included test to verify installation:

```bash
python test_spherical_nion.py
```

## Requirements

- Python 3.8+
- NumPy 1.21+
- SciPy 1.7+
- Matplotlib 3.5+
- Pandas 1.3+
- Numba 0.56+

## License

MIT License
