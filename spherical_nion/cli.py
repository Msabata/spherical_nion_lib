"""
Simple command-line interface for spherical_nion library.
"""

import argparse
import sys
from pathlib import Path

from .core import SphericalSimulation
from .config import SimulationConfig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spherical Nion - Spherical aberration correction simulation"
    )
    
    parser.add_argument(
        "--electrons", "-n",
        type=int, 
        default=10000,
        help="Number of electrons to simulate (default: 10000)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast test configuration (fewer electrons and iterations)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="spherical_results.png",
        help="Output plot filename (default: spherical_results.png)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        if args.fast:
            config = SimulationConfig.fast_test()
            print("🏃 Using fast test configuration")
        else:
            config = SimulationConfig.default()
            
        if args.electrons != 10000:
            config.n_electrons = args.electrons
            
        if args.verbose:
            print(f"Configuration:")
            print(f"  - Electrons: {config.n_electrons:,}")
            print(f"  - Max iterations: {config.tracing_maxiter:,}")
            print(f"  - Voltage: {config.acc_voltage_nominal:,} V")
            
        # Run simulation
        print(f"🚀 Starting simulation with {config.n_electrons:,} electrons...")
        sim = SphericalSimulation(config)
        results = sim.run_full_simulation()
        
        # Plot results
        print(f"📊 Saving plot to {args.output}")
        sim.plot_results(save_path=args.output)
        
        # Print summary
        timing = results['timing']
        print(f"\n✅ Simulation completed successfully!")
        print(f"📈 Total time: {timing.get('total', 0):.2f}s")
        print(f"📊 Plot saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
