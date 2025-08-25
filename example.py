#!/usr/bin/env python3
"""
Spherical Nion Library - Quick Example
======================================

This example demonstrates basic usage of the spherical_nion library.
"""

from spherical_nion import SphericalSimulation, SimulationConfig

def main():
    print("🚀 Spherical Nion Library - Quick Example")
    print("=" * 50)
    
    # Create a fast configuration for quick demo
    print("⚙️  Creating configuration...")
    config = SimulationConfig.fast_test()
    config.n_electrons = 1000  # Small number for quick demo
    
    print(f"   - Electrons: {config.n_electrons:,}")
    print(f"   - Voltage: {config.acc_voltage_nominal:,} V")
    print(f"   - Max iterations: {config.tracing_maxiter:,}")
    
    # Create and run simulation
    print("\n🔬 Running simulation...")
    sim = SphericalSimulation(config)
    
    # Run full simulation (aberrated + no aberration + analysis)
    results = sim.run_full_simulation()
    
    # Display results
    print("\n📊 Results:")
    timing = results['timing']
    print(f"   - Aberrated simulation: {timing.get('aberrated', 0):.2f}s")
    print(f"   - No-aberration simulation: {timing.get('no_aberration', 0):.2f}s") 
    print(f"   - Total time: {timing.get('total', 0):.2f}s")
    
    # Get detector positions
    positions = results['detector_positions']
    if 'aberrated' in positions:
        aberrated_spread = positions['aberrated'].std()
        print(f"   - Aberrated beam spread: {aberrated_spread*1e9:.2f} nm")
    
    if 'no_aberration' in positions:
        ideal_spread = positions['no_aberration'].std()
        print(f"   - Ideal beam spread: {ideal_spread*1e9:.2f} nm")
        
        if 'aberrated' in positions:
            improvement = aberrated_spread / ideal_spread
            print(f"   - Aberration effect: {improvement:.1f}x beam spread increase")
    
    # Save plot
    print("\n📈 Saving results plot...")
    sim.plot_results(save_path="spherical_nion_example.png")
    print("   Plot saved as: spherical_nion_example.png")
    
    # Export data
    print("\n💾 Exporting data...")
    df = sim.get_cross_section_data()
    df.to_csv("spherical_nion_data.csv", index=False)
    print(f"   Data exported: spherical_nion_data.csv ({len(df)} rows)")
    
    print("\n✅ Example completed successfully!")
    print("\nℹ️  To customize the simulation:")
    print("   - Modify SimulationConfig parameters")
    print("   - Use config.n_electrons for different electron counts")
    print("   - Use config.tracing_maxiter for simulation accuracy")
    print("   - See README.md for more details")

if __name__ == "__main__":
    main()
