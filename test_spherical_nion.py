#!/usr/bin/env python3
"""
Spherical Nion Library - Comprehensive Test
==========================================

This test verifies that the spherical nion library is working correctly.
It includes:
- Import tests
- Configuration tests  
- Basic simulation tests
- Performance verification
"""

import sys
import os
import time
import traceback
import numpy as np
import pandas as pd

def test_imports():
    """Test all essential imports."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import scipy
        print("  ✓ Scientific libraries (numpy, matplotlib, pandas, scipy)")
        
        # Test numba
        import numba
        print("  ✓ Numba for performance optimization")
        
        # Test main library
        from spherical_nion import SphericalSimulation, SimulationConfig
        print("  ✓ Main spherical_nion library")
        
        # Test engine modules
        from spherical_nion.engine import lib_electron_tracing
        from spherical_nion.engine import lib_laser_wave_FFT
        print("  ✓ Engine modules")
        
        # Test services
        from spherical_nion.services import lib_service_spherical
        print("  ✓ Services modules")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from spherical_nion import SimulationConfig
        
        # Test default config
        config = SimulationConfig.default()
        assert config.acc_voltage_nominal == 200000
        assert config.n_electrons == 10000
        print("  ✓ Default configuration")
        
        # Test fast config
        fast_config = SimulationConfig.fast_test()
        assert fast_config.n_electrons == 1000
        assert fast_config.tracing_maxiter == 500
        print("  ✓ Fast test configuration")
        
        # Test no aberration config
        no_aberr_config = SimulationConfig.no_aberration()
        assert no_aberr_config.lens2_Cs == 0
        assert no_aberr_config.lens2_Cc == 0
        print("  ✓ No aberration configuration")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_basic_simulation():
    """Test basic simulation functionality."""
    print("\n🚀 Testing basic simulation...")
    
    try:
        from spherical_nion import SphericalSimulation, SimulationConfig
        
        # Create a very fast test configuration
        config = SimulationConfig.fast_test()
        config.n_electrons = 500  # Even smaller for test
        config.tracing_maxiter = 200
        config.nbins = 50
        config.nlayers = 25
        
        print(f"  Using {config.n_electrons} electrons, {config.tracing_maxiter} max iterations")
        
        # Create simulator
        sim = SphericalSimulation(config)
        print("  ✓ Simulator created")
        
        # Test solver setup
        sim.setup_solver()
        assert sim.solver is not None
        print("  ✓ Solver setup")
        
        # Test aberrated simulation
        start_time = time.time()
        sim.run_aberrated_simulation()
        aberrated_time = time.time() - start_time
        print(f"  ✓ Aberrated simulation ({aberrated_time:.2f}s)")
        
        # Test no-aberration simulation
        start_time = time.time()
        sim.run_no_aberration_simulation()
        no_aberr_time = time.time() - start_time
        print(f"  ✓ No-aberration simulation ({no_aberr_time:.2f}s)")
        
        # Test cross-section computation
        start_time = time.time()
        sim.compute_cross_sections()
        cross_section_time = time.time() - start_time
        print(f"  ✓ Cross-section computation ({cross_section_time:.2f}s)")
        
        # Verify results
        assert sim.sample_crossection_aberrated is not None
        assert sim.sample_crossection_no_aberration is not None
        assert sim.bins is not None
        assert sim.depths is not None
        print("  ✓ Results verified")
        
        # Test results export
        results = sim.get_results()
        assert 'config' in results
        assert 'timing' in results
        assert 'detector_positions' in results
        print("  ✓ Results export")
        
        # Test DataFrame export
        df = sim.get_cross_section_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'depth' in df.columns
        assert 'radius' in df.columns
        print("  ✓ DataFrame export")
        
        total_test_time = aberrated_time + no_aberr_time + cross_section_time
        print(f"  📊 Total simulation time: {total_test_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic simulation test failed: {e}")
        traceback.print_exc()
        return False


def test_full_workflow():
    """Test the complete workflow."""
    print("\n🎯 Testing full workflow...")
    
    try:
        from spherical_nion import SphericalSimulation, SimulationConfig
        
        # Create fast configuration
        config = SimulationConfig.fast_test()
        config.n_electrons = 300  # Very small for quick test
        
        # Run full simulation
        sim = SphericalSimulation(config)
        start_time = time.time()
        results = sim.run_full_simulation()
        total_time = time.time() - start_time
        
        # Verify results
        assert results is not None
        assert 'timing' in results
        assert results['cross_section_aberrated'] is not None
        assert results['cross_section_no_aberration'] is not None
        
        print(f"  ✓ Full workflow completed in {total_time:.2f}s")
        
        # Test plotting (without showing)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        sim.plot_results(save_path="test_plot.png")
        
        # Check if plot file was created
        if os.path.exists("test_plot.png"):
            os.remove("test_plot.png")  # Clean up
            print("  ✓ Plot generation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Full workflow test failed: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """Test performance with different configurations."""
    print("\n⚡ Testing performance...")
    
    try:
        from spherical_nion import SphericalSimulation, SimulationConfig
        
        # Test different electron counts
        electron_counts = [100, 500, 1000]
        times = []
        
        for n_electrons in electron_counts:
            config = SimulationConfig.fast_test()
            config.n_electrons = n_electrons
            config.tracing_maxiter = 100  # Keep iterations low
            
            sim = SphericalSimulation(config)
            
            start_time = time.time()
            sim.run_aberrated_simulation()
            sim_time = time.time() - start_time
            times.append(sim_time)
            
            print(f"  {n_electrons} electrons: {sim_time:.2f}s")
            
        # Check that performance scales reasonably
        # (should not be worse than quadratic scaling)
        if len(times) >= 2:
            scaling_factor = times[-1] / times[0]
            electron_factor = electron_counts[-1] / electron_counts[0]
            
            if scaling_factor < electron_factor ** 2:
                print(f"  ✓ Performance scaling acceptable ({scaling_factor:.1f}x for {electron_factor}x electrons)")
            else:
                print(f"  ⚠️ Performance scaling could be better ({scaling_factor:.1f}x for {electron_factor}x electrons)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and return overall success status."""
    print("=" * 60)
    print("🧪 SPHERICAL NION LIBRARY - COMPREHENSIVE TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Simulation", test_basic_simulation),
        ("Full Workflow", test_full_workflow),
        ("Performance", test_performance),
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    print(f"Time: {total_time:.2f}s")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Library is working correctly.")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check the output above.")
        return False


if __name__ == "__main__":
    # Set up path for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
