#!/usr/bin/env python3
"""
Comprehensive test script for xwrf_diagnostics surface variables.
Tests all new surface variable functions with WRF output files.
"""

import xarray as xr
import numpy as np
import sys
import os

# Add current directory to path for import
sys.path.insert(0, os.path.dirname(__file__))
import xwrf_diagnostics

def test_surface_variables():
    """Test all surface variable functions."""
    
    print("="*70)
    print("XWRF DIAGNOSTICS - SURFACE VARIABLES TEST")
    print("="*70)
    
    # List of wrfout files to test
    wrf_files = [
        "./wrfout_d02_2022-07-14_0600.nc",
        "./wrfout_d02_2022-07-14_0700.nc",
        "./wrfout_d02_2022-07-14_0800.nc",
        "./wrfout_d02_2022-07-14_0900.nc",
    ]
    
    test_results = []
    
    for wrf_file in wrf_files:
        if not os.path.exists(wrf_file):
            print(f"\n⚠️  File not found: {wrf_file}")
            continue
            
        print(f"\n{'='*70}")
        print(f"Testing: {os.path.basename(wrf_file)}")
        print(f"{'='*70}")
        
        try:
            ds = xr.open_dataset(wrf_file)
            print(f"✓ Dataset loaded successfully")
            print(f"  Variables: {len(ds.data_vars)}")
            print(f"  Dimensions: {list(ds.dims.keys())}")
            print(f"  Shape: {dict(ds.dims)}")
            
            # Test T2
            print(f"\n--- T2: 2m Temperature ---")
            try:
                t2 = ds.wrf_diag.t2()
                print(f"✓ T2 extracted")
                print(f"  Shape: {t2.shape}")
                print(f"  Units: {t2.attrs.get('units')}")
                print(f"  Range: {t2.min().values:.2f} to {t2.max().values:.2f} {t2.attrs.get('units')}")
                
                # Test T2C
                t2c = ds.wrf_diag.t2c()
                print(f"✓ T2C converted")
                print(f"  Range: {t2c.min().values:.2f} to {t2c.max().values:.2f} °C")
                
                # Verify conversion
                diff = np.abs((t2 - 273.15).max() - t2c.max())
                if diff < 0.01:
                    print(f"✓ Unit conversion correct")
                else:
                    print(f"✗ Unit conversion error: {diff:.4f}")
                    
                test_results.append(('T2', os.path.basename(wrf_file), 'PASS'))
            except Exception as e:
                print(f"✗ T2 failed: {e}")
                test_results.append(('T2', os.path.basename(wrf_file), 'FAIL'))
            
            # Test RH2
            print(f"\n--- RH2: 2m Relative Humidity ---")
            try:
                rh2 = ds.wrf_diag.rh2()
                print(f"✓ RH2 computed")
                print(f"  Shape: {rh2.shape}")
                print(f"  Units: {rh2.attrs.get('units')}")
                print(f"  Range: {rh2.min().values:.2f}% to {rh2.max().values:.2f}%")
                print(f"  Mean: {rh2.mean().values:.2f}%")
                
                # Validate RH range (0-100%)
                if rh2.min() >= 0 and rh2.max() <= 100:
                    print(f"✓ RH range valid")
                else:
                    print(f"⚠️  RH out of valid range")
                    
                test_results.append(('RH2', os.path.basename(wrf_file), 'PASS'))
            except Exception as e:
                print(f"✗ RH2 failed: {e}")
                test_results.append(('RH2', os.path.basename(wrf_file), 'FAIL'))
            
            # Test TD2
            print(f"\n--- TD2: 2m Dewpoint Temperature ---")
            try:
                td2 = ds.wrf_diag.td2()
                print(f"✓ TD2 computed")
                print(f"  Shape: {td2.shape}")
                print(f"  Units: {td2.attrs.get('units')}")
                print(f"  Range: {td2.min().values:.2f} to {td2.max().values:.2f} °C")
                
                test_results.append(('TD2', os.path.basename(wrf_file), 'PASS'))
            except Exception as e:
                print(f"✗ TD2 failed: {e}")
                test_results.append(('TD2', os.path.basename(wrf_file), 'FAIL'))
            
            # Test Q2
            print(f"\n--- Q2: 2m Specific Humidity ---")
            try:
                q2 = ds.wrf_diag.q2()
                print(f"✓ Q2 extracted")
                print(f"  Shape: {q2.shape}")
                print(f"  Units: {q2.attrs.get('units')}")
                print(f"  Range: {q2.min().values:.4f} to {q2.max().values:.4f}")
                
                test_results.append(('Q2', os.path.basename(wrf_file), 'PASS'))
            except Exception as e:
                print(f"✗ Q2 failed: {e}")
                test_results.append(('Q2', os.path.basename(wrf_file), 'FAIL'))
            
            # Test PSFC
            print(f"\n--- PSFC: Surface Pressure ---")
            try:
                psfc = ds.wrf_diag.psfc()
                print(f"✓ PSFC extracted")
                print(f"  Shape: {psfc.shape}")
                print(f"  Units: {psfc.attrs.get('units')}")
                print(f"  Range: {psfc.min().values:.2f} to {psfc.max().values:.2f} {psfc.attrs.get('units')}")
                
                # Test unit conversion
                psfc_hpa = ds.wrf_diag.psfc(units="hPa")
                print(f"✓ PSFC unit conversion works")
                print(f"  hPa range: {psfc_hpa.min().values:.2f} to {psfc_hpa.max().values:.2f} hPa")
                
                test_results.append(('PSFC', os.path.basename(wrf_file), 'PASS'))
            except Exception as e:
                print(f"✗ PSFC failed: {e}")
                test_results.append(('PSFC', os.path.basename(wrf_file), 'FAIL'))
            
            # Test 10m Winds
            print(f"\n--- 10m Winds ---")
            try:
                u10 = ds.wrf_diag.u10()
                v10 = ds.wrf_diag.v10()
                print(f"✓ U10 extracted: {u10.min().values:.2f} to {u10.max().values:.2f} m/s")
                print(f"✓ V10 extracted: {v10.min().values:.2f} to {v10.max().values:.2f} m/s")
                
                wspd10 = ds.wrf_diag.wspd10()
                print(f"✓ WSPD10 computed: {wspd10.min().values:.2f} to {wspd10.max().values:.2f} m/s")
                
                wdir10 = ds.wrf_diag.wdir10()
                print(f"✓ WDIR10 computed: {wdir10.min().values:.2f} to {wdir10.max().values:.2f} degrees")
                
                # Validate wind direction range (0-360)
                if wdir10.min() >= 0 and wdir10.max() < 360:
                    print(f"✓ Wind direction range valid")
                else:
                    print(f"⚠️  Wind direction out of valid range")
                
                test_results.append(('U10', os.path.basename(wrf_file), 'PASS'))
                test_results.append(('V10', os.path.basename(wrf_file), 'PASS'))
                test_results.append(('WSPD10', os.path.basename(wrf_file), 'PASS'))
                test_results.append(('WDIR10', os.path.basename(wrf_file), 'PASS'))
            except Exception as e:
                print(f"✗ 10m winds failed: {e}")
                test_results.append(('WIND10', os.path.basename(wrf_file), 'FAIL'))
            
            ds.close()
            
        except Exception as e:
            print(f"\n✗ Failed to process file: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, _, result in test_results if result == 'PASS')
    failed = sum(1 for _, _, result in test_results if result == 'FAIL')
    total = len(test_results)
    
    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")
    
    if failed > 0:
        print(f"\n❌ FAILED TESTS:")
        for test, file, result in test_results:
            if result == 'FAIL':
                print(f"  - {test}: {file}")
        return False
    else:
        print(f"\n✅ ALL TESTS PASSED!")
        return True

if __name__ == "__main__":
    success = test_surface_variables()
    sys.exit(0 if success else 1)