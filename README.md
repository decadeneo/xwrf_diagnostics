# xwrf-diagnostics

An xarray accessor plugin that bridges xwrf/xarray with wrf-python's raw computational APIs for WRF model diagnostics.

## Features

- **Comprehensive diagnostics**: Thermodynamic, dynamic, and surface variables
- **xarray integration**: Seamless use with xarray datasets via accessor pattern
- **Variable mapping**: Automatic handling of WRF variable naming conventions
- **Staggered grid support**: Automatic destaggering for mass grid calculations
- **Unit conversions**: Built-in unit conversion support (K/°C, Pa/hPa)
- **Dask support**: Parallel processing for large datasets

## Installation

```bash
pip install xarray wrf-python numpy
```

Then copy `xwrf_diagnostics.py` to your project directory.

## Usage

```python
import xarray as xr
import xwrf_diagnostics  # Import to register the accessor

# Load WRF output
ds = xr.open_dataset("wrfout_d01_2022-07-14_00:00:00")

# Calculate diagnostics
slp = ds.wrf_diag.slp()          # Sea level pressure
rh = ds.wrf_diag.rh()             # Relative humidity
t2 = ds.wrf_diag.t2()             # 2m temperature
rh2 = ds.wrf_diag.rh2()           # 2m relative humidity
wspd10 = ds.wrf_diag.wspd10()     # 10m wind speed
cape = ds.wrf_diag.cape_2d()      # CAPE and CIN (returns Dataset)
```

## Available Diagnostics

### Thermodynamic Variables
- `slp()` - Sea level pressure
- `rh()` - Relative humidity
- `tk()` / `tc()` - Temperature (Kelvin/Celsius)
- `eth()` - Equivalent potential temperature
- `td()` - Dewpoint temperature
- `pw()` - Precipitable water

### Dynamic Variables
- `avo()` - Absolute vorticity
- `omega()` - Omega (vertical pressure velocity)
- `udhel()` - Updraft helicity

### Advanced Diagnostics
- `dbz()` - Radar reflectivity
- `cape_2d()` - CAPE, CIN, LCL, LFC
- `cloudfrac()` - Low/mid/high cloud fractions
- `ctt()` - Cloud top temperature

### Surface Variables (2m and 10m)
- `t2()` / `t2c()` - 2m temperature (K/°C)
- `rh2()` - 2m relative humidity
- `td2()` - 2m dewpoint temperature
- `q2()` - 2m specific humidity
- `psfc()` - Surface pressure
- `u10()` / `v10()` - 10m U/V wind components
- `wspd10()` - 10m wind speed
- `wdir10()` - 10m wind direction

## Variable Mapping

The accessor automatically handles common variable naming variations:

| WRF Standard | CF Convention | Alternative Names |
|--------------|---------------|-------------------|
| P | pres | - |
| T | theta | - |
| PSFC | psfc | - |
| T2 | - | - |
| Q2 | QV2M | - |
| U10 | u10 | - |
| V10 | v10 | - |

## Requirements

- Python >= 3.7
- xarray >= 0.18
- wrf-python >= 1.3
- numpy >= 1.20

## Testing

Run the test suite:

```bash
python test_surface_vars.py
```

## Examples

### Multi-file processing with dask

```python
import glob
import xarray as xr
import xwrf_diagnostics

files = sorted(glob.glob("wrfout_d02_*.nc"))

ds = xr.open_mfdataset(
    files,
    parallel=True,
    concat_dim="Time",
    combine="nested",
    chunks={'Time': 1},
)

slp = ds.wrf_diag.slp()
```

### Plotting surface variables

```python
import matplotlib.pyplot as plt

t2 = ds.wrf_diag.t2c()  # 2m temperature in Celsius
t2[0].plot()
plt.title("2m Temperature")
plt.show()
```

## License

MIT

## Contributing

This plugin is designed to be integrated into the xwrf library. Contributions are welcome!

## Acknowledgments

- [wrf-python](https://wrf-python.readthedocs.io/) for the computational routines
- [xwrf](https://xwrf.readthedocs.io/) for WRF-xarray integration
- [xarray](https://xarray.dev/) for labeled multidimensional arrays
