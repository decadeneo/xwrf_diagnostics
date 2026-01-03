import xarray as xr
import numpy as np
import warnings

# 尝试导入 wrf-python
try:
    import wrf
    WRF_AVAILABLE = True
except ImportError:
    WRF_AVAILABLE = False
    warnings.warn("wrf-python not found. Diagnostic calculations will fail.")

@xr.register_dataset_accessor("wrf_diag")
class WRFDiagnosticsAccessor:
    """
    xarray accessor to calculate WRF diagnostics using wrf-python's raw computational routines.
    Bridge the gap between xwrf/xarray and wrf-python raw APIs.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _get_var(self, names, mandatory=True):
        """Helper to find a variable in the dataset (case insensitive)."""
        if isinstance(names, str):
            names = [names]
        
        for name in names:
            if name in self._obj:
                return self._obj[name]
            # Handle standard xwrf/cf names
            if name == 'P' and 'pres' in self._obj: return self._obj['pres']
            if name == 'T' and 'theta' in self._obj: return self._obj['theta']
            
        if mandatory:
            missing_names = "/".join(names)
            raise KeyError(f"Variable '{missing_names}' not found in Dataset.")
        return None

    def _destagger(self, var, dim_name):
        """Helper to destagger a variable along a dimension."""
        if dim_name in var.dims:
            axis = var.get_axis_num(dim_name)
            slices_left = [slice(None)] * var.ndim
            slices_right = [slice(None)] * var.ndim
            slices_left[axis] = slice(None, -1)
            slices_right[axis] = slice(1, None)
            return 0.5 * (var.values[tuple(slices_left)] + var.values[tuple(slices_right)])
        return var.values

    def _wrap_result(self, data, ref_var, name, units, description):
        """Wrap numpy result back into xarray DataArray."""
        dims = list(ref_var.dims)
        coords = dict(ref_var.coords)
        
        # 1. 维度完全一致
        if data.ndim == len(dims):
            final_dims = dims
            
        # 2. 维度减少了1维 (通常是垂直层)
        elif data.ndim == len(dims) - 1:
            vert_names = ['bottom_top', 'bottom_top_stag', 'nz', 'lev', 'z']
            final_dims = [d for d in dims if d not in vert_names]
            
            # 兜底匹配
            if len(final_dims) != data.ndim:
                if len(dims) == 4: final_dims = [dims[0], dims[2], dims[3]]
                elif len(dims) == 3: final_dims = [dims[1], dims[2]]
        else:
            return xr.DataArray(data, name=name, attrs={'units': units, 'description': description})

        # 4. 筛选坐标
        final_coords = {}
        for k, v in coords.items():
            if set(v.dims).issubset(set(final_dims)):
                final_coords[k] = v

        return xr.DataArray(
            data, coords=final_coords, dims=final_dims, name=name, 
            attrs={'units': units, 'description': description}
        )

    def _get_base_vars(self, need_height_m=False, need_tk=False):
        """Extract and compute common Mass Grid variables (Full P, TK, Height, QV)"""
        # 1. Pressure (Pa)
        if 'air_pressure' in self._obj:
            full_p = self._obj['air_pressure'].values
        else:
            P = self._get_var(['P', 'pres'])
            PB = self._get_var('PB', mandatory=False)
            full_p = P.values + (PB.values if PB is not None else 0)

        # 2. Temperature (TK)
        T = self._get_var(['T', 'theta'])
        tk = None
        if need_tk:
            theta = T.values
            if np.nanmean(theta) < 200: theta = theta + 300.0
            tk = theta * ((full_p / 100000.0) ** 0.286)

        # 3. QV
        QV = self._get_var(['QVAPOR', 'QV'])
        qv_val = QV.values

        # 4. Height (m)
        height = None
        if need_height_m:
            PH = self._get_var('PH')
            PHB = self._get_var('PHB', mandatory=False)
            full_ph_stag = PH.values + (PHB.values if PHB is not None else 0)
            
            # Destagger Z -> Mass Grid
            if full_ph_stag.ndim >= 3 and full_ph_stag.shape[-3] == full_p.shape[-3] + 1:
                slices_l = [slice(None)] * full_ph_stag.ndim
                slices_r = [slice(None)] * full_ph_stag.ndim
                slices_l[-3] = slice(None, -1)
                slices_r[-3] = slice(1, None)
                full_ph = 0.5 * (full_ph_stag[tuple(slices_l)] + full_ph_stag[tuple(slices_r)])
            else:
                full_ph = full_ph_stag
            height = full_ph / 9.81

        return full_p, tk, height, qv_val

    # ==========================================
    # 1. Basic Thermodynamics
    # ==========================================

    def slp(self, units="hPa"):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        full_p, tk, height, qv = self._get_base_vars(need_height_m=True, need_tk=True)
        res = wrf.slp(height, tk, full_p, qv)
        return self._wrap_result(res, self._get_var(['T', 'theta']), 'slp', units, 'Sea Level Pressure')

    def dbz(self, use_varint=True, use_liqskin=False):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        full_p, tk, _, qv = self._get_base_vars(need_height_m=False, need_tk=True)
        QR = self._get_var(['QRAIN', 'QR']).values
        def get_vals(name):
            v = self._get_var(name, mandatory=False)
            return v.values if v is not None else np.zeros(full_p.shape)
        QS = get_vals(['QSNOW', 'QS'])
        QG = get_vals(['QGRAUP', 'QG'])
        res = wrf.dbz(full_p, tk, qv, QR, QS, QG, use_varint=use_varint, use_liqskin=use_liqskin)
        return self._wrap_result(res, self._get_var(['T', 'theta']), 'dbz', 'dBZ', 'Radar Reflectivity')

    def rh(self):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        full_p, tk, _, qv = self._get_base_vars(need_height_m=False, need_tk=True)
        res = wrf.rh(qv, full_p, tk)
        return self._wrap_result(res, self._get_var(['T', 'theta']), 'rh', '%', 'Relative Humidity')
    
    def tk(self):
        _, tk, _, _ = self._get_base_vars(need_height_m=False, need_tk=True)
        return self._wrap_result(tk, self._get_var(['T', 'theta']), 'tk', 'K', 'Temperature')
    
    def tc(self):
        """Temperature in Celsius."""
        tk = self.tk()
        tc = tk - 273.15
        tc.attrs['units'] = 'degC'
        tc.name = 'tc'
        return tc

    def eth(self):
        full_p, tk, _, qv = self._get_base_vars(need_height_m=False, need_tk=True)
        res = wrf.eth(qv, tk, full_p)
        return self._wrap_result(res, self._get_var(['T', 'theta']), 'eth', 'K', 'Equivalent Potential Temperature')

    def td(self):
        full_p, _, _, qv = self._get_base_vars(need_height_m=False, need_tk=False)
        res = wrf.td(full_p * 0.01, qv)
        return self._wrap_result(res, self._get_var(['T', 'theta']), 'td', 'degC', 'Dewpoint Temperature')
    
    def pw(self):
        full_p, tk, height, qv = self._get_base_vars(need_height_m=True, need_tk=True)
        res = wrf.pw(full_p, tk, qv, height)
        return self._wrap_result(res, self._get_var(['T', 'theta']), 'pw', 'kg m-2', 'Precipitable Water')

    # ==========================================
    # 2. Advanced Diagnostics
    # ==========================================

    def cape_2d(self):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        full_p, tk, height, qv = self._get_base_vars(need_height_m=True, need_tk=True)
        HGT = self._get_var(['HGT', 'terrain'])
        PSFC = self._get_var(['PSFC', 'psfc'])
        res = wrf.cape_2d(full_p * 0.01, tk, qv, height, HGT.values, PSFC.values * 0.01, ter_follow=True)
        
        ds_out = xr.Dataset(coords=self._obj.coords)
        ref = self._get_var('T')
        def wrap(d, n, u, desc): return self._wrap_result(d, ref, n, u, desc)
        ds_out['mcape'] = wrap(res[0], 'mcape', 'J kg-1', 'Max CAPE')
        ds_out['mcin']  = wrap(res[1], 'mcin',  'J kg-1', 'Max CIN')
        ds_out['lcl']   = wrap(res[2], 'lcl',   'm', 'LCL')
        ds_out['lfc']   = wrap(res[3], 'lfc',   'm', 'LFC')
        return ds_out

    def cloudfrac(self):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        full_p, tk, _, qv = self._get_base_vars(need_height_m=False, need_tk=True)
        relh = wrf.rh(qv, full_p, tk)
        res = wrf.cloudfrac(full_p, relh, vert_inc_w_height=0, low_thresh=97000, mid_thresh=80000, high_thresh=45000)
        
        ds_out = xr.Dataset(coords=self._obj.coords)
        ref = self._get_var('T')
        def wrap(d, n, desc): return self._wrap_result(d, ref, n, '%', desc)
        ds_out['low_cloud_frac'] = wrap(res[0], 'low_cf', 'Low Cloud Frac')
        ds_out['mid_cloud_frac'] = wrap(res[1], 'mid_cf', 'Mid Cloud Frac')
        ds_out['high_cloud_frac'] = wrap(res[2], 'high_cf', 'High Cloud Frac')
        return ds_out
        
    def ctt(self):
        full_p, tk, height, qv = self._get_base_vars(need_height_m=True, need_tk=True)
        QCLOUD = self._get_var(['QCLOUD', 'QC'])
        HGT = self._get_var(['HGT', 'terrain'])
        res = wrf.ctt(full_p * 0.01, tk, qv, QCLOUD.values, height, HGT.values)
        return self._wrap_result(res, self._get_var('T'), 'ctt', 'degC', 'Cloud Top Temperature')

    # ==========================================
    # 3. Dynamics (Vorticity, Helicity, Omega)
    # ==========================================

    def omega(self):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        full_p, tk, _, qv = self._get_base_vars(need_height_m=False, need_tk=True)
        W = self._get_var('W')
        w_mass = self._destagger(W, 'bottom_top_stag')
        res = wrf.omega(qv, tk, w_mass, full_p)
        return self._wrap_result(res, self._get_var('T'), 'omega', 'Pa s-1', 'Omega')

    def avo(self):
        """Absolute Vorticity. Requires Staggered U/V and Staggered Map Factors."""
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        
        U = self._get_var('U') # Staggered X
        V = self._get_var('V') # Staggered Y
        
        # 修复点：显式获取交错网格的 MAPFAC_U 和 MAPFAC_V
        try:
            MSFU = self._get_var(['MAPFAC_U', 'msfu']).values
            MSFV = self._get_var(['MAPFAC_V', 'msfv']).values
            MSFM = self._get_var(['MAPFAC_M', 'MAPFAC_MS']).values
        except KeyError:
             raise KeyError("Calculation of 'avo' requires MAPFAC_U and MAPFAC_V. Please ensure these variables are in your dataset.")

        F = self._get_var(['F', 'cor', 'Coriolis'])
        # dx = self._obj.attrs.get('DX', 30000)
        # dy = self._obj.attrs.get('DY', 30000)
        
        # 1. 尝试大写 DX
        if 'DX' in self._obj.attrs:
            dx = self._obj.attrs['DX']
            dy = self._obj.attrs['DY']
        # 2. 尝试小写 dx (有些处理过的 nc 文件属性名可能是小写)
        elif 'dx' in self._obj.attrs:
            dx = self._obj.attrs['dx']
            dy = self._obj.attrs['dy']
        # 3. 实在找不到，直接报错，强迫用户检查文件
        else:
            raise KeyError("Global attributes 'DX' and 'DY' (grid spacing) are missing from the Dataset. "
                           "Calculation requires grid spacing. Please add ds.attrs['DX'] = ... manually.")
        # 此时 U, V, MSFU, MSFV, MSFM 维度应该完全匹配，不会再报 Shape Error
        res = wrf.avo(U.values, V.values, MSFU, MSFV, MSFM, F.values, dx, dy)
        return self._wrap_result(res, self._get_var('T'), 'avo', '10-5 s-1', 'Absolute Vorticity')

    def udhel(self):
        if not WRF_AVAILABLE: raise ImportError("wrf-python required")
        PH = self._get_var('PH')
        PHB = self._get_var('PHB', mandatory=False)
        zstag = (PH.values + (PHB.values if PHB is not None else 0)) / 9.81
        W = self._get_var('W')
        U = self._get_var('U')
        V = self._get_var('V')
        u_mass = self._destagger(U, 'west_east_stag')
        v_mass = self._destagger(V, 'south_north_stag')
        MAPFAC_M = self._get_var(['MAPFAC_M', 'MAPFAC_MS'])
        dx = self._obj.attrs.get('DX', 30000)
        dy = self._obj.attrs.get('DY', 30000)
        res = wrf.udhel(zstag, MAPFAC_M.values, u_mass, v_mass, W.values, dx, dy)
        return self._wrap_result(res, self._get_var('T'), 'udhel', 'm2 s-2', 'Updraft Helicity')

    # ==========================================
    # 4. Surface Variables (2m & 10m)
    # ==========================================

    def t2(self, units="K"):
        """2m Temperature."""
        T2 = self._get_var(['T2', 'T2'])
        t2 = T2.values
        if units.lower() == "k":
            return self._wrap_result(t2, T2, 't2', 'K', '2m Temperature')
        elif units.lower() == "c":
            tc = t2 - 273.15
            return self._wrap_result(tc, T2, 't2c', 'degC', '2m Temperature (Celsius)')
        else:
            return self._wrap_result(t2, T2, 't2', units, '2m Temperature')

    def t2c(self):
        """2m Temperature in Celsius."""
        return self.t2(units="c")

    def q2(self):
        """2m Specific Humidity."""
        Q2 = self._get_var(['Q2', 'QV2M'])
        return self._wrap_result(Q2.values, Q2, 'q2', 'kg kg-1', '2m Specific Humidity')

    def psfc(self, units="Pa"):
        """Surface Pressure."""
        PSFC = self._get_var(['PSFC', 'psfc', 'PSFC'])
        psfc = PSFC.values
        if units.lower() == "pa":
            return self._wrap_result(psfc, PSFC, 'psfc', 'Pa', 'Surface Pressure')
        elif units.lower() in ["hpa", "mb"]:
            psfc_hpa = psfc * 0.01
            return self._wrap_result(psfc_hpa, PSFC, 'psfc', 'hPa', 'Surface Pressure')
        else:
            return self._wrap_result(psfc, PSFC, 'psfc', units, 'Surface Pressure')

    def u10(self):
        """10m U-component of wind."""
        U10 = self._get_var(['U10', 'u10'])
        return self._wrap_result(U10.values, U10, 'u10', 'm s-1', '10m U-wind')

    def v10(self):
        """10m V-component of wind."""
        V10 = self._get_var(['V10', 'v10'])
        return self._wrap_result(V10.values, V10, 'v10', 'm s-1', '10m V-wind')

    def wspd10(self):
        """10m Wind Speed."""
        U10 = self._get_var(['U10', 'u10'])
        V10 = self._get_var(['V10', 'v10'])
        wspd = np.sqrt(U10.values**2 + V10.values**2)
        return self._wrap_result(wspd, U10, 'wspd10', 'm s-1', '10m Wind Speed')

    def wdir10(self):
        """10m Wind Direction."""
        U10 = self._get_var(['U10', 'u10'])
        V10 = self._get_var(['V10', 'v10'])
        u = U10.values
        v = V10.values
        # Calculate wind direction (meteorological convention: from which direction wind blows)
        wdir = np.degrees(np.arctan2(-u, -v))
        # Convert to 0-360 range
        wdir = np.where(wdir < 0, wdir + 360, wdir)
        return self._wrap_result(wdir, U10, 'wdir10', 'degrees', '10m Wind Direction')

    def rh2(self):
        """2m Relative Humidity."""
        Q2 = self._get_var(['Q2', 'QV2M'])
        T2 = self._get_var(['T2', 'T2'])
        PSFC = self._get_var(['PSFC', 'psfc'])
        qv = Q2.values
        p = PSFC.values
        tk = T2.values
        
        if WRF_AVAILABLE:
            res = wrf.rh(qv, p, tk)
        else:
            # Simple RH approximation: RH ~ (Q/Qs) * 100
            # Qs (saturation mixing ratio) approximation
            es = 6.112 * np.exp(17.67 * (tk - 273.15) / (tk - 273.15 + 243.5))  # hPa
            qs = 0.622 * es / (p * 0.01 - es)
            res = (qv / qs) * 100
            res = np.clip(res, 0, 100)
        return self._wrap_result(res, Q2, 'rh2', '%', '2m Relative Humidity')

    def td2(self):
        """2m Dewpoint Temperature."""
        if not WRF_AVAILABLE:
            Q2 = self._get_var(['Q2', 'QV2M'])
            PSFC = self._get_var(['PSFC', 'psfc'])
            qv = Q2.values
            p = PSFC.values * 0.01  # Convert to hPa
            # Approximate dewpoint calculation
            # Magnus formula approximation
            rh = self.rh2().values / 100.0
            # Simple dewpoint approximation from RH and T
            T2 = self._get_var(['T2', 'T2']).values
            es = 6.112 * np.exp(17.67 * (T2 - 273.15) / (T2 - 273.15 + 243.5))
            e = es * rh
            # Inverse Magnus for dewpoint
            td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
            res = td + 273.15  # Convert to Kelvin
            return self._wrap_result(res, Q2, 'td2', 'K', '2m Dewpoint Temperature')
        else:
            Q2 = self._get_var(['Q2', 'QV2M'])
            PSFC = self._get_var(['PSFC', 'psfc'])
            res = wrf.td(PSFC.values * 0.01, Q2.values)
            return self._wrap_result(res, Q2, 'td2', 'K', '2m Dewpoint Temperature')