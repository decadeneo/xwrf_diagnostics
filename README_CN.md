# xwrf-diagnostics

一个 xarray accessor 插件，用于连接 xwrf/xarray 和 wrf-python 的底层计算 API，提供 WRF 模型诊断变量的便捷计算。

## 功能特性

- **全面的诊断变量**: 包含热力学、动力学和地表变量
- **xarray 集成**: 通过 accessor 模式无缝使用 xarray 数据集
- **变量名映射**: 自动处理 WRF 变量命名约定
- **交错网格支持**: 自动去交错处理用于质量网格计算
- **单位转换**: 内置单位转换支持 (K/°C, Pa/hPa)
- **Dask 支持**: 支持大规模数据并行处理

## 安装

```bash
pip install xarray wrf-python numpy
```

然后将 `xwrf_diagnostics.py` 复制到项目目录中。

## 使用方法

```python
import xarray as xr
import xwrf_diagnostics  # 导入以注册 accessor

# 加载 WRF 输出
ds = xr.open_dataset("wrfout_d01_2022-07-14_00:00:00")

# 计算诊断变量
slp = ds.wrf_diag.slp()          # 海平面气压
rh = ds.wrf_diag.rh()             # 相对湿度
t2 = ds.wrf_diag.t2()             # 2米温度
rh2 = ds.wrf_diag.rh2()           # 2米相对湿度
wspd10 = ds.wrf_diag.wspd10()     # 10米风速
cape = ds.wrf_diag.cape_2d()      # CAPE 和 CIN (返回 Dataset)
```

## 可用诊断变量

### 热力学变量
- `slp()` - 海平面气压
- `rh()` - 相对湿度
- `tk()` / `tc()` - 温度 (开尔文/摄氏度)
- `eth()` - 相当位温
- `td()` - 露点温度
- `pw()` - 可降水量

### 动力学变量
- `avo()` - 绝对涡度
- `omega()` - Omega (垂直气压速度)
- `udhel()` - 上升螺旋度

### 高级诊断
- `dbz()` - 雷达反射率
- `cape_2d()` - CAPE, CIN, LCL, LFC
- `cloudfrac()` - 低/中/高云量
- `ctt()` - 云顶温度

### 地表变量 (2米和10米)
- `t2()` / `t2c()` - 2米温度 (K/°C)
- `rh2()` - 2米相对湿度
- `td2()` - 2米露点温度
- `q2()` - 2米比湿
- `psfc()` - 地表气压
- `u10()` / `v10()` - 10米 U/V 风分量
- `wspd10()` - 10米风速
- `wdir10()` - 10米风向

## 变量名映射

accessor 自动处理常见的变量命名变体：

| WRF 标准名称 | CF 约定名称 | 备用名称 |
|--------------|---------------|-------------------|
| P | pres | - |
| T | theta | - |
| PSFC | psfc | - |
| T2 | - | - |
| Q2 | QV2M | - |
| U10 | u10 | - |
| V10 | v10 | - |

## 依赖要求

- Python >= 3.7
- xarray >= 0.18
- wrf-python >= 1.3
- numpy >= 1.20

## 测试

运行测试套件：

```bash
python test_surface_vars.py
```

## 示例

### 使用 dask 多文件处理

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

### 绘制地表变量

```python
import matplotlib.pyplot as plt

t2 = ds.wrf_diag.t2c()  # 2米温度（摄氏度）
t2[0].plot()
plt.title("2米温度")
plt.show()
```

## 许可证

MIT

## 贡献

该插件设计用于集成到 xwrf 库中，欢迎贡献！

## 致谢

- [wrf-python](https://wrf-python.readthedocs.io/) - 计算例程
- [xwrf](https://xwrf.readthedocs.io/) - WRF-xarray 集成
- [xarray](https://xarray.dev/) - 标记多维数组
