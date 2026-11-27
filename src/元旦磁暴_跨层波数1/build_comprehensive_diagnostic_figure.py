"""Build the compact mechanism-chain figure for the 2025 New Year event.

The main figure is restricted to 40--80N and links four diagnostics: SABER
ozone anomaly, background-wind anomaly, stationary W1 refractive index, and
the W1 amplitude/phase response. ERA5 is used through 42.5 km and Aura MLS
GPH/temperature fits from 45 km upward. SABER nighttime ozone is available
only for 25 Dec--10 Jan.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LEGACY = PROJECT / "A研究进展报告" / "ERA5_2025元旦磁暴研究报告"
AUDIT = ROOT / "propagation_audit_20260927"

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
import xarray as xr

# The project-local netCDF4 wheel is needed by the Python 3.10 plotting
# environment. Append it after importing the core scientific stack so its
# bundled NumPy/Matplotlib copies cannot shadow the active environment.
sys.path.append(str(LEGACY / "plot_packages"))


OUT = ROOT / "results" / "comprehensive_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2024-12-15")
END = pd.Timestamp("2025-01-20")
DATES = pd.date_range(START, END, freq="D")
STORM = pd.Timestamp("2025-01-01")
BANDS = [(20, 40), (40, 60), (60, 80)]
LAT_CENTERS = np.arange(25.0, 80.0, 10.0)
HEIGHTS = np.arange(20.0, 90.1, 2.5)
LOWER = HEIGHTS <= 42.5
UPPER = HEIGHTS >= 45.0

A = 6_371_000.0
G = 9.80665
OMEGA = 7.2921159e-5
KAPPA = 0.2854
CP = 1004.0
H = 7000.0


def ascii_open(path: Path) -> xr.Dataset:
    """Open NetCDF through an ASCII temp path for Windows libraries."""
    target = Path(tempfile.gettempdir()) / ("diag_" + path.name)
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        shutil.copy2(path, target)
    ds = xr.open_dataset(target, engine="netcdf4")
    rename = {}
    for old, new in (("valid_time", "time"), ("pressure_level", "level")):
        if old in ds.dims or old in ds.coords:
            rename[old] = new
    ds = ds.rename(rename)
    for coordinate in ("latitude", "longitude", "level"):
        if coordinate in ds.coords:
            ds = ds.sortby(coordinate)
    return ds


def edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mid = (x[:-1] + x[1:]) / 2
    return np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]


def fourier1(values: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    return 2.0 * np.nanmean(values * np.exp(1j * np.deg2rad(lon_deg)), axis=-1)


def interp_profile(z: np.ndarray, value: np.ndarray, targets: np.ndarray) -> np.ndarray:
    good = np.isfinite(z) & np.isfinite(value)
    if good.sum() < 2:
        return np.full(targets.size, np.nan)
    z = z[good]
    value = value[good]
    order = np.argsort(z)
    z, value = z[order], value[order]
    unique = np.r_[True, np.diff(z) > 0.05]
    z, value = z[unique], value[unique]
    if z.size < 2:
        return np.full(targets.size, np.nan)
    return np.interp(targets, z, value, left=np.nan, right=np.nan)


def interp_complex(z: np.ndarray, value: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return interp_profile(z, value.real, targets) + 1j * interp_profile(z, value.imag, targets)


def load_era5() -> dict[str, np.ndarray]:
    files = sorted((AUDIT / "data" / "ERA5").glob("ERA5_6h_*.nc"))
    datasets = []
    for path in files:
        ds = ascii_open(path)[["z", "t", "u", "v"]]
        datasets.append(ds.load())
        ds.close()
    ds = xr.concat(datasets, dim="time").sortby("time")
    ds = ds.sel(time=slice(START, END + pd.Timedelta(hours=23)))
    _, unique = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(unique)).resample(time="1D").mean()

    times = pd.DatetimeIndex(ds.time.values)
    lon = ds.longitude.values
    lat = ds.latitude.values
    p_hpa = ds.level.values.astype(float)
    p_pa = p_hpa * 100.0
    coslat = np.cos(np.deg2rad(lat))
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))

    order = ("time", "level", "latitude", "longitude")
    z = ds.z.transpose(*order).values / G
    t = ds.t.transpose(*order).values
    u = ds.u.transpose(*order).values
    v = ds.v.transpose(*order).values
    zbar, tbar, ubar = [np.nanmean(x, axis=-1) for x in (z, t, u)]
    cz, ct, cu, cv = [fourier1(x, lon) for x in (z, t, u, v)]
    heat = 0.5 * np.real(cv * np.conj(ct))
    theta_bar = tbar * (1000.0 / p_hpa[None, :, None]) ** KAPPA
    theta_p = np.gradient(theta_bar, p_pa, axis=1)
    ctheta = ct * (1000.0 / p_hpa[None, :, None]) ** KAPPA
    uv1 = 0.5 * np.real(cu * np.conj(cv))
    vtheta1 = 0.5 * np.real(cv * np.conj(ctheta))
    fphi = -A * coslat[None, None, :] * uv1
    fp = A * coslat[None, None, :] * f[None, None, :] * vtheta1 / theta_p
    phi = np.deg2rad(lat)
    mer = np.gradient(fphi * coslat[None, None, :], phi, axis=2)
    mer /= A * coslat[None, None, :]
    div = mer + np.gradient(fp, p_pa, axis=1)
    accel = div / (A * coslat[None, None, :]) * 86400.0

    shape = (len(BANDS), len(DATES), HEIGHTS.size)
    amp = np.full(shape, np.nan)
    phase = np.full(shape, np.nan)
    flux = np.full(shape, np.nan)
    epfd = np.full(shape, np.nan)
    bg_u = np.full(shape, np.nan)
    input100 = np.full((len(BANDS), len(DATES)), np.nan)

    # Fixed-height zonal means on six 10-degree latitude cells for n^2.
    uz = np.full((len(DATES), LAT_CENTERS.size, HEIGHTS.size), np.nan)
    tz = np.full_like(uz, np.nan)

    for it, day in enumerate(times):
        date_index = DATES.get_indexer([day.normalize()])[0]
        if date_index < 0:
            continue
        for ilc, center in enumerate(LAT_CENTERS):
            mask = (lat >= center - 5) & (lat < center + 5)
            weights = coslat[mask]
            weights /= weights.sum()
            zz = np.sum(zbar[it][:, mask] * weights[None, :], axis=1) / 1000.0
            uu = np.sum(ubar[it][:, mask] * weights[None, :], axis=1)
            tt = np.sum(tbar[it][:, mask] * weights[None, :], axis=1)
            uz[date_index, ilc, LOWER] = interp_profile(zz, uu, HEIGHTS[LOWER])
            tz[date_index, ilc, LOWER] = interp_profile(zz, tt, HEIGHTS[LOWER])
        for ib, (lo, hi) in enumerate(BANDS):
            mask = (lat >= lo) & (lat <= hi)
            weights = coslat[mask]
            weights /= weights.sum()
            zz = np.sum(zbar[it][:, mask] * weights[None, :], axis=1) / 1000.0
            cc = np.sum(cz[it][:, mask] * weights[None, :], axis=1)
            hh = np.sum(heat[it][:, mask] * weights[None, :], axis=1)
            aa = np.sum(accel[it][:, mask] * weights[None, :], axis=1)
            uu = np.sum(ubar[it][:, mask] * weights[None, :], axis=1)
            ci = interp_complex(zz, cc, HEIGHTS[LOWER])
            amp[ib, date_index, LOWER] = np.abs(ci)
            phase[ib, date_index, LOWER] = np.angle(ci, deg=True)
            flux[ib, date_index, LOWER] = interp_profile(zz, hh, HEIGHTS[LOWER])
            epfd[ib, date_index, LOWER] = interp_profile(zz, aa, HEIGHTS[LOWER])
            bg_u[ib, date_index, LOWER] = interp_profile(zz, uu, HEIGHTS[LOWER])
            j100 = int(np.argmin(abs(p_hpa - 100.0)))
            input100[ib, date_index] = np.sum(heat[it, j100, mask] * weights)
    ds.close()
    return dict(amp=amp, phase=phase, flux=flux, epfd=epfd, u=bg_u,
                input100=input100, uz=uz, tz=tz)


def mls_files(product: str) -> dict[pd.Timestamp, Path]:
    result: dict[pd.Timestamp, Path] = {}
    legacy_name = "MLS_GPH_v06_20241225_20250110" if product == "GPH" else "MLS_Temperature_v06_20241225_20250110"
    sources = [AUDIT / "data" / "MLS" / product, ROOT / "data" / legacy_name]
    paths = [path for source in sources for path in source.glob("*.he5")]
    for path in sorted(paths):
        stamp = path.stem.rsplit("_", 1)[-1]
        day = pd.Timestamp(datetime.strptime(stamp, "%Yd%j").date())
        if START <= day <= END:
            # Sorted v06-02 names replace v06-01 if both exist.
            result[day] = path
    return result


def read_mls(path: Path, product: str) -> dict[str, np.ndarray]:
    with h5py.File(path) as handle:
        swath = handle[f"HDFEOS/SWATHS/{product}"]
        data = swath["Data Fields"]
        geo = swath["Geolocation Fields"]
        keys_d = ["L2gpValue", "L2gpPrecision", "Quality", "Convergence", "Status"]
        keys_g = ["Pressure", "Latitude", "Longitude", "LocalSolarTime"]
        return {k: data[k][:] for k in keys_d} | {k: geo[k][:] for k in keys_g}


def fit_mls_level(gph: dict, temp: dict, j: int, lo: float, hi: float) -> dict | None:
    lat = gph["Latitude"]
    lon = gph["Longitude"]
    lst = gph["LocalSolarTime"]
    keep = (lat >= lo) & (lat <= hi) & np.isfinite(lon) & np.isfinite(lst)
    for source in (gph, temp):
        keep &= ((source["Status"] % 2) == 0) & (source["Convergence"] < 1.03)
        keep &= (source["Quality"] > 0.2) & (source["L2gpPrecision"][:, j] > 0)
        keep &= np.isfinite(source["L2gpValue"][:, j])
    z = gph["L2gpValue"][:, j]
    tt = temp["L2gpValue"][:, j]
    keep &= (z > 0) & (z < 130000) & (tt > 100) & (tt < 350)
    if keep.sum() < 55:
        return None
    sectors = np.unique(np.floor((lon[keep] % 360) / 30).astype(int))
    if sectors.size < 9:
        return None
    la = lat[keep]
    lam = np.deg2rad(lon[keep] % 360)
    xlat = (la - (lo + hi) / 2) / max((hi - lo) / 2, 1)
    x = np.column_stack([
        np.ones(keep.sum()), xlat, xlat**2, np.where(lst[keep] < 8, 1.0, -1.0),
        np.cos(lam), np.sin(lam),
    ])
    w = np.cos(np.deg2rad(la))
    xw = x * np.sqrt(w)[:, None]
    y = np.column_stack([z[keep], tt[keep]])
    beta = np.linalg.lstsq(xw, y * np.sqrt(w)[:, None], rcond=None)[0]
    if np.linalg.cond(xw) > 35:
        return None
    cz = beta[-2, 0] + 1j * beta[-1, 0]
    ct = beta[-2, 1] + 1j * beta[-1, 1]
    center = (lo + hi) / 2
    fac = G / (2 * OMEGA * np.sin(np.deg2rad(center)) * A * np.cos(np.deg2rad(center)))
    return dict(z0=beta[0, 0], t0=beta[0, 1], cz=cz, ct=ct,
                flux=0.5 * fac * np.imag(cz * np.conj(ct)))


def load_mls(era: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    gfiles, tfiles = mls_files("GPH"), mls_files("Temperature")
    common = sorted(set(gfiles) & set(tfiles))
    upper_h = HEIGHTS[UPPER]
    for day in common:
        idx = DATES.get_indexer([day])[0]
        if idx < 0:
            continue
        gph, temp = read_mls(gfiles[day], "GPH"), read_mls(tfiles[day], "Temperature")
        if not np.array_equal(gph["Pressure"], temp["Pressure"]):
            continue
        p = gph["Pressure"].astype(float)
        jj = np.flatnonzero((p <= 3.0) & (p >= 0.001))
        for ib, (lo, hi) in enumerate(BANDS):
            rows = [(j, fit_mls_level(gph, temp, j, lo, hi)) for j in jj]
            rows = [(j, r) for j, r in rows if r is not None]
            if len(rows) < 3:
                continue
            z = np.array([r["z0"] / 1000 for _, r in rows])
            cc = np.array([r["cz"] for _, r in rows])
            ff = np.array([r["flux"] for _, r in rows])
            ci = interp_complex(z, cc, upper_h)
            era["amp"][ib, idx, UPPER] = np.abs(ci)
            era["phase"][ib, idx, UPPER] = np.angle(ci, deg=True)
            era["flux"][ib, idx, UPPER] = interp_profile(z, ff, upper_h)

        # Ten-degree zonal means used to derive geostrophic U and n^2.
        z_lat = np.full((LAT_CENTERS.size, jj.size), np.nan)
        t_lat = np.full_like(z_lat, np.nan)
        p_lat = np.broadcast_to(p[jj][None, :], z_lat.shape).copy()
        for ilc, center in enumerate(LAT_CENTERS):
            for ik, j in enumerate(jj):
                row = fit_mls_level(gph, temp, j, center - 5, center + 5)
                if row is not None:
                    z_lat[ilc, ik] = row["z0"] / 1000
                    t_lat[ilc, ik] = row["t0"]
        phi = np.deg2rad(LAT_CENTERS)
        f = 2 * OMEGA * np.sin(phi)
        # Geostrophic wind must be differentiated on pressure surfaces first;
        # differentiating after mapping GPH itself to fixed geometric heights
        # would trivially return zero because every latitude is then evaluated
        # at the same target height.
        dz_dphi_native = np.gradient(z_lat * 1000.0, phi, axis=0)
        ug_native = -G * dz_dphi_native / (A * f[:, None])
        ug = np.full((LAT_CENTERS.size, upper_h.size), np.nan)
        t0 = np.full_like(ug, np.nan)
        for ilc in range(LAT_CENTERS.size):
            ug[ilc] = interp_profile(z_lat[ilc], ug_native[ilc], upper_h)
            t0[ilc] = interp_profile(z_lat[ilc], t_lat[ilc], upper_h)
        era["uz"][idx][:, UPPER] = ug
        era["tz"][idx][:, UPPER] = t0
        for ib in range(len(BANDS)):
            era["u"][ib, idx, UPPER] = np.nanmean(ug[2 * ib:2 * ib + 2], axis=0)
    return era


def refractive_index(uz: np.ndarray, tz: np.ndarray) -> np.ndarray:
    """Stationary zonal-wavenumber-1 QG refractive index, scaled by 1e12."""
    result = np.full((len(BANDS), len(DATES), HEIGHTS.size), np.nan)
    z_m = HEIGHTS * 1000.0
    phi = np.deg2rad(LAT_CENTERS)
    cosphi = np.cos(phi)
    f = 2 * OMEGA * np.sin(phi)
    beta = 2 * OMEGA * np.cos(phi) / A
    rho = np.exp(-z_m / H)
    for it in range(len(DATES)):
        u = uz[it]
        t = tz[it]
        dtdz = np.gradient(t, z_m, axis=1)
        n2_bv = G / t * (dtdz + G / CP)
        mer_inner = np.gradient(u * cosphi[:, None], phi, axis=0) / cosphi[:, None]
        mer_term = np.gradient(mer_inner, phi, axis=0) / A**2
        du_dz = np.gradient(u, z_m, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vertical_inside = rho[None, :] * f[:, None]**2 / n2_bv * du_dz
            vertical_term = np.gradient(vertical_inside, z_m, axis=1) / rho[None, :]
            qy = beta[:, None] - mer_term - vertical_term
            ref = qy / u - 1.0 / (A**2 * cosphi[:, None]**2) - f[:, None]**2 / (4 * n2_bv * H**2)
        ref[(np.abs(u) < 5) | (n2_bv <= 0)] = np.nan
        for ib in range(len(BANDS)):
            result[ib, it] = np.nanmean(ref[2 * ib:2 * ib + 2], axis=0) * 1e12
    return result


def load_saber_ozone() -> np.ndarray:
    root = LEGACY / "data" / "direct_observations_2025_new_year" / "SABER_L2A_temperature_ozone"
    alt = np.arange(50.0, 96.1, 2.5)
    raw = np.full((len(BANDS), len(DATES), alt.size), np.nan)
    for iday, day in enumerate(DATES):
        folder = root / day.strftime("%Y-%m-%d")
        if not folder.exists():
            continue
        vals = [[[] for _ in alt] for _ in BANDS]
        for path in folder.glob("*.nc"):
            ds = ascii_open(path)
            need = ds[["tpaltitude", "tplatitude", "tpSolarZen", "O3_96"]].load()
            ds.close()
            zz = np.asarray(need.tpaltitude.values, float)
            la = np.asarray(need.tplatitude.values, float)
            sza = np.asarray(need.tpSolarZen.values, float)
            o3 = np.asarray(need.O3_96.values, float) * 1e6
            for ib, (lo, hi) in enumerate(BANDS):
                good = (la >= lo) & (la <= hi) & (sza > 95) & np.isfinite(o3) & (o3 > 0) & (o3 < 100)
                for iz, target in enumerate(alt):
                    selected = o3[good & (abs(zz - target) <= 1.25)]
                    if selected.size:
                        vals[ib][iz].extend(selected.tolist())
        for ib in range(len(BANDS)):
            for iz in range(alt.size):
                if vals[ib][iz]:
                    raw[ib, iday, iz] = np.nanmedian(vals[ib][iz])
    anomaly = np.full((len(BANDS), len(DATES), HEIGHTS.size), np.nan)
    base_mask = (DATES >= "2024-12-25") & (DATES <= "2024-12-30")
    for ib in range(len(BANDS)):
        base = np.nanmedian(raw[ib, base_mask], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100 * (raw[ib] - base[None, :]) / base[None, :]
        for it in range(len(DATES)):
            anomaly[ib, it] = interp_profile(alt, pct[it], HEIGHTS)
    return anomaly


def plot(data: dict[str, np.ndarray], n2: np.ndarray, ozone: np.ndarray) -> Path:
    width_cm = 20.0
    height_cm = 20.5
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.titleweight": "normal",
        "xtick.direction": "in", "ytick.direction": "in",
    })
    fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54))
    grid = fig.add_gridspec(
        5, 3, height_ratios=(1.0, 0.10, 1.0, 1.0, 1.0),
        left=0.06, right=0.95, bottom=0.09, top=0.90,
        wspace=0.18, hspace=0.24,
    )
    axes = np.empty((4, 3), dtype=object)
    color_axes = np.empty((4, 3), dtype=object)
    grid_rows = (0, 2, 3, 4)
    for row in range(4):
        for col in range(3):
            panel = grid[grid_rows[row], col].subgridspec(
                1, 2, width_ratios=(1, 0.052), wspace=0.055)
            shared_axis = axes[1, col] if row >= 2 else None
            axes[row, col] = fig.add_subplot(
                panel[0, 0], sharex=shared_axis)
            color_axes[row, col] = fig.add_subplot(panel[0, 1])
    x = mdates.date2num(DATES.to_pydatetime())
    xe, ze = edges(x), edges(HEIGHTS)
    baseline = (DATES >= "2024-12-15") & (DATES <= "2024-12-24")
    wind_anomaly = data["u"] - np.nanmean(data["u"][:, baseline], axis=1)[:, None, :]
    band_indices = (0, 1, 2)
    row_titles = ("O$_3$ anomaly", "Wind anomaly",
                  "W1 $n^2$", "W1 amplitude")
    fields = (ozone, wind_anomaly, n2, data["amp"])
    cmaps = ("BrBG", "RdBu_r", "PuOr", "viridis")
    units = ("%", "m s$^{-1}$", "$10^{-12}$ m$^{-2}$", "m")

    for row, (field, cmap, title, unit) in enumerate(
            zip(fields, cmaps, row_titles, units)):
        for col, ib in enumerate(band_indices):
            lo, hi = BANDS[ib]
            ax = axes[row, col]
            panel_field = field[ib]
            finite = panel_field[np.isfinite(panel_field)]
            if finite.size:
                low, high = float(finite.min()), float(finite.max())
            else:
                low, high = 0.0, 1.0
            if low == high:
                high = low + 1.0
            if row == 3 and col == 0:
                high = 520.0
            if row == 3:
                norm = Normalize(vmin=max(0.0, low), vmax=high)
            elif low < 0 < high:
                norm = TwoSlopeNorm(vcenter=0.0, vmin=low, vmax=high)
            else:
                norm = Normalize(vmin=low, vmax=high)
            mesh = ax.pcolormesh(xe, ze, panel_field.T, cmap=cmap, norm=norm,
                                 shading="flat", rasterized=True)
            ax.axvline(mdates.date2num(STORM), color="black", ls="--", lw=0.9)
            if row > 0:
                ax.axhline(43.75, color="0.25", ls=":", lw=0.8)
            if row == 0:
                valid_dates = np.flatnonzero(np.any(np.isfinite(panel_field), axis=1))
                valid_heights = np.flatnonzero(np.any(np.isfinite(panel_field), axis=0))
                if valid_dates.size:
                    ax.set_xlim(x[valid_dates[0]] - 0.5, x[valid_dates[-1]] + 0.5)
                if valid_heights.size:
                    y0 = HEIGHTS[valid_heights[0]]
                    y1 = HEIGHTS[valid_heights[-1]]
                    ax.set_ylim(y0 - 1.25, y1 + 1.25)
                    ticks = np.arange(np.ceil(y0 / 10) * 10, y1 + 0.1, 10)
                    ax.set_yticks(ticks)
            else:
                ax.set_ylim(20, 90)
                ax.set_yticks([20, 40, 60, 80])
                ax.set_xlim(xe[0], xe[-1])
            ax.tick_params(which="both", direction="in", top=True, right=True,
                           labelleft=(col == 0), labelbottom=(row in (0, 3)))
            if col == 0:
                ax.set_title(f"({chr(97 + row)}) {title}", loc="left", pad=5)
                if row in (0, 2):
                    ax.set_ylabel("Approx. alt. (km)", labelpad=2)
            if row == 0:
                ax.text(0.5, 1.14, f"{lo}°–{hi}°N", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=12)
            cbar = fig.colorbar(mesh, cax=color_axes[row, col])
            cbar.ax.set_title(unit, fontsize=9, pad=2)
            cbar.ax.tick_params(labelsize=9, direction="in", length=2.5, pad=1)
            if row in (0, 3):
                ax.xaxis.set_major_locator(
                    mdates.DayLocator(interval=4 if row == 0 else 8))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                ax.tick_params(axis="x", rotation=25 if row == 0 else 30)
                if row == 3:
                    ax.set_xlabel("UTC date")
            else:
                ax.tick_params(labelbottom=False)
    fig.suptitle(
        "Upper forcing, waveguide and W1 response\n"
        "15 Dec 2024–20 Jan 2025 | 20°–80°N | 20–90 km",
        fontsize=14, weight="normal", y=0.998,
    )
    png = OUT / "2025_new_year_storm_comprehensive_W1_diagnostics.png"
    fig.savefig(png, dpi=600)
    plt.close(fig)
    return png


def save_summary(data: dict[str, np.ndarray], n2: np.ndarray, ozone: np.ndarray) -> None:
    rows = []
    for ib, (lo, hi) in enumerate(BANDS):
        for it, day in enumerate(DATES):
            for iz, height in enumerate(HEIGHTS):
                rows.append({
                    "date": day.strftime("%Y-%m-%d"), "lat_band": f"{lo}-{hi}N",
                    "height_km": height, "w1_amplitude_m": data["amp"][ib, it, iz],
                    "w1_phase_deg_Epositive": data["phase"][ib, it, iz],
                    "w1_heat_flux_proxy_Kms": data["flux"][ib, it, iz],
                    "w1_epfd_accel_ms_day": data["epfd"][ib, it, iz],
                    "background_u_ms": data["u"][ib, it, iz],
                    "refractive_index_1e12_m2": n2[ib, it, iz],
                    "saber_ozone_anomaly_percent": ozone[ib, it, iz],
                    "wave1_100hPa_input_Kms": data["input100"][ib, it],
                })
    pd.DataFrame(rows).to_csv(OUT / "comprehensive_W1_diagnostic_values.csv", index=False)


def main() -> None:
    data = load_era5()
    data = load_mls(data)
    n2 = refractive_index(data["uz"], data["tz"])
    ozone = load_saber_ozone()
    save_summary(data, n2, ozone)
    png = plot(data, n2, ozone)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
