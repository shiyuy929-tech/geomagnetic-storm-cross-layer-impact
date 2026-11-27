"""Daily tropospheric/stratospheric vortex maps, 28 Dec 2024 - 11 Jan 2025.

The pressure-level Z contours are explicit, fixed case-reference envelopes,
NOT material PV edges. Independent isentropic PV-gradient edges are saved
separately. No fixed-area core is presented as a vortex boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "plot_packages"))
sys.path.insert(0, str(ROOT / "python_packages"))

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from PIL import Image

OUT = ROOT / "results" / "ERA5_daily_vortex_boundaries"
CACHE = OUT / "data" / "daily_vortex_diagnostics.npz"
START = np.datetime64("2024-12-28")
END = np.datetime64("2025-01-11")
DAYS = np.arange(START, END + np.timedelta64(1, "D"))
RADIUS = 6371000.0
G = 9.80665
OMEGA = 7.2921159e-5
KAPPA = 0.2854
P_LEVELS = [500, 300, 100, 50, 10, 3, 1]
THETA_LEVELS = [450, 650, 850, 1200, 1600]
PC = ccrs.PlateCarree()
PROJ = ccrs.NorthPolarStereo(central_longitude=0)
cartopy.config["data_dir"] = str(ROOT / "cartopy_data")


def read_file(path, wanted_vars, selected_pressures=None):
    """Read only the event times and a common 2.5-degree NH grid."""
    with h5py.File(path, "r") as f:
        time = f["valid_time"][:].astype("datetime64[s]")
        ii = np.flatnonzero((time >= START) & (time < END + np.timedelta64(1, "D")))
        if len(ii) == 0:
            return None
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        sy = round(2.5 / abs(float(lat[1] - lat[0])))
        sx = round(2.5 / abs(float(lon[1] - lon[0])))
        lat = lat[::sy]
        lon = lon[::sx]
        lon = ((lon + 180) % 360) - 180
        ix = np.argsort(lon)
        p = f["pressure_level"][:]
        sel = np.arange(len(p)) if selected_pressures is None else np.flatnonzero(np.isin(p, selected_pressures))
        data = {}
        for name in wanted_vars:
            if name not in f:
                continue
            # Basic contiguous time slices keep HDF5 reads efficient.
            a = f[name][ii[0]:ii[-1]+1, :, ::sy, ::sx][:, sel]
            data[name] = np.asarray(a[..., ix], dtype=np.float64)
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = {k: a[..., ::-1, :] for k, a in data.items()}
        return time[ii], p[sel], lat, lon[ix], data


def load_6hourly():
    sources = [
        (ROOT / "data" / "ERA5_pl_diagnostics_20241215_20241231.nc", [100, 10, 500]),
        (ROOT / "data" / "ERA5_pl_diagnostics_20250101_20250115.nc", [100, 10, 500]),
        (ROOT / "data" / "ERA5_epflux_pv" / "ERA5_epflux_pv_aux_20241215_20241231.nc", None),
        (ROOT / "data" / "ERA5_epflux_pv" / "ERA5_epflux_pv_aux_20250101_20250115.nc", None),
        (ROOT / "data" / "ERA5_epflux_pv" / "ERA5_epflux_upper_1_2_3_5hPa_20241215_20241231.nc", None),
        (ROOT / "data" / "ERA5_epflux_pv" / "ERA5_epflux_upper_1_2_3_5hPa_20250101_20250115.nc", None),
        (ROOT / "data" / "ERA5_daily_vortex" / "ERA5_zuv_300hPa_20241228_20241231_6hourly_2p5deg.nc", None),
        (ROOT / "data" / "ERA5_daily_vortex" / "ERA5_zuv_300hPa_20250101_20250111_6hourly_2p5deg.nc", None),
    ]
    times = np.arange(START.astype("datetime64[h]"), (END + np.timedelta64(1, "D")).astype("datetime64[h]"), np.timedelta64(6, "h"))
    pressures = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 300, 500], dtype=float)
    arrays = {k: np.full((len(times), len(pressures), 29, 144), np.nan) for k in ["z", "t", "u", "v"]}
    counts = {k: np.zeros((len(times), len(pressures)), dtype=int) for k in arrays}
    used = []
    for path, psel in sources:
        if not path.exists():
            if "300hPa" in path.name:
                print(f"Pending 300-hPa input: {path.name}",flush=True)
                continue
            raise FileNotFoundError(f"Required input is not yet available: {path}")
        print(f"Reading {path.name}", flush=True)
        time, pp, lat, lon, data = read_file(path, arrays.keys(), psel)
        assert np.allclose(lat, np.arange(20, 90.01, 2.5))
        assert np.allclose(lon, np.arange(-180, 180, 2.5))
        tt = np.searchsorted(times, time)
        assert np.array_equal(times[tt], time)
        for j, pressure in enumerate(pp):
            jnew = np.flatnonzero(pressures == pressure)[0]
            for k, a in data.items():
                arrays[k][tt, jnew] = a[:, j]
                counts[k][tt, jnew] += 1
        used.append(str(path))
    for name in ["z", "u", "v"]:
        assert np.all(counts[name][:,pressures!=300] == 1), (name, counts[name].min(), counts[name].max())
    assert np.all(counts["t"][:, pressures <= 150] == 1)
    return times, pressures, lat, lon, arrays, used


def periodic_derivative(a, step):
    return (np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1)) / (2 * step)


def ertel_pv(p, lat, data):
    """Hydrostatic Ertel PV on pressure surfaces, before daily averaging.

    Full horizontal-tilting terms retained. Pressure is in Pa. Pole values
    are replaced by the zonal mean of the adjacent resolved latitude.
    """
    phi = np.deg2rad(lat)
    cos = np.cos(phi)[None, None, :, None]
    cp = np.maximum(cos, 1e-6)
    dlambda = np.deg2rad(2.5)
    u, v, t = (data[k] for k in ["u", "v", "t"])
    theta = t * (100000.0 / (p[None, :, None, None] * 100)) ** KAPPA
    theta_p = np.gradient(theta, p * 100, axis=1, edge_order=2)
    up = np.gradient(u, p * 100, axis=1, edge_order=2)
    vp = np.gradient(v, p * 100, axis=1, edge_order=2)
    ty = np.gradient(theta, phi, axis=2, edge_order=2) / RADIUS
    tx = periodic_derivative(theta, dlambda) / (RADIUS * cp)
    zeta = (periodic_derivative(v, dlambda) - np.gradient(u * cos, phi, axis=2, edge_order=2)) / (RADIUS * cp)
    q = -G * ((zeta + 2 * OMEGA * np.sin(phi)[None, None, :, None]) * theta_p - vp * tx + up * ty) * 1e6
    q[:, :, -1, :] = np.mean(q[:, :, -2, :], axis=-1)[..., None]
    return theta, q


def interpolate_theta(theta, p, fields, levels):
    """Linear theta interpolation, no extrapolation; stable brackets only."""
    shape = (theta.shape[0], len(levels), *theta.shape[2:])
    result = {k: np.full(shape, np.nan) for k in fields}
    result["p"] = np.full(shape, np.nan)
    for it, target in enumerate(levels):
        # p is ascending: first bracket is the highest-altitude crossing.
        # In the stratosphere used here the selected theta profiles are monotone.
        assigned = np.zeros((theta.shape[0], *theta.shape[2:]), dtype=bool)
        for j in range(len(p) - 1):
            high, low = theta[:, j], theta[:, j + 1]
            good = (~assigned) & (high >= target) & (low <= target) & (high > low)
            frac = np.divide(target - low, high - low, out=np.zeros_like(high), where=high > low)
            for k, a in fields.items():
                vals = a[:, j + 1] * (1 - frac) + a[:, j] * frac
                result[k][:, it][good] = vals[good]
            pv = np.exp(np.log(p[j+1]) * (1-frac) + np.log(p[j]) * frac)
            result["p"][:, it][good] = pv[good]
            assigned |= good
    return result


def cell_weights(lat, nlon=144):
    south = np.maximum(-90, lat - 1.25)
    north = np.minimum(90, lat + 1.25)
    row = (np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))) * (2*np.pi/nlon)
    return np.broadcast_to(row[:, None], (len(lat), nlon)).copy()


def smooth_map(a, sigma=0.55):
    # Only light smoothing for plotted boundaries / diagnostic PV curve.
    valid = np.isfinite(a)
    numer = gaussian_filter(np.where(valid, a, 0), sigma, mode=["nearest", "wrap"])
    denom = gaussian_filter(valid.astype(float), sigma, mode=["nearest", "wrap"])
    return np.divide(numer, denom, out=np.full_like(a, np.nan), where=denom > .95)


def pv_edge(q, speed, weights, lat, jet_halfwidth=10):
    """Documented Nash-inspired PV-gradient/jet-constrained operational edge.

    Area-equivalent latitudes are formed by descending PV rank. A positive
    PV-gradient maximum is selected within 10 degrees of the contour-band
    maximum wind speed, in 40-80 N. This is an explicitly documented variant,
    not a claim to reproduce every numerical step of Nash et al. (1996).
    """
    good = np.isfinite(q) & np.isfinite(speed)
    order = np.argsort(q[good])[::-1]
    qq, ww, ss = q[good][order], weights[good][order], speed[good][order]
    area = np.cumsum(ww) - .5 * ww
    eqlat = np.rad2deg(np.arcsin(np.clip(1 - area/(2*np.pi), -1, 1)))
    grid = np.arange(30, 85.01, .5)
    qc = np.interp(grid, eqlat[::-1], qq[::-1])
    wc = np.full(grid.shape, np.nan)
    for j, center in enumerate(grid):
        mask = np.abs(eqlat - center) <= 1.5
        if mask.any():
            wc[j] = np.sum(ww[mask]*ss[mask]) / np.sum(ww[mask])
    if not np.isfinite(wc).any():
        return np.nan, np.nan, np.nan, False
    wc = np.interp(grid, grid[np.isfinite(wc)], wc[np.isfinite(wc)])
    qc = gaussian_filter1d(qc, 2.0, mode="nearest")
    wc = gaussian_filter1d(wc, 2.0, mode="nearest")
    grad = np.gradient(qc, np.deg2rad(grid))
    band = (grid >= 40) & (grid <= 80)
    jet = np.flatnonzero(band)[np.argmax(wc[band])]
    candidate = band & (np.abs(grid-grid[jet]) <= jet_halfwidth)
    peaks, _ = find_peaks(grad)
    peaks = peaks[candidate[peaks] & (grad[peaks] > 0)]
    idx = peaks[np.argmax(grad[peaks])] if len(peaks) else np.flatnonzero(candidate)[np.argmax(grad[candidate])]
    threshold = float(qc[idx])
    # Report whether the available domain truncates any high-PV region.
    touches_south = bool(np.any(q[0] >= threshold))
    return threshold, float(grid[idx]), float(wc[idx]), touches_south


def region_stats(mask, lat, lon, weights, z=None):
    """Spherical area centroid of all enclosed lobes; not the coldest point."""
    weight = np.where(mask, weights, 0.0)
    total = weight.sum()
    if total <= 0:
        return [np.nan] * 5
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    xyz = np.array([(weight*np.cos(phi)*np.cos(lam)).sum(),
                    (weight*np.cos(phi)*np.sin(lam)).sum(),
                    (weight*np.sin(phi)*np.ones_like(lam)).sum()]) / total
    clon = float(np.rad2deg(np.arctan2(xyz[1], xyz[0])))
    clat = float(np.rad2deg(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1]))))
    south = float(np.min(np.broadcast_to(lat[:,None], mask.shape)[mask]))
    sector = (lon >= -130) & (lon <= -60)
    sm = mask[:, sector]
    sector_south = float(np.min(np.broadcast_to(lat[:,None], sm.shape)[sm])) if sm.any() else np.nan
    return [clat, clon, float(total*RADIUS**2/1e12), south, sector_south]


def build_cache():
    times, p, lat, lon, a, sources = load_6hourly()
    # Twelve 1-150 hPa surfaces are used for PV, excluding tropospheric gaps.
    pick = p <= 150
    theta, q = ertel_pv(p[pick], lat, {k:v[:,pick] for k,v in a.items()})
    iso = interpolate_theta(theta, p[pick], {
        "pv": q, "z": a["z"][:,pick] / G,
        "u": a["u"][:,pick], "v": a["v"][:,pick],
    }, THETA_LEVELS)
    daily = {k: np.nanmean(v.reshape(15,4,*v.shape[1:]), axis=1) for k,v in a.items()}
    daily["z"] /= G
    iso_daily = {k: np.nanmean(v.reshape(15,4,*v.shape[1:]), axis=1) for k,v in iso.items()}
    nvalid = {k: np.sum(np.isfinite(v).reshape(15,4,*v.shape[1:]), axis=1) for k,v in iso.items()}
    # Daily mean PV edges are based on mean instantaneous-PV interpolations,
    # not on PV calculated from time-averaged winds/temperatures.
    for k in iso_daily:
        iso_daily[k][nvalid[k] < 4] = np.nan
    weights = cell_weights(lat)
    qlevels = np.full((15,len(THETA_LEVELS)), np.nan)
    eqlats = qlevels.copy()
    edgewind = qlevels.copy()
    southflags = np.zeros(qlevels.shape, bool)
    for d in range(15):
        for j in range(len(THETA_LEVELS)):
            qs = smooth_map(iso_daily["pv"][d,j])
            speed = np.hypot(iso_daily["u"][d,j],iso_daily["v"][d,j])
            qlevels[d,j], eqlats[d,j], edgewind[d,j], southflags[d,j] = pv_edge(qs, speed, weights, lat)
    baseline_q = np.nanmean(iso_daily["pv"][:3], axis=0)
    baseline_u = np.nanmean(iso_daily["u"][:3], axis=0)
    baseline_v = np.nanmean(iso_daily["v"][:3], axis=0)
    baseline_edges = np.array([pv_edge(smooth_map(baseline_q[j]), np.hypot(baseline_u[j],baseline_v[j]), weights,lat)[0] for j in range(len(THETA_LEVELS))])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, time=DAYS.astype(str), pressure=p, lat=lat,lon=lon,
                        z=daily["z"],u=daily["u"],v=daily["v"],
                        theta=np.array(THETA_LEVELS),
                        pv=iso_daily["pv"],iso_z=iso_daily["z"],iso_p=iso_daily["p"],
                        iso_u=iso_daily["u"],iso_v=iso_daily["v"],
                        pv_edge=qlevels,edge_equivalent_latitude=eqlats,edge_wind=edgewind,
                        pv_edge_touches_south=southflags,baseline_pv=baseline_q,
                        baseline_pv_edge=baseline_edges,iso_sample_count=nvalid["pv"],
                        sources=np.array(sources),sixhourly_samples_per_day=np.full(15,4))
    print(f"Cached diagnostics: {CACHE}",flush=True)


def lon_labels(ax):
    # Geographic orientation matches the existing temperature/wind series.
    for text,x,y,rotation in [
        ("180°",.5,1.085,0), ("120°W",.010,.790,55),
        ("60°W",.000,.215,-55), ("0°",.5,-.065,0),
        ("60°E",1.000,.215,55), ("120°E",.990,.790,-55),
    ]:
        ax.text(x,y,text,transform=ax.transAxes,ha="center",va="center",rotation=rotation,
                fontsize=10,color=".25",clip_on=False,zorder=12)


def map_frame(ax, first):
    ax.set_extent([-180,180,20,90],crs=PC)
    angle=np.linspace(0,2*np.pi,241)
    ax.set_boundary(MplPath(np.c_[np.sin(angle),np.cos(angle)]*.5+.5), transform=ax.transAxes)
    ax.coastlines(resolution="110m",linewidth=.45,color=".25",zorder=4)
    meridians=ax.gridlines(crs=PC,xlocs=np.arange(-180,181,60),ylocs=[20,40,60,80],
                          linewidth=.5,color=".28",alpha=.65,linestyle="--",zorder=5)
    meridians.ylines=False
    parallels=ax.gridlines(crs=PC,xlocs=np.arange(-180,181,60),ylocs=[20,40,60,80],
                          linewidth=.70,color="black",alpha=.68,linestyle="--",zorder=5)
    parallels.xlines=False
    if first:
        lon_labels(ax)
        for yy in [40,60]:
            ax.text(180,yy,f"{yy}°N",transform=PC,fontsize=10,color=".25",ha="center",va="center",zorder=12)
    ax.plot(0,90,marker="+",color=".25",markersize=4,markeredgewidth=.7,transform=PC,zorder=10)


def contour(ax, field, lat, lon, value, color, style="-", width=1.3, zorder=8):
    if not np.isfinite(value) or not (np.nanmin(field)<value<np.nanmax(field)):
        return
    cyc, ll = add_cyclic_point(field,coord=lon)
    return ax.contour(ll,lat,cyc,levels=[value],colors=[color],linewidths=width,
                      linestyles=style,transform=PC,zorder=zorder)


def nice_limits(fields, step):
    lo,hi=np.nanpercentile(fields,[1,99])
    return np.floor(lo/step)*step,np.ceil(hi/step)*step


def plot_maps(data, kind, j):
    lat,lon=data["lat"],data["lon"]
    weights=cell_weights(lat)
    if kind=="z":
        pressure=int(data["pressure"][j])
        fields=data["z"][:,j]
        pre=fields[:3].mean(axis=0)
        reference_lat=50 if pressure>=300 else 60
        # Fixed in time, calculated from the stated case baseline, not climatology.
        edge=float(pre[np.argmin(abs(lat-reference_lat))].mean())
        edges=np.full(15,edge)
        smooth=np.array([smooth_map(a,.4) for a in fields])
        pre_smooth=smooth_map(pre,.4)
        preedge=edge
        height=np.median(fields[:,lat>=50])/1000
        title=f"{pressure} hPa (~{height:.0f} km) | {'Tropospheric' if pressure>=300 else 'Stratospheric'} vortex"
        unit="km"
        displayed=fields/1000
        cmap="turbo"
        levels=np.linspace(*nice_limits(displayed,.2 if pressure>=50 else .5),19)
        filename=f"ERA5_{pressure:03d}hPa_daily_vortex_Zboundary_20241228_20250111"
        folder=OUT/"01_pressure_level_maps"
        mask=lambda d: smooth[d]<=edge
        type_name="fixed_case_Z_envelope"
    else:
        theta=int(data["theta"][j])
        fields=data["pv"][:,j]
        smooth=np.array([smooth_map(a) for a in fields])
        edges=data["pv_edge"][:,j]
        pre_smooth=smooth_map(data["baseline_pv"][j])
        preedge=data["baseline_pv_edge"][j]
        medp=float(np.nanmedian(data["iso_p"][:,j,lat>=50]))
        height=float(np.nanmedian(data["iso_z"][:,j,lat>=50]))/1000
        title=f"{theta} K | PV vortex edge (~{height:.0f} km, {medp:.0f} hPa)"
        unit="PVU"
        displayed=fields
        cmap="YlGnBu"
        vmax=float(np.ceil(np.nanpercentile(fields,99)/10)*10)
        levels=np.linspace(0,vmax,17)
        filename=f"ERA5_{theta:04d}K_daily_vortex_PVedge_20241228_20250111"
        folder=OUT/"02_isentropic_PV_boundaries"
        mask=lambda d: smooth[d]>=edges[d]
        type_name="isentropic_PV_gradient_jet_constrained"
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,
                         "axes.titleweight":"normal","xtick.direction":"in","ytick.direction":"in",
                         "pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(3,5,figsize=(20/2.54,15.4/2.54),subplot_kw={"projection":PROJ})
    fig.subplots_adjust(left=.035,right=.912,bottom=.065,top=.865,wspace=.065,hspace=.46)
    fig.suptitle(title,fontsize=16,fontweight="normal",y=.985)
    handles=[Line2D([0],[0],color="black",lw=1.5,label="Daily edge" if kind=="pv" else "Daily Z envelope"),
             Line2D([0],[0],color="#B318A0",lw=1.4,ls="--",label="Dec 28-30 reference"),
             Line2D([0],[0],marker="o",ls="none",color="black",ms=4,label="Centroid")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.946),ncol=3,
               frameon=False,fontsize=12,handlelength=1.8,columnspacing=1.0,handletextpad=.4)
    records=[]
    for d,ax in enumerate(axes.flat):
        map_frame(ax,d%5==0)
        cyc,ll=add_cyclic_point(np.clip(displayed[d],levels[0],levels[-1]),coord=lon)
        mesh=ax.contourf(ll,lat,cyc,levels=levels,cmap=cmap,extend="neither",transform=PC,zorder=1)
        # Same pre-storm shape on every panel; contour colour differs from black daily edge.
        contour(ax,pre_smooth,lat,lon,preedge,"#B318A0","--",1.25,zorder=7)
        contour(ax,smooth[d],lat,lon,edges[d],"black","-",1.4,zorder=8)
        region=mask(d)&np.isfinite(smooth[d])
        stats=region_stats(region,lat,lon,weights)
        if np.isfinite(stats[0]):
            ax.plot(stats[1],stats[0],"o",ms=3.2,color="black",transform=PC,zorder=11)
        date=str(DAYS[d]); month="Dec." if date[5:7]=="12" else "Jan."
        ax.set_title(f"{month} {int(date[-2:])}",y=-.25,fontsize=12,fontweight="normal",pad=0)
        records.append({"date_UTC":date,"diagnostic":type_name,
                        "level":pressure if kind=="z" else theta,
                        "level_unit":"hPa" if kind=="z" else "K",
                        "edge_value":float(edges[d]),"edge_unit":"gpm" if kind=="z" else "PVU",
                        "centroid_latitude":stats[0],"centroid_longitude":stats[1],
                        "area_million_km2":stats[2],"southernmost_grid_latitude":stats[3],
                        "southernmost_130W_60W_grid_latitude":stats[4],
                        "reference_latitude_Z_only":reference_lat if kind=="z" else "",
                        "edge_equivalent_latitude_PV_only":float(data["edge_equivalent_latitude"][d,j]) if kind=="pv" else "",
                        "touches_southern_domain":bool(region[0].any())})
    cax=fig.add_axes([.933,.205,.012,.57])
    cb=fig.colorbar(mesh,cax=cax)
    cb.locator=MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=10,direction="in",length=3,pad=2)
    cb.ax.set_title(unit,fontsize=12,pad=5)
    png=folder/f"{filename}.png"
    fig.savefig(png,dpi=600,facecolor="white")
    preview=OUT/"previews"/f"{filename}.jpg"
    preview.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(png) as im:
        assert abs(im.width-4724)<=1,(png,im.size)
        im.thumbnail((1600,1600))
        im.convert("RGB").save(preview,quality=93)
    plt.close(fig)
    print(f"Saved {png.name}",flush=True)
    return records


def write_metadata(data, records):
    with (OUT/"vortex_daily_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    metadata={"time_UTC":[str(d) for d in DAYS],"baseline_UTC":["2024-12-28","2024-12-30"],
              "samples_per_day":4,"original_maps_comparison":"same polar orientation, daily 3x5 layout, coastline and graticule",
              "grid_degrees":2.5,"domain":"20-90 N, all longitudes",
              "pressure_levels_hPa":P_LEVELS,"isentropic_levels_K":THETA_LEVELS,
              "generated_pressure_levels_hPa":sorted(set(int(r["level"]) for r in records if r["level_unit"]=="hPa")),
              "missing_pressure_levels_hPa":[int(pp) for pp in P_LEVELS if not np.isfinite(data["z"][:,np.flatnonzero(data["pressure"]==pp)[0]]).all()],
              "z_boundary":"Fixed pre-storm zonal-mean Z at 50 N for 300/500 hPa, 60 N for stratospheric levels; case-adapted contour proxy, NOT climatology or a material PV edge.",
              "pv_boundary":"Maximum positive d(PV)/d(equivalent latitude) among local maxima within +/-10 degrees of contour-band maximum mean wind speed in equivalent latitude 40-80 N. Rank-based area; 1.5-degree halfwidth wind bands; 1-degree Gaussian smoothing of PV-equivalent-latitude curve. Nash-inspired variant, not exact original algorithm.",
              "pv_calculation":"Hydrostatic Ertel PV with horizontal tilting terms on 1-150 hPa 6-hourly fields; interpolated onto theta before daily averaging; 4/4 samples required; no extrapolation.",
              "pv_polar_cap":"90 N PV approximated with zonal mean at 87.5 N; affects small cap, not edge latitudes",
              "plot_smoothing":"Boundary fields only: Gaussian sigma 0.4 grid cells for Z, 0.55 for PV; filled fields unsmoothed",
              "centroid":"Area-weighted spherical centroid of all thresholded lobes; not a temperature minimum, not Seviour weighted moment centroid.",
              "important_limitations":["Case-reference Z threshold is not a unique or objective transport edge; position/area depend on threshold.","Nash-inspired PV edges can be affected by grid, smoothing, finite pressure spacing and jet constraint; use as an independent diagnostic, not proof of material coherence.","No causal geomagnetic-storm or surface-weather attribution follows from these figures alone.","No 60-80 km maps are generated; 1 hPa near stratopause and theta levels have variable actual geometric heights."],
              "sources":data["sources"].tolist(),
              "references":[{"doi":"10.1175/BAMS-D-15-00212.1","use":"Distinguish tropospheric Z envelope from stratospheric PV vortex."},
                            {"doi":"10.1002/grl.50927","use":"Z contour approach; here explicitly adapted to case baseline, not original DJFM climatology."},
                            {"doi":"10.1029/96JD00066","use":"PV-gradient edge constrained by wind jet; explicit numerical variant documented."}],
              "checks":{"all_days_present":len(data["time"])==15,
                        "all_days_four_times":bool(np.all(data["sixhourly_samples_per_day"]==4)),
                        "PV_edge_touches_20N":bool(np.any(data["pv_edge_touches_south"])),
                        "finite_PV_edge_thresholds":bool(np.isfinite(data["pv_edge"]).all())}}
    (OUT/"method_and_QA.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache",action="store_true")
    parser.add_argument("--only",choices=["z","pv","all"],default="all")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.rebuild_cache or not CACHE.exists():
        build_cache()
    with np.load(CACHE,allow_pickle=False) as source:
        data={k:source[k] for k in source.files}
    records=[]
    if args.only in ["all","z"]:
        for pp in P_LEVELS:
            j=int(np.flatnonzero(data["pressure"]==pp)[0])
            if not np.isfinite(data["z"][:,j]).all():
                print(f"Skipping {pp} hPa: input not complete",flush=True)
                continue
            records.extend(plot_maps(data,"z",j))
    if args.only in ["all","pv"]:
        for j in range(len(THETA_LEVELS)):
            records.extend(plot_maps(data,"pv",j))
    write_metadata(data,records)
    print(f"Completed {len(records)//15} maps in {OUT}",flush=True)


if __name__=="__main__":
    main()
