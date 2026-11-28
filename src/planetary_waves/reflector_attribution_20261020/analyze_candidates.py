"""Audit two candidate impediment regions, without assuming storm causation.

Uses original MLS native harmonics and ERA5 fields. No fitted propagation
paths, no imposed time ordering, no control simulation and no data download
when the required source observations are already available locally.
"""
from pathlib import Path
import sys
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from build_acp2018_event_figures_20261010 import extract, flux, background
from test_background_turning_joint_20261009 import diff_native
from chain_test_20261015 import read_zonal

A, OM, G, R, CP, H = 6371000., 7.292115e-5, 9.80665, 287., 1004., 7000.
DAYS = pd.date_range("2024-12-27", "2025-01-10")
PRE = DAYS <= "2024-12-30"
# Fixed before this calculation from the user's described candidate heights.
REGIONS = {
    "yellow_cap": (40, 60, 60, 70),
    "green_cap": (60, 80, 80, 90),
    "yellow_response": (40, 60, 53, 64),
    "green_response": (60, 80, 55, 88),
}
PERIODS = {"pre": ("2024-12-27", "2024-12-30"),
           "early": ("2025-01-01", "2025-01-02"),
           "requested_post": ("2025-01-03", "2025-01-06"),
           "late": ("2025-01-07", "2025-01-09")}
SOURCES = {}


def source(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def anomaly(a):
    return a - np.mean(a[PRE], axis=0)


def fields(z0, t0, lat, p, points=5):
    """Identical qy/n2 operator to the arrow figure; expose additive terms."""
    phi = np.deg2rad(lat)
    cosine = np.cos(phi)[:, None]
    f = 2*OM*np.sin(phi)[:, None]
    cot = 1/np.tan(phi)[:, None]
    zstar = H*np.log(1000/p)
    dy = lambda v, n: diff_native(v, phi, n, points, 0)
    dz = lambda v, n: diff_native(v, zstar, n, points, -1)
    zp, zpp, zppp = [dy(z0, n) for n in (1, 2, 3)]
    u = -G*zp/(A*f)
    up = -G*(zpp-cot*zp)/(A*f)
    upp = -G*(zppp-2*cot*zpp+(2*cot*cot+1)*zp)/(A*f)
    uz, uzz = -G*dz(zp, 1)/(A*f), -G*dz(zp, 2)/(A*f)
    tz, tzz = dz(t0, 1), dz(t0, 2)
    ns = R/H*(tz+R/CP*t0/H)
    nsz = R/H*(tzz+R/CP*tz/H)
    beta = np.broadcast_to(2*OM*cosine/A, z0.shape)
    qm = -(upp-np.tan(phi)[:, None]*up-u/cosine**2)/A**2
    qv = -f*f/ns*(uzz-(1/H+nsz/ns)*uz)
    qy = beta+qm+qv
    terms = dict(beta_over_U=beta/u, meridional_over_U=qm/u,
                 vertical_over_U=qv/u,
                 stratification_term=-f*f/(4*ns*H*H))
    index = sum(terms.values()) - 1/(A*cosine)**2
    good = (ns > 0) & (abs(u) >= 5) & np.isfinite(index)
    grad = dy(t0, 1)/A
    return dict(U=u, Uz=uz*1000, T=t0, T_gradient=grad*1e6,
                thermal_shear=-R/(H*f)*grad*1000, N2=ns,
                qy=qy, qmer=qm, qvert=qv, beta=beta,
                n2=np.where(good, index, np.nan), **terms)


def weighted(values, mask, lat):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)[mask]
    return np.average(values[..., mask], weights=w, axis=-1)


def region_mask(lat, height, limits):
    lo, hi, bot, top = limits
    return ((lat[:, None] >= lo) & (lat[:, None] < hi)
            & (height >= bot) & (height <= top))


def mls():
    d = dict(np.load(source(ROOT / "W1_three_bands_Dec27_Jan20_20261016/MLS_W123_fullheight_Dec27_Jan20.npz")))
    ids = pd.DatetimeIndex(d["dates"]).get_indexer(DAYS)
    assert (ids >= 0).all()
    lat, p = d["lat"], d["pressure"]
    Z, T, z0, t0 = [v[ids] for v in extract(d)]
    # A fixed display altitude prevents daily motion of the sample selection.
    heights = np.mean(z0[PRE], axis=0)/1000
    cosine = np.cos(np.deg2rad(lat))[:, None]
    full = {}
    settings = [("pair_p5", -1, 0, 5), ("pair_p7", -1, 0, 7),
                ("LT1_p5", 1, 0, 5), ("LT2_p5", 2, 0, 5)]
    settings += [(f"delete{i:02}_p5", -1, i, 5) for i in range(1, 13)]
    rows, supports, cellrows = [], [], []
    for name, branch, sector, points in settings:
        cz, ct, zz, tt = [v[ids] for v in extract(d, branch, sector)]
        raw = [fields(zz[i], tt[i], lat, p, points) for i in range(len(DAYS))]
        fs = {k: np.stack([x[k] for x in raw]) for k in raw[0]}
        fl = [flux(cz[i, ..., 0], ct[i, ..., 0], zz[i], tt[i], lat, p) for i in range(len(DAYS))]
        fs["Fz"] = np.stack([f["W"]*cosine for f in fl])
        fs["Q"] = np.stack([f["Q"] for f in fl])
        fs["amplitude"] = abs(cz[..., 0])
        full[name] = fs
        if name == "pair_p5":
            for i in range(len(DAYS)):
                old = background(zz[i], tt[i], lat, p)
                for k in ["n2", "qy", "Uz", "N2"]:
                    np.testing.assert_allclose(fs[k][i], old[k], equal_nan=True, rtol=1e-12, atol=1e-20)
        for region, limits in REGIONS.items():
            geom = region_mask(lat, heights, limits)
            # Keep near-zero-U cells in the fixed geometry, but mark their
            # refractive index unknown. They are NOT counted as propagating.
            fixed = geom.copy()
            for k in ["qy", "U", "Uz", "N2"]:
                fixed &= np.isfinite(fs[k]).all(0)
            fixed &= (fs["N2"] > 0).all(0)
            fluxmask = geom & np.isfinite(fs["Fz"]).all(0)
            indexmask = fixed & np.isfinite(fs["n2"]).all(0)
            n = int(fixed.sum())
            il, ip = np.where(fixed)
            supports.append(dict(setting=name, region=region,
                requested_cells=int(geom.sum()), fixed_cells=n,
                always_valid_n2_cells=int(indexmask.sum()), flux_cells=int(fluxmask.sum()),
                latitudes=";".join(map(str, np.unique(lat[il]))),
                pressure_hPa=";".join(map(str, np.unique(p[ip]))),
                min_height=float(heights[fixed].min()) if n else np.nan,
                max_height=float(heights[fixed].max()) if n else np.nan))
            if not n:
                continue
            existed = (fs["n2"][PRE, :, :][:, fixed] < 0).any(0)
            preknown = np.isfinite(fs["n2"][PRE, :, :][:, fixed]).all(0)
            for i, day in enumerate(DAYS):
                known = np.isfinite(fs["n2"][i][fixed])
                neg = fs["n2"][i][fixed] < 0
                row = dict(date=str(day.date()), setting=name, region=region, cells=n,
                    negative_percent=100*neg.mean(),
                    new_negative_percent=100*(neg & ~existed & preknown).mean(),
                    previously_negative_percent=100*existed.mean(),
                    unknown_index_percent=100*np.mean(~known),
                    unknown_baseline_percent=100*np.mean(~preknown),
                    flux_cells=int(fluxmask.sum()),
                    downward_percent=100*np.mean(fs["Fz"][i][fluxmask] < 0))
                for k in fs:
                    v = fs[k][i]
                    mask = fluxmask if k in ["Fz","Q","amplitude"] else fixed
                    if np.isfinite(v[mask]).all():
                        row[k] = float(weighted(v, mask, lat))
                rows.append(row)
            if name == "pair_p5":
                for l, j in zip(il, ip):
                    for i, day in enumerate(DAYS):
                        cellrows.append(dict(date=str(day.date()), region=region,
                            latitude=lat[l], pressure_hPa=p[j], height_km=heights[l, j],
                            n2=fs["n2"][i,l,j], qy=fs["qy"][i,l,j],
                            Uz=fs["Uz"][i,l,j], Fz=fs["Fz"][i,l,j]))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "candidate_regions_daily.csv", index=False)
    pd.DataFrame(supports).to_csv(OUT / "candidate_support.csv", index=False)
    pd.DataFrame(cellrows).to_csv(OUT / "candidate_native_cells.csv", index=False)
    np.savez_compressed(OUT / "MLS_native_fields.npz", dates=DAYS.values, lat=lat,
                        pressure=p, height_km=heights, **full["pair_p5"])
    # Compare derivative choices on precisely the same support.
    robust = []
    for region, limits in REGIONS.items():
        base = region_mask(lat, heights, limits)
        for names in [("pair_p5", "pair_p7"), ("pair_p5", "LT1_p5", "LT2_p5")]:
            common = base.copy()
            for name in names:
                common &= np.isfinite(full[name]["qy"]).all(0)
                common &= (full[name]["N2"] > 0).all(0)
            for name in names:
                for i, day in enumerate(DAYS):
                    robust.append(dict(region=region, comparison="+".join(names),
                        setting=name, date=str(day.date()), cells=int(common.sum()),
                        negative_percent=100*np.mean(full[name]["n2"][i][common] < 0) if common.any() else np.nan,
                        unknown_index_percent=100*np.mean(~np.isfinite(full[name]["n2"][i][common])) if common.any() else np.nan))
    pd.DataFrame(robust).to_csv(OUT / "common_support_sensitivity.csv", index=False)
    return d, ids, lat, p, full, heights


def thermal_crosscheck(d, ids, lat, p, mls_fields, heights):
    """MLS T-based thermal wind vs ERA5 U, collocated at identical p/lat.

    ERA5 native daily shear is linearly interpolated in log-pressure only
    inside valid stencils; exact native latitude matches, no extrapolation.
    Reanalysis assimilates satellite information: this is cross-product
    consistency, not a statistically independent causal test.
    """
    rawpath = source(ROOT / "zonal_wind_diagnostics_20261005/data/ERA5_ml_1_50_uvt_20241227_20250110_2p5deg.nc")
    with h5py.File(rawpath) as h:
        times = pd.to_datetime(h["valid_time"][:], unit="s")
        la, lev = h["latitude"][:].astype(float), h["model_level"][:].astype(int)
        u, t = read_zonal(h, "u"), read_zonal(h, "t")
    order = np.argsort(la)
    la, u, t = la[order], u[..., order], t[..., order]
    u, t = [np.stack([a[times.normalize() == day].mean(0) for day in DAYS]) for a in (u, t)]
    defs = pd.read_csv(source(ROOT / "zonal_wind_diagnostics_20261005/ERA5_L137_definitions.csv")).set_index("level")
    ep = ((defs.a_Pa+defs.a_Pa.shift(1))/200).loc[lev].to_numpy()
    assert (defs.loc[lev, "b"] == 0).all()
    phi = np.deg2rad(la)
    rows = []
    cz, ct, z0, t0 = [v[ids] for v in extract(d)]
    for points in (3, 5):
        shear = -diff_native(u, np.log(ep), 1, points, 1)/(H/1000)
        egrad = diff_native(t, A*phi, 1, points, 2)
        eth = -R/(H/1000)*egrad/(2*OM*np.sin(phi)[None,None,:])
        mg = diff_native(t0, A*np.deg2rad(lat), 1, points, 1)
        mth = -R/(H/1000)*mg/(2*OM*np.sin(np.deg2rad(lat))[None,:,None])
        eheight = defs.loc[lev, "geometric_m"].to_numpy()/1000
        jlat = np.array([np.flatnonzero(np.isclose(la, l))[0] for l in lat])
        for j in np.flatnonzero((np.nanmean(heights,0) >= 55) & (np.nanmean(heights,0) <= 85)):
            # Explicitly restrict brackets to centered-derivative support.
            half = points//2
            inside = ep[half] <= p[j] <= ep[-half-1]
            for lo, hi in [(40, 60), (60, 80)]:
                lm = (lat >= lo) & (lat < hi)
                # Identical latitude centers for all compared products and
                # days; record the usable subset instead of averaging NaNs.
                lm &= np.isfinite(mth[:,:,j]).all(0)
                if not lm.any():
                    continue
                w = np.cos(np.deg2rad(lat[lm]))
                if inside:
                    actual = interp1d(np.log(ep), shear[:,:,jlat][:,:,lm], axis=1,
                                      bounds_error=False, fill_value=np.nan)(np.log(p[j]))
                    predicted = interp1d(np.log(ep), eth[:,:,jlat][:,:,lm], axis=1,
                                         bounds_error=False, fill_value=np.nan)(np.log(p[j]))
                    actual, predicted = [np.average(v, axis=-1, weights=w) for v in (actual,predicted)]
                else:
                    actual = predicted = np.full(len(DAYS), np.nan)
                mgmean = np.average(mg[:,lm,j],axis=-1,weights=w)*1e6
                mtmean = np.average(mth[:,lm,j],axis=-1,weights=w)
                mh = np.average(heights[lm,j],weights=w)
                eh = float(np.interp(np.log(p[j]),np.log(ep),eheight)) if ep.min() <= p[j] <= ep.max() else np.nan
                for i, day in enumerate(DAYS):
                    rows.append(dict(date=str(day.date()), band=f"{lo}-{hi}", points=points,
                        latitudes=";".join(map(str,lat[lm])), latitude_cells=int(lm.sum()),
                        pressure_hPa=p[j], MLS_reference_height_km=mh, ERA5_standard_height_km=eh,
                        ERA5_U_shear=actual[i], ERA5_T_shear=predicted[i],
                        MLS_T_shear=mtmean[i], MLS_T_gradient=mgmean[i],
                        delta_ERA5_U_shear=anomaly(actual)[i],
                        delta_ERA5_T_shear=anomaly(predicted)[i],
                        delta_MLS_T_shear=anomaly(mtmean)[i],
                        delta_MLS_T_gradient=anomaly(mgmean)[i],
                        valid_ERA5_stencil=inside))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "thermal_wind_cross_product.csv", index=False)
    summary = []
    for (band, pp, pts), sub in tab.groupby(["band","pressure_hPa","points"]):
        for period, (start,end) in PERIODS.items():
            use = sub[sub.date.between(start,end)]
            row = dict(band=band, pressure_hPa=pp, points=int(pts), period=period,
                       MLS_reference_height_km=use.MLS_reference_height_km.mean())
            for k in ["ERA5_U_shear","ERA5_T_shear","MLS_T_shear","MLS_T_gradient",
                      "delta_ERA5_U_shear","delta_ERA5_T_shear","delta_MLS_T_shear","delta_MLS_T_gradient"]:
                row[k] = use[k].mean()
            summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "thermal_wind_periods.csv", index=False)


def composition():
    comp = pd.read_csv(source(ROOT / "storm_two_path_test_20261020/MLS_night_O3_T_reference.csv"))
    rows = []
    for key, (lo, hi, bot, top) in REGIONS.items():
        if not key.endswith("cap"):
            continue
        band = f"{lo}-{hi}"
        sub = comp[comp.band == band]
        href = sub[sub.date <= "2024-12-30"].groupby("pressure_hPa").height_km.mean()
        pp = href.index[(href >= bot) & (href <= top)]
        # Native-layer average, not mass-weighted or geometric volume average.
        for day in DAYS:
            block = sub[(sub.date == str(day.date())) & sub.pressure_hPa.isin(pp)]
            good = len(block) == len(pp) and len(pp) > 0
            rows.append(dict(date=str(day.date()), region=key, layers=len(pp),
                height_min=href.loc[pp].min(), height_max=href.loc[pp].max(),
                O3_percent=block.ozone_ppmv_percent.mean() if good else np.nan,
                T_anomaly_K=block.temperature_K_anomaly.mean() if good else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "candidate_composition.csv",index=False)


def summarize():
    d = pd.read_csv(OUT / "candidate_regions_daily.csv")
    rows = []
    for (reg, setting), sub in d.groupby(["region", "setting"]):
        pre = sub[sub.date <= "2024-12-30"]
        for period,(start,end) in PERIODS.items():
            block = sub[sub.date.between(start,end)]
            row = dict(region=reg,setting=setting,period=period,cells=block.cells.iloc[0])
            for k in ["negative_percent","new_negative_percent","previously_negative_percent","unknown_index_percent",
                      "Fz","T","T_gradient","Uz","qy","qmer","qvert",
                      "beta_over_U","meridional_over_U","vertical_over_U","stratification_term"]:
                row[k] = block[k].mean()
                row["delta_"+k] = block[k].mean()-pre[k].mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "candidate_periods.csv", index=False)
    timeline = pd.read_csv(source(ROOT / "linked_mechanism_followup_20261020/linked_daily.csv"))
    timeline.to_csv(OUT / "forcing_context_daily.csv",index=False)
    c = pd.read_csv(OUT / "candidate_composition.csv")
    onset = []
    for reg in ["yellow_cap","green_cap"]:
        s = d[(d.region==reg)&(d.setting=="pair_p5")].set_index("date").reindex(DAYS.strftime("%Y-%m-%d"))
        co = c[c.region==reg].set_index("date")
        s = s.join(co[["O3_percent","T_anomaly_K"]])
        for k,direction in [("O3_percent","lower"),("T_anomaly_K","higher"),
                             ("Uz","lower"),("negative_percent","higher")]:
            pre = s.loc[:"2024-12-30",k]
            threshold = pre.min() if direction=="lower" else pre.max()
            hits = s[k]<threshold if direction=="lower" else s[k]>threshold
            # Descriptive two-day run; not a significance or causal-change test.
            consecutive = hits & hits.shift(-1,fill_value=False)
            onset.append(dict(region=reg,variable=k,direction=direction,
                first_two_day_start=next(iter(s.index[consecutive]),"not detected") if pre.notna().all() else "insufficient support",
                baseline_range_min=pre.min(),baseline_range_max=pre.max()))
    pd.DataFrame(onset).to_csv(OUT/"descriptive_onsets.csv",index=False)
    checks = dict(window=[str(DAYS[0].date()),str(DAYS[-1].date())],
        baseline=["2024-12-27","2024-12-30"],candidate_regions=REGIONS,
        units=dict(Uz="m/s/km in z-star=-7km ln(p/p0)",qy="1/(m s)",n2="1/m^2",Fz="reduced EP m^2/s^2"),
        exact_original_qy_n2_Uz_N2_operator_verified=True,
        fixed_support="all-day finite qy/U/Uz with stable N2; n2 invalid at weak U remains explicitly unknown; flux uses separate all-day valid full-ROI support",
        previously_negative="at least one of the four baseline daily n2 values was negative",
        new_negative="daily n2<0 on cells with all four baseline values valid and none negative",
        fraction_type="unweighted native-cell count, NOT geometric area; adjacent cells are correlated",
        thermal_comparison="exact latitude matching and bounded log-pressure interpolation of ERA5 native derivatives onto MLS levels",
        missing_or_critical_wind_cells_not_filled=True,PDF_generated=False,
        interpretation="candidate impediments, not identified reflecting surfaces; no storm causal effect estimate",
        source_sha256=SOURCES)
    for path, digest in SOURCES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (OUT/"methods_and_checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf8")
    print(pd.DataFrame(rows).query("setting=='pair_p5'")[
        ["region","period","cells","negative_percent","new_negative_percent","delta_Uz","Fz"]].to_string(index=False))
    print(pd.DataFrame(onset).to_string(index=False))


if __name__=="__main__":
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    OUT.mkdir(exist_ok=True)
    args = mls()
    thermal_crosscheck(*args)
    composition()
    summarize()
