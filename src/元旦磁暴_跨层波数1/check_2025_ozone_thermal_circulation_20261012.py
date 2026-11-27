"""Matched observations and conditional chemistry test; not storm attribution.

Strict Dec 27--30 baseline. All computations use existing data, no download,
time smoothing, imputation, or PDF. Chemical equilibrium is a hypothesis,
not an assumption used to retrieve unobserved O/H or external forcing.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import warnings

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from threadpoolctl import threadpool_limits
import build_comprehensive_diagnostic_figure as base

threadpool_limits(1)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / '2025_ozone_thermal_circulation_20261012'
DAYS = pd.date_range('2024-12-27', '2025-01-10')
BANDS = [(20, 40), (40, 60), (60, 80), (30, 40), (40, 50)]
CASES = ['reference', 'W1_only', 'coslat_only'] + [f'delete_{i}' for i in range(12)]
CHEM = ['H_O2_M', 'H_O3', 'O_O3', 'O_OH', 'O_HO2', 'O_O_M', 'O_O2_M']


def factor(T):
    # Fixed isobar, fixed O2 mixing ratio and fixed O/H mixing-ratio ratio.
    # k_formation ~ T^-2.4; k_H+O3 ~ exp(-470/T); air density ~ p/T.
    # Reference 200 K only stabilizes numerical scaling, not event baseline.
    return (T / 200.) ** -3.4 * np.exp(470. * (1. / T - 1. / 200.))


def fit_frame(a, names, lo, hi, case, branch_offset=False):
    a = a[a.lat.between(lo, hi, inclusive='left')].copy()
    a = a[np.isfinite(a[['lat', 'lon'] + names]).all(1)]
    if case.startswith('branch_'):
        a = a[a.branch.astype(str) == case[7:]]
    if case.startswith('delete_'):
        a = a[(a.lon % 360 // 30).astype(int) != int(case[7:])]
    rec = dict(n=len(a), sectors=0, condition=np.nan)
    rec.update({k: np.nan for k in names})
    if not len(a):
        return rec
    lam = np.deg2rad(a.lon.to_numpy() % 360)
    la = (a.lat.to_numpy() - (lo + hi) / 2.) / 10.
    cols = [np.ones(len(a)), la, la * la]
    for k in ([1] if case == 'W1_only' else [1, 2, 3]):
        cols.extend([np.cos(k * lam), np.sin(k * lam)])
    if branch_offset and 'branch' in a:
        branches = np.unique(a.branch)
        if len(branches) == 2:
            cols.append(np.where(a.branch.to_numpy() == branches[0], -1., 1.))
    x = np.column_stack(cols)
    sectors = (a.lon.to_numpy() % 360 // 30).astype(int)
    _, inv, count = np.unique(sectors, return_inverse=True, return_counts=True)
    rec['sectors'] = len(count)
    if len(a) < (40 if hi - lo >= 20 else 25) or len(count) < 8:
        return rec
    rec['latitude_min'] = float(a.lat.min())
    rec['latitude_max'] = float(a.lat.max())
    # Do not evaluate a full SABER latitude interval outside its sampled span.
    if branch_offset and (a.lat.min() > lo + 2. or a.lat.max() < hi - 2.):
        return rec
    w = np.cos(np.deg2rad(a.lat.to_numpy()))
    if case != 'coslat_only':
        w = w / count[inv]
    xw = x * np.sqrt(w)[:, None]
    rec['condition'] = float(np.linalg.cond(xw))
    if rec['condition'] > 30:
        return rec
    beta = np.linalg.lstsq(xw, a[names].to_numpy() * np.sqrt(w)[:, None], rcond=None)[0]
    target = np.linspace(lo, hi, 1001)
    wt = np.cos(np.deg2rad(target))
    ll = (target - (lo + hi) / 2.) / 10.
    state = beta[0] + np.average(ll, weights=wt) * beta[1] + np.average(ll * ll, weights=wt) * beta[2]
    rec.update(dict(zip(names, map(float, state))))
    for col, name in enumerate(names):
        rec[name + '_W1'] = float(np.hypot(beta[3, col], beta[4, col]))
    rec['sample_latitude'] = float(np.average(a.lat, weights=w))
    if 'lst' in a:
        rec['sample_lst'] = float(np.average(a.lst, weights=w))
    return rec


def read_mls(path, product):
    with h5py.File(path) as f:
        s = f[f'HDFEOS/SWATHS/{product}']
        a = {k: s['Data Fields/' + k][:] for k in ['L2gpValue', 'L2gpPrecision', 'Quality', 'Convergence', 'Status']}
        a.update({k: s['Geolocation Fields/' + k][:] for k in ['Pressure', 'Latitude', 'Longitude', 'LocalSolarTime', 'SolarZenithAngle', 'Time']})
    return a


def rebase(d, group_keys, variables):
    rows = []
    for _, g in d.groupby(group_keys, dropna=False):
        g = g.sort_values('date').copy()
        pre = g[g.date <= '2024-12-30']
        for k in variables:
            b = pre[k].mean() if len(pre) == 4 and pre[k].notna().all() else np.nan
            g[k + '_baseline'] = b
            g[k + '_anomaly'] = g[k] - b
            g[k + '_percent'] = 100. * (g[k] / b - 1.) if np.isfinite(b) and (pre[k] > .05).all() else np.nan
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def mls():
    tfiles, gfiles = base.mls_files('Temperature'), base.mls_files('GPH')
    opaths = list((ROOT / 'event_evidence_20261002/data/O3').glob('*.he5'))
    ofiles = {pd.to_datetime(p.stem.rsplit('_', 1)[-1], format='%Yd%j'): p for p in sorted(opaths)}
    rows, hashes = [], {}
    for day in DAYS:
        paths = [ofiles[day], tfiles[day], gfiles[day]]
        for p in paths:
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        o, t, z = [read_mls(p, k) for p, k in zip(paths, ['O3', 'Temperature', 'GPH'])]
        for k in ['Pressure', 'Latitude', 'Longitude', 'Time', 'LocalSolarTime']:
            assert np.array_equal(o[k], t[k]) and np.array_equal(o[k], z[k]), (day, k)
        quality = ((o['Status'] % 2 == 0) & (o['Quality'] > 1) & (o['Convergence'] < 1.03)
                   & (t['Status'] % 2 == 0) & (t['Quality'] > .2) & (t['Convergence'] < 1.03)
                   & (z['Status'] % 2 == 0) & (z['Quality'] > .2) & (z['Convergence'] < 1.03))
        for j, p in enumerate(o['Pressure']):
            if not .001 <= p <= 1.:
                continue
            oo, tt, zz = o['L2gpValue'][:, j] * 1e6, t['L2gpValue'][:, j], z['L2gpValue'][:, j] / 1000.
            good = (quality & (o['L2gpPrecision'][:, j] > 0) & (t['L2gpPrecision'][:, j] > 0)
                    & (z['L2gpPrecision'][:, j] > 0) & np.isfinite(oo) & (abs(oo) < 100.)
                    & (tt > 100.) & (tt < 350.) & (zz > 0.) & (zz < 120.))
            for mode, sampling in [('night', (o['LocalSolarTime'] < 8.) & (o['SolarZenithAngle'] > 95.)),
                                   ('day', (o['LocalSolarTime'] > 8.) & (o['SolarZenithAngle'] < 80.))]:
                keep = good & sampling
                a = pd.DataFrame(dict(lat=o['Latitude'][keep], lon=o['Longitude'][keep], lst=o['LocalSolarTime'][keep],
                                      ozone_ppmv=oo[keep], temperature_K=tt[keep], height_km=zz[keep],
                                      normalized_ozone_ppmv=oo[keep] / factor(tt[keep])))
                for lo, hi in BANDS:
                    for case in CASES:
                        r = fit_frame(a, ['ozone_ppmv', 'temperature_K', 'height_km', 'normalized_ozone_ppmv'], lo, hi, case)
                        rows.append(dict(date=str(day.date()), mode=mode, band=f'{lo}-{hi}', pressure_hPa=float(p), case=case, **r))
        print('MLS joint samples', day.date(), flush=True)
    d = rebase(pd.DataFrame(rows), ['mode', 'band', 'pressure_hPa', 'case'], ['ozone_ppmv', 'temperature_K', 'normalized_ozone_ppmv'])
    d['temperature_only_percent'] = 100. * (factor(d.temperature_K.to_numpy()) / factor(d.temperature_K_baseline.to_numpy()) - 1.)
    # Valid only as a two-reaction, fixed-composition diagnostic. Never a causal residual.
    d.loc[d['mode'] != 'night', ['temperature_only_percent', 'normalized_ozone_ppmv_percent']] = np.nan
    d.to_csv(OUT / 'MLS_matched_O3_T_native_daily.csv', index=False)
    envelope = []
    for keys, g in d.groupby(['mode', 'band', 'pressure_hPa', 'date']):
        r = dict(zip(['mode', 'band', 'pressure_hPa', 'date'], keys))
        for name in ['ozone_ppmv_percent', 'temperature_K_anomaly', 'temperature_only_percent', 'normalized_ozone_ppmv_percent']:
            r[name + '_min'] = g[name].min()
            r[name + '_max'] = g[name].max()
            r[name + '_settings'] = int(g[name].notna().sum())
        envelope.append(r)
    pd.DataFrame(envelope).to_csv(OUT / 'MLS_matched_sampling_envelope.csv', index=False)
    return hashes


def clean(a):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a) & (a > -998.), a, np.nan)


def sum_valid(arrays):
    v = np.stack(arrays)
    return np.where(np.isfinite(v).all(0), v.sum(0), np.nan)


def sample_thermal(path):
    cache = OUT / 'thermal_samples' / (path.name + '.csv')
    if cache.exists():
        a = pd.read_csv(cache, dtype={'branch': str})
        a['branch'] = pd.to_numeric(a['branch']).astype(int).astype(str)
        a['known_subset_K_day'] = sum_valid([a[k].to_numpy() for k in ['CO2_K_day', 'O3_IR_K_day', 'chem_total_K_day']])
        return a
    day = pd.to_datetime(path.name.split('_')[2], format='%Y%j')
    ta = base.LEGACY / 'data/direct_observations_2025_new_year/SABER_L2A_temperature_ozone' / str(day.date()) / path.name.replace('L2B', 'L2A').removesuffix('.gz')
    if not ta.exists():
        return pd.DataFrame()
    with xr.open_dataset(io.BytesIO(gzip.decompress(path.read_bytes())), engine='scipy', decode_times=False) as b, xr.open_dataset(io.BytesIO(ta.read_bytes()), engine='scipy', decode_times=False) as a:
        assert np.array_equal(b.event.values, a.event.values)
        assert np.allclose(b.tpaltitude.values, a.tpaltitude.values, equal_nan=True)
        z = b.tpaltitude.values
        fields = dict(temperature_K=clean(a.ktemp.values), ozone_ppmv=clean(a.O3_96.values) * 1e6,
                      CO2_K_day=sum_valid([clean(b[k].values) for k in b if k.startswith('CO2_cool_')]),
                      H2O_K_day=sum_valid([clean(b[k].values) for k in b if k.startswith('H2O_cool_')]),
                      O3_IR_K_day=clean(b.O3_cool_666_001_000.values),
                      O3_solar_K_day=sum_valid([clean(b[k].values) for k in b if k.startswith('O3_solar_heat_')]))
        for k in CHEM:
            assert b['chem_heat_' + k].attrs['units'] == 'K/day'
            fields['chem_' + k] = clean(b['chem_heat_' + k].values)
        fields['chem_total_K_day'] = sum_valid([fields['chem_' + k] for k in CHEM])
        fields['night_partial_K_day'] = sum_valid([fields['CO2_K_day'], fields['H2O_K_day'], fields['O3_IR_K_day'], fields['chem_total_K_day']])
        fields['known_subset_K_day'] = sum_valid([fields['CO2_K_day'], fields['O3_IR_K_day'], fields['chem_total_K_day']])
        rows = []
        for h in np.arange(70., 96., 5.):
            distance = np.where(np.isfinite(z), abs(z - h), np.inf)
            jj = np.argmin(distance, axis=1)
            ii = np.arange(len(jj))
            lat, lon = b.tplatitude.values[ii, jj], b.tplongitude.values[ii, jj]
            good = (distance[ii, jj] < .7) & (lat >= 20.) & (lat < 60.) & np.isfinite(lon) & (abs(lon) <= 360.)
            if not good.any():
                continue
            r = pd.DataFrame(dict(height=h, lat=lat[good], lon=lon[good] % 360,
                                  lst=(b.tpSolarLT.values[ii, jj][good] / 3600000.) % 24,
                                  utc=(b.time.values[ii, jj][good] / 3600000.) % 24,
                                  sza=b.tpSolarZen.values[ii, jj][good], branch=b.tpAD.values[good].astype(int).astype(str), orbit=path.name))
            for k, v in fields.items():
                r[k] = v[ii, jj][good]
            r.loc[~r.temperature_K.between(100., 350.), 'temperature_K'] = np.nan
            # Retain finite negative ozone retrievals, not a positive-only filter.
            r.loc[r.ozone_ppmv.abs() > 100., 'ozone_ppmv'] = np.nan
            rows.append(r)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(result):
        result.to_csv(cache, index=False)
    return result


def saber():
    (OUT / 'thermal_samples').mkdir(exist_ok=True)
    rows, hashes = [], {}
    variables = ['CO2_K_day', 'H2O_K_day', 'O3_IR_K_day', 'O3_solar_K_day', 'chem_total_K_day',
                 'night_partial_K_day', 'known_subset_K_day'] + ['chem_' + k for k in CHEM]
    for day in DAYS:
        paths = sorted((ROOT / 'source_origin_highalt_20261006/raw_SABER_L2B').glob(f'SABER_L2B_{day.year}{day:%j}_*.gz'))
        for p in paths:
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        parts = [sample_thermal(p) for p in paths]
        parts = [a for a in parts if len(a)]
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        for h, g in frame.groupby('height'):
            for mode, mask in [('night', g.sza > 105.), ('day', g.sza < 80.)]:
                for lo, hi in [(20, 40), (40, 60), (30, 40), (40, 50)]:
                    for var in variables:
                        if mode == 'day' and var in ['night_partial_K_day', 'known_subset_K_day']:
                            continue
                        for case in ['reference', 'W1_only', 'branch_0', 'branch_1'] + [f'delete_{i}' for i in range(12)]:
                            # Each component paired with T/O3. Partial sum requires
                            # all four component groups on exactly the same scans.
                            r = fit_frame(g[mask], ['temperature_K', 'ozone_ppmv', var], lo, hi, case, True)
                            rows.append(dict(date=str(day.date()), mode=mode, height=h, band=f'{lo}-{hi}', variable=var, case=case, **r))
        print('SABER extra thermal terms', day.date(), flush=True)
    d = pd.DataFrame(rows)
    # Tall format with shared column prevents treating unmatched components as a net budget.
    d['value_K_day'] = [r.get(r['variable'], np.nan) for r in d.to_dict('records')]
    d = rebase(d, ['mode', 'height', 'band', 'variable', 'case'], ['temperature_K', 'ozone_ppmv', 'value_K_day'])
    d.to_csv(OUT / 'SABER_extra_thermal_native_daily.csv', index=False)
    coverage = d[d.case == 'reference'].groupby(['mode', 'band', 'variable', 'date']).value_K_day.apply(lambda s: int(s.notna().sum())).reset_index(name='supported_heights')
    coverage.to_csv(OUT / 'SABER_extra_thermal_coverage.csv', index=False)
    fixed = []
    for keys, g in d.groupby(['mode', 'band', 'variable', 'case']):
        piv = g.pivot(index='date', columns='height', values='value_K_day').reindex(DAYS.strftime('%Y-%m-%d'))
        for label, low, high in [('70_85', 70., 85.), ('85_95', 85., 95.)]:
            nodes = piv.columns[(piv.columns >= low) & (piv.columns <= high) & piv.notna().all(0)]
            if not len(nodes):
                continue
            val = piv[nodes].mean(1)
            for i, day in enumerate(DAYS):
                fixed.append(dict(zip(['mode', 'band', 'variable', 'case'], keys), height_range=label,
                                  height_nodes=','.join(f'{h:g}' for h in nodes), date=str(day.date()),
                                  value_K_day=val.iloc[i], anomaly_K_day=val.iloc[i] - val.iloc[:4].mean()))
    pd.DataFrame(fixed).to_csv(OUT / 'SABER_extra_thermal_fixed_height_daily.csv', index=False)
    return hashes


def validation():
    rng = np.random.default_rng(372)
    la, lon = rng.uniform(20, 40, 1200), rng.uniform(0, 360, 1200)
    ll = (la - 30.) / 10.
    a = pd.DataFrame(dict(lat=la, lon=lon, value=4. + 2 * ll + ll * ll + 3 * np.cos(np.deg2rad(lon))))
    r = fit_frame(a, ['value'], 20, 40, 'reference')
    la = np.linspace(20, 40, 1001)
    expected = np.average(4. + 2 * (la - 30.) / 10. + ((la - 30.) / 10.) ** 2, weights=np.cos(np.deg2rad(la)))
    assert abs(r['value'] - expected) < 1e-9
    assert factor(np.array([190., 200., 210.])).tolist()[0] > 1. > factor(np.array([190., 200., 210.])).tolist()[2]
    assert np.isnan(sum_valid([np.array([1.]), np.array([np.nan])])[0])
    assert sum_valid([np.array([-2.]), np.array([1.])])[0] == -1.


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--part', choices=['mls', 'saber', 'all'], default='all')
    choice = args.parse_args().part
    OUT.mkdir(exist_ok=True)
    validation()
    hashes = {}
    if choice in ['mls', 'all']:
        hashes.update(mls())
    if choice in ['saber', 'all']:
        hashes.update(saber())
    method = dict(baseline=['2024-12-27', '2024-12-30'], dates=[str(DAYS[0].date()), str(DAYS[-1].date())],
                  part=choice, hashes=hashes, no_download=True, no_filling=True, no_time_filter=True,
                  joint_MLS_geolocation_and_time_asserted=True, negative_ozone_retained=True,
                  spatial_reduction='latitude polynomial + W1-W3, fixed-band cosine-latitude mean; sector and harmonic sensitivities',
                  MLS_night='LT<8, SZA>95; joint O3/T/GPH flags and precision screening',
                  MLS_day='LT>8, SZA<80; small ozone baseline can invalidate percentages',
                  SABER_night='SZA>105, no direct solar irradiation; variable-specific matched T/O3',
                  SABER_partial='CO2+H2O+O3 signed IR cooling + seven chemical heat terms, all finite on same scan; not a full heat budget',
                  SABER_known_subset='CO2+O3 signed IR product + seven chemical heat terms, excludes missing H2O; never full net heating',
                  SABER_no_latitude_extrapolation=True,
                  chemical_test='conditional two-reaction equilibrium with fixed isobar, fixed O2 mixing ratio and O/H ratio; factor=(T/200)^-3.4 exp[470(1/T-1/200)]',
                  chemical_test_not_OH_retrieval=True, equilibrium_unverified_near_86km=True,
                  missing_terms=['advection/adiabatic tendencies', 'gravity-wave dynamical heating', 'unmeasured additional species and radiation', 'Joule/particle heating'],
                  SABER_chemical_and_cooling_rates_are_retrieval_model_dependent=True,
                  references=['https://doi.org/10.1038/s43247-024-01419-2', 'https://doi.org/10.5194/acp-23-14593-2023',
                              'https://doi.org/10.1038/s41467-025-55966-z', 'https://saber.gats-inc.com/data.php'])
    (OUT / ('method_' + choice + '.json')).write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding='utf-8')
    print('VALIDATED AND SAVED', OUT, flush=True)


if __name__ == '__main__':
    main()
