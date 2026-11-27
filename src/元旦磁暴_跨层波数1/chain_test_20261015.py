"""First observational audit of EPP -> chemistry/thermal state -> PNJ.

Uses existing sources only. No missing-data filling, source-spliced derivatives,
counterfactual propagation, inferred particle heating or causal residual.
"""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
import pandas as pd
from test_background_turning_joint_20261009 import diff_native

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'chain_test_20261015'
ERA = ROOT / 'zonal_wind_diagnostics_20261005/data/ERA5_ml_1_50_uvt_20241227_20250110_2p5deg.nc'
TABLE = ROOT / 'zonal_wind_diagnostics_20261005/ERA5_L137_definitions.csv'
MLS = ROOT / '2025_ozone_thermal_circulation_20261012/MLS_matched_O3_T_native_daily.csv'
PARTICLE = ROOT / '2025_observed_storm_mechanism_20261013/particle_common_cell_daily.csv'
THERMAL = ROOT / '2025_ozone_thermal_circulation_20261012/SABER_extra_thermal_fixed_height_daily.csv'
DAYS = pd.date_range('2024-12-27', '2025-01-10')
PRE = DAYS <= '2024-12-30'
A, R, OMEGA, H_KM = 6371000., 287., 7.292115e-5, 7.


def anomaly(v):
    v = np.asarray(v, float)
    return v - np.mean(v[PRE], axis=0)


def read_zonal(h, name):
    data = h[name][:].astype(float)
    attrs = h[name].attrs
    for key in ['missing_value', '_FillValue']:
        if key in attrs:
            data[np.isin(data, attrs[key])] = np.nan
    data = data * float(np.asarray(attrs.get('scale_factor', 1.)).ravel()[0])
    data += float(np.asarray(attrs.get('add_offset', 0.)).ravel()[0])
    assert np.isfinite(data).all()
    return data.mean(-1)


def thermal_wind():
    with h5py.File(ERA) as h:
        times = pd.to_datetime(h['valid_time'][:], unit='s')
        lat = h['latitude'][:].astype(float)
        levels = h['model_level'][:].astype(int)
        u, t = read_zonal(h, 'u'), read_zonal(h, 't')
    order = np.argsort(lat)
    lat, u, t = lat[order], u[:, :, order], t[:, :, order]
    assert all(sum(times.normalize() == d) == 4 for d in DAYS)
    u, t = [np.stack([v[times.normalize() == d].mean(0) for d in DAYS]) for v in [u, t]]
    defs = pd.read_csv(TABLE).set_index('level')
    p = ((defs.a_Pa + defs.a_Pa.shift(1)) / 200.).loc[levels].to_numpy()
    assert np.isfinite(p).all() and (np.diff(p) > 0).all()
    assert np.all(defs.loc[levels, 'b'] == 0)
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))[None, None, :]
    rows = []
    for n in [3, 5]:
        dtdy = diff_native(t, A * np.deg2rad(lat), 1, n, 2)
        dulnp = diff_native(u, np.log(p), 1, n, 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            predicted = R * dtdy / f
        actual_shear, predicted_shear = -dulnp / H_KM, -predicted / H_KM
        fields = {'U_ms': u, 'T_K': t, 'dTdy_K_1000km': dtdy * 1e6,
                  'actual_shear_ms_km': actual_shear,
                  'thermal_wind_shear_ms_km': predicted_shear,
                  'actual_minus_thermal_wind_ms_km': actual_shear - predicted_shear}
        for lev in [3, 4, 5, 6, 8, 10, 12, 13, 16, 20]:
            j = int(np.flatnonzero(levels == lev)[0])
            for lab, lm in [(str(x), lat == x) for x in [45., 55., 65., 75.]] + [
                ('40-60', (lat >= 40) & (lat < 60)), ('60-70', (lat >= 60) & (lat < 70))]:
                w = np.cos(np.deg2rad(lat[lm]))
                reduced = {k: np.average(v[:, j, lm], weights=w, axis=1) for k, v in fields.items()}
                delta = {k: anomaly(v) for k, v in reduced.items()}
                for i, day in enumerate(DAYS):
                    rows.append(dict(date=str(day.date()), stencil_points=n, model_level=lev,
                                     pressure_hPa=p[j], latitude_or_band=lab,
                                     **{k: v[i] for k, v in reduced.items()},
                                     **{'delta_' + k: v[i] for k, v in delta.items()}))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'chain_test_ERA5_daily_gradient_shear.csv', index=False)
    period = []
    for key, g in d.groupby(['stencil_points', 'model_level', 'pressure_hPa', 'latitude_or_band']):
        for label, lo, hi in [('baseline', '2024-12-27', '2024-12-30'),
                              ('early', '2025-01-01', '2025-01-03'), ('late', '2025-01-04', '2025-01-10')]:
            g1 = g[g.date.between(lo, hi)]
            period.append(dict(zip(['stencil_points', 'model_level', 'pressure_hPa', 'latitude_or_band'], key),
                               period=label, **g1.select_dtypes('number').drop(columns=['stencil_points', 'model_level', 'pressure_hPa']).mean().to_dict()))
    pd.DataFrame(period).to_csv(OUT / 'chain_test_ERA5_period_gradient_shear.csv', index=False)
    return d


def chemistry_and_particles():
    m = pd.read_csv(MLS)
    m = m[m['mode'].eq('night') & m.band.isin(['20-40', '40-60', '60-80'])].copy()
    m = m[np.isclose(m.pressure_hPa.to_numpy()[:, None],
                    np.array([.0021544348, .01, .021544348, .046415888, .1, .46415888])[None, :], rtol=1e-5).any(1)]
    m.to_csv(OUT / 'chain_test_MLS_night_O3_T_values_settings.csv', index=False)
    rows = []
    for key, g in m.groupby(['date', 'band', 'pressure_hPa']):
        ref = g[g['case'].eq('reference')].iloc[0]
        row = dict(zip(['date', 'band', 'pressure_hPa'], key), height_km=ref.height_km)
        for field in ['ozone_ppmv_percent', 'temperature_K_anomaly']:
            row[field] = ref[field]
            row[field + '_min'] = g[field].min()
            row[field + '_max'] = g[field].max()
            row[field + '_nsettings'] = int(g[field].notna().sum())
        rows.append(row)
    env = pd.DataFrame(rows)
    env.to_csv(OUT / 'chain_test_MLS_night_envelope.csv', index=False)
    # A finite, broad-band T contrast is not relabelled as a local gradient.
    contrast = []
    ph = np.linspace(40., 60., 1001)
    c1 = np.average(ph, weights=np.cos(np.deg2rad(ph)))
    ph = np.linspace(60., 80., 1001)
    c2 = np.average(ph, weights=np.cos(np.deg2rad(ph)))
    dist = A * np.deg2rad(c2 - c1)
    for key, g in m.groupby(['date', 'pressure_hPa', 'case']):
        a = g[g.band.eq('40-60')]
        b = g[g.band.eq('60-80')]
        if len(a) != 1 or len(b) != 1:
            continue
        contrast.append(dict(zip(['date', 'pressure_hPa', 'case'], key),
            north_minus_south_T_K=float(b.temperature_K.iloc[0] - a.temperature_K.iloc[0]),
            delta_north_minus_south_T_K=float(b.temperature_K_anomaly.iloc[0] - a.temperature_K_anomaly.iloc[0]),
            finite_band_contrast_K_1000km=float((b.temperature_K.iloc[0] - a.temperature_K.iloc[0]) / dist * 1e6),
            delta_finite_band_contrast_K_1000km=float((b.temperature_K_anomaly.iloc[0] - a.temperature_K_anomaly.iloc[0]) / dist * 1e6)))
    pd.DataFrame(contrast).to_csv(OUT / 'chain_test_MLS_broad_meridional_T_contrast.csv', index=False)
    a = pd.read_csv(PARTICLE)
    a = a[a.band.eq('60-80')].copy()
    parts = []
    for _, g in a.groupby(['satellite', 'channel', 'band']):
        g = g.copy()
        pre = g[g.date.le('2024-12-30')]
        assert len(pre) == 4 and pre.value.notna().all()
        baseline = pre.value.mean()
        g['baseline_Dec27_30'] = baseline
        g['delta_from_baseline'] = g.value - baseline
        # No ratio for background-subtracted TED energy flux.
        g['ratio_to_baseline_MEPED_only'] = g.value / baseline if g.channel.iloc[0].startswith('MEPED') else np.nan
        parts.append(g)
    particles = pd.concat(parts)
    particles.to_csv(OUT / 'chain_test_particle_fixed_cells.csv', index=False)
    s = pd.read_csv(THERMAL)
    s = s[s['case'].eq('reference') & s['mode'].eq('night') &
          s.variable.isin(['CO2_K_day', 'O3_IR_K_day', 'chem_total_K_day', 'known_subset_K_day'])].copy()
    s.to_csv(OUT / 'chain_test_SABER_partial_thermal.csv', index=False)
    return env, particles


def main():
    OUT.mkdir(exist_ok=True)
    ea = thermal_wind()
    env, particles = chemistry_and_particles()
    meta = dict(question='Does real chronology support each EPP/composition/thermal-state/PNJ/W1 link?',
        baseline='2024-12-27 through 2024-12-30 inclusive',
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ERA, TABLE, MLS, PARTICLE, THERMAL]},
        no_new_download=True, no_imputation=True, no_time_smoothing=True,
        thermal_wind='du/dln(p) = (R/f) dT/dy; z*=-7 km ln(p/p_ref); both derivatives on native same-source ERA5 grid',
        derivative_settings=[3, 5], finite_centered_stencils_only=True,
        MLS_T_contrast='difference of broad, cosine-latitude-weighted bands; not a resolved local gradient',
        MLS_envelope='processing sensitivity including sector deletion; not confidence interval',
        particle_notes=['m01=MetOp-B and m03=MetOp-C; verify source metadata',
          'MEPED directional e2/e3 cone-screened, explicit IFC0, common lat/MLT cells; proton contamination/degradation uncorrected',
          'Not deposited power, ionization rate or loss-cone-integrated flux',
          'TED energy flux maps to 120 km, not direct 50-80 km heating',
          'Dec31 common-cell coverage incomplete; retained missing',
          'Baseline already includes Dec30 enhancement; exposure need not begin Jan1 Dst minimum'],
        SABER_notes=['Existing retrieval/model-dependent thermal components, 20-60N support only',
          'CO2 + O3 IR + chemical heat subset omits H2O where missing and excludes dynamics and particle/Joule heating',
          'No polar PNJ heating budget or causal ozone radiative tendency'],
        attribution_limits=['Thermal-wind consistency is a balance constraint, not causal direction',
          'No estimate of ozone-induced heating from observed ozone alone',
          'Pre-existing upward W1 and wave-mean-flow feedback remain viable',
          'Jan4 secondary activity prevents treating Jan4-10 as a single-isolated-storm response'])
    (OUT / 'chain_test_method.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SAVED', OUT)
    d = ea[(ea.stencil_points == 5) & (ea.model_level.isin([5, 8, 12, 13])) & ea.latitude_or_band.isin(['40-60', '60-70'])]
    print(d[d.date.isin(['2024-12-30', '2024-12-31', '2025-01-01', '2025-01-03', '2025-01-06'])][
        ['date', 'model_level', 'pressure_hPa', 'latitude_or_band', 'delta_dTdy_K_1000km',
         'delta_actual_shear_ms_km', 'delta_thermal_wind_shear_ms_km']].to_string(index=False))
    print('MLS O3 60-80N ~86km')
    print(env[(env.band == '60-80') & np.isclose(env.pressure_hPa, .0021544348)][
        ['date', 'ozone_ppmv_percent', 'ozone_ppmv_percent_min', 'ozone_ppmv_percent_max', 'temperature_K_anomaly']].to_string(index=False))


if __name__ == '__main__':
    main()
