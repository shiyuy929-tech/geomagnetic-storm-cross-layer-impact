"""Replot manuscript Fig. 2 using temperature-only nighttime daily means.

All non-temperature panels, pressure-level heights and layout are taken from
the 2026-11-09 figure script. No inputs or existing outputs are overwritten.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import plot_figure1_f_min30_20261109 as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'figure2_temperature_dailymean_20261121'
DATE_RANGE = pd.date_range('2024-12-28', '2025-01-11')
REF_START, REF_END = '2024-12-28', '2024-12-30'


def source_files():
    dirs = [ROOT / 'propagation_audit_20260927/data/MLS/Temperature',
            ROOT / 'data/MLS_Temperature_v06_20241225_20250110',
            ROOT / 'W1_three_bands_Dec27_Jan20_20261016/raw_MLS_20250111_20250120']
    found = {}
    for directory in dirs:
        for path in sorted(directory.glob('MLS-Aura_L2GP-Temperature_v06-*.he5')):
            day = pd.to_datetime(path.stem.rsplit('_', 1)[-1], format='%Yd%j')
            if day in DATE_RANGE:
                found[day] = path
    missing = DATE_RANGE.difference(pd.DatetimeIndex(found))
    if len(missing):
        raise FileNotFoundError(f'Missing MLS Temperature files: {missing.tolist()}')
    return found


def calculate():
    files = source_files()
    rows, manifest = [], []
    for day in DATE_RANGE:
        path = files[day]
        manifest.append({'date': day.strftime('%Y-%m-%d'), 'path': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with h5py.File(path, 'r') as f:
            swath = f['HDFEOS/SWATHS/Temperature']
            d, geo = swath['Data Fields'], swath['Geolocation Fields']
            pressure, lat = geo['Pressure'][:], geo['Latitude'][:]
            qc = ((d['Status'][:] % 2 == 0) & (d['Quality'][:] > .2)
                  & (d['Convergence'][:] < 1.03) & (geo['SolarZenithAngle'][:] > 95))
            for j, p in enumerate(pressure):
                if not .001 <= p <= 1.:
                    continue
                value = d['L2gpValue'][:, j].astype(float)
                good = (qc & (d['L2gpPrecision'][:, j] > 0) & np.isfinite(value)
                        & (value > 100) & (value < 350))
                for lo, hi in [(40, 60), (60, 80)]:
                    selected = good & (lat >= lo) & (lat < hi)
                    rows.append(dict(date=day.strftime('%Y-%m-%d'), band=f'{lo}-{hi}',
                                     pressure_hPa=float(p), count=int(selected.sum()),
                                     temperature_K=float(np.mean(value[selected])) if selected.any() else np.nan))
    frame = pd.DataFrame(rows)
    frame['pressure_key'] = frame.pressure_hPa.round(9)
    baseline = (frame[frame.date.between(REF_START, REF_END)]
                .groupby(['band', 'pressure_key'])
                .agg(baseline_K=('temperature_K', 'mean'), baseline_days=('temperature_K', 'count'))
                .reset_index())
    frame = frame.merge(baseline, on=['band', 'pressure_key'], validate='many_to_one')
    frame['dT_daily_mean'] = np.where(frame.baseline_days == 3,
                                      frame.temperature_K - frame.baseline_K, np.nan)
    return frame, manifest


def main():
    OUT.mkdir(exist_ok=True)
    temp, manifest = calculate()
    temp.to_csv(OUT / 'night_temperature_daily_means_and_counts.csv', index=False)
    source_load = original.src.load
    checks = {}

    def load_with_new_temperature():
        chem, particles, hourly, kp, dst, palette, hashes, minima, end = source_load()
        old = chem[['date', 'band', 'pressure_hPa', 'reference_height_km', 'dT']].copy()
        chem['pressure_key'] = chem.pressure_hPa.round(9)
        merged = chem.merge(temp[['date', 'band', 'pressure_key', 'dT_daily_mean', 'count']],
                            on=['date', 'band', 'pressure_key'], validate='one_to_one')
        assert len(merged) == len(chem)
        merged['dT'] = merged.dT_daily_mean
        # The unweighted temperature observations exceed the old colour limits.
        # Expand only panel i so all of its valid values remain distinguishable.
        scale = next(s for s in palette['individual_panel_colour_scales'] if s['panel'] == 'i')
        scale['colour_limits'] = [-14.0, 17.0]
        scale['colourbar_ticks'] = [-14.0, -7.0, 0.0, 8.0, 17.0]
        compare = old.merge(merged[['date', 'band', 'pressure_key', 'dT_daily_mean', 'count']],
                            left_on=['date', 'band', old.pressure_hPa.round(9)],
                            right_on=['date', 'band', 'pressure_key'], validate='one_to_one')
        compare = compare.rename(columns={'dT': 'old_fitted_dT_K', 'dT_daily_mean': 'new_mean_dT_K'})
        compare['difference_K'] = compare.new_mean_dT_K - compare.old_fitted_dT_K
        compare.to_csv(OUT / 'temperature_old_vs_new.csv', index=False)
        checks['new_finite_cells'] = int(merged.dT.notna().sum())
        checks['old_finite_cells'] = int(old.dT.notna().sum())
        return merged.drop(columns=['pressure_key', 'dT_daily_mean', 'count']), particles, hourly, kp, dst, palette, hashes, minima, end

    original.src.load = load_with_new_temperature
    original.OUT = OUT
    plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none'})
    savefig = Figure.savefig

    def save_with_alignment_check(self, fname, *args, **kwargs):
        name = Path(fname)
        if name.suffix.lower() == '.png':
            # The author's local-only panel-alignment audit is omitted in this shared copy.
            self.savefig(OUT / 'Figure2_temperature_dailymean.svg', *args, **kwargs)
        return savefig(self, fname, *args, **kwargs)

    Figure.savefig = save_with_alignment_check
    try:
        original.main()
    finally:
        Figure.savefig = savefig
        original.src.load = source_load
    shutil.copy2(OUT / 'Figure1_ozone_dailymean_candidate.png',
                 OUT / 'Figure2_temperature_dailymean.png')
    (OUT / 'temperature_source_manifest.json').write_text(json.dumps({
        'method': 'temperature-only QC; solar zenith angle >95 degrees; arithmetic daily mean',
        'temperature_qc': 'even Status, Quality>0.2, Convergence<1.03, precision>0, 100<T<350 K',
        'reference': '2024-12-28 through 2024-12-30, same band and native pressure',
        'temperature_night_filter_matches_ozone': True,
        'height_reference': 'unchanged original geopotential-height reference coordinate',
        'panel_i_colour_limits_K': [-14, 17],
        'files': manifest, **checks}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    main()
