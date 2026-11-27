"""Twelve daily observation-derived W1 EP-flux maps, without temporal smoothing.

Reuses the native MLS coefficients and the same QG flux calculation as the
existing period comparison. Both use variable arrow lengths and remove the
p-proportional reference-density weighting from displayed magnitudes.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from PIL import Image

from build_acp2018_event_figures_20261010 import extract, flux, background, A
from ep_flux_display_20261011 import reduced_components, shown_levels, draw_arrows, add_key, ARROW_SCALE, validate_text_frame, READABLE_MIN_INCHES, READABLE_MAX_INCHES, READABLE_REFERENCE, reference_arrow_cmap, READABLE_WIDTH_INCHES, ARROW_COLOURS
from ep_flux_display_20261011 import UNIFORM_HEAD_POINTS, UNIFORM_SHAFT_POINTS, UNIFORM_MIN_INCHES, UNIFORM_MAX_INCHES, shown_levels_with_01hpa

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'acp2018_event_figures_20261010'
CACHE = OUT / 'MLS_W123_fullheight_native.npz'
FIGURE = OUT / '04_daily_W1_EP_directions_12days.png'
DAYS = pd.date_range('2024-12-27', '2025-01-07')
VERTICAL_DISPLAY_FACTOR = 100.
DENSE_LENGTH_BOUNDS = (.050, .110)
DENSE_HEAD_POINTS = 4.5
DENSE_SHAFT_POINTS = .8
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main():
    d = dict(np.load(CACHE))
    source_dates = pd.DatetimeIndex(d['dates'])
    indices = source_dates.get_indexer(DAYS)
    assert len(indices) == 12 and (indices >= 0).all()
    lat, pressure = d['lat'], d['pressure']
    Z, T, Z0, T0 = extract(d, branch=-1, sector=0)
    fields = []
    for index in indices:
        q = flux(Z[index, ..., 0], T[index, ..., 0], Z0[index], T0[index],
                 lat, pressure, wave=1, points=3)
        q['height'] = Z0[index] / 1000.
        q['mer_reduced'], q['vert_reduced'], q['magnitude'] = reduced_components(q['P'],q['W'],lat)
        # Same c=0 total QG refractive-index diagnostic as the period figure,
        # but calculated from THIS day's background, without temporal averaging.
        bg = background(Z0[index], T0[index], lat, pressure, points=5)
        q['n2'], q['background_U'], q['background_qy'] = bg['n2'], bg['U'], bg['qy']
        fields.append(q)

    # Thin the display at existing pressure levels, never interpolate the field.
    # All days use the same levels, selected from the common approximate heights.
    hh = np.stack([q['height'] for q in fields])
    level_mask = shown_levels_with_01hpa(hh, pressure)
    level_indices = np.flatnonzero(level_mask)

    valid_magnitudes = []
    for q in fields:
        use = ((lat[:, None] >= 27.5) & (lat[:, None] <= 77.5)
               & (q['height'] >= 20) & (q['height'] <= 90))
        valid_magnitudes.extend(q['magnitude'][use & np.isfinite(q['magnitude'])])
    valid_magnitudes = np.asarray(valid_magnitudes)
    maximum = float(valid_magnitudes.max())
    vmax = float(np.ceil(maximum / 50) * 50)
    norm = PowerNorm(gamma=.5,vmin=0,vmax=vmax)
    cmap = reference_arrow_cmap()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 12, 'axes.labelsize': 12,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.linewidth': .8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 600})
    fig = plt.figure(figsize=(20 / 2.54, 18.8 / 2.54))
    gs = fig.add_gridspec(3, 4, left=.078, right=.925, bottom=.073, top=.905,
                         wspace=.16, hspace=.17)
    cells, region_rows, arrow_counts = [], [], []
    xx = np.broadcast_to(lat[:, None], fields[0]['P'].shape)
    regions = [('midlatitude_upward', 45, 65, 55, 75),
               ('upper_connection', 35, 50, 70, 85),
               ('lowlatitude_downward', 25, 40, 70, 90)]
    for k, (day, q) in enumerate(zip(DAYS, fields)):
        row, col = divmod(k, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.set(xlim=(20, 80), ylim=(20, 90), xticks=[20, 40, 60, 80],
               yticks=[20, 40, 50, 60, 80, 90])
        ax.tick_params(direction='in', top=True, right=True,
                       labelleft=col == 0, labelbottom=row == 2, pad=2)
        if col == 0:
            ax.set_ylabel('Approx. altitude (km)', labelpad=5)
        if row == 2:
            ax.set_xlabel('Latitude (°N)', labelpad=2)
        ax.set_title(f'({chr(97+k)}) {day.day} {day.strftime("%b")}',
                     loc='left', pad=5)
        # Grey cells only mark finite n^2<0. No interpolation, missing-data dots,
        # outlined boundary or rectangle; white does not certify propagation.
        grey = np.where(np.isfinite(q['n2']) & (q['n2'] < 0), 1., np.nan)
        height_centres = np.nanmedian(q['height'], axis=0)
        assert np.isfinite(height_centres).all()
        ax.pcolormesh(lat, height_centres, grey.T,
            cmap=ListedColormap(['#9a9a9a']), vmin=0, vmax=1,
            shading='auto', alpha=.35, zorder=.5)
        ax.axhline(50, color='#b0b0b0', ls=':', lw=.55, zorder=1)

        artist, use = draw_arrows(ax,lat,q['height'],q['P'],q['W'],level_mask,cmap,norm,
                                  readable=True,uniform_heads=True,length_bounds_inches=DENSE_LENGTH_BOUNDS,
                                  head_points=DENSE_HEAD_POINTS,shaft_points=DENSE_SHAFT_POINTS)
        arrow_counts.append(int(use.sum()))

        # Export full native fields and the exact display mask for reproducibility.
        for i in range(len(lat)):
            for j in range(len(pressure)):
                if not (20 <= q['height'][i, j] <= 90):
                    continue
                cells.append(dict(date=str(day.date()), latitude=float(lat[i]),
                    pressure_hPa=float(pressure[j]), height_km=q['height'][i, j],
                    Q_K_ms=q['Q'][i, j], P_ms2=q['P'][i, j], W_ms2=q['W'][i, j],
                    reduced_Fy_ms2=q['mer_reduced'][i, j], reduced_Fz_ms2=q['vert_reduced'][i, j],
                    density_weighted_Fy_over_a_ms2=q['Fy'][i,j]/A,
                    density_weighted_Fz_over_a_ms2=q['Fz'][i,j]/A,
                    reduced_magnitude_ms2=q['magnitude'][i, j],
                    EP_divergence_ms_day=q['div'][i, j], arrow_shown=bool(use[i, j]),
                    refractive_index_n2_m2=q['n2'][i,j],
                    refractive_index_valid=bool(np.isfinite(q['n2'][i,j])),
                    propagation_restricted=bool(q['n2'][i,j]<0) if np.isfinite(q['n2'][i,j]) else None))
        for name, lo, hi, bottom, top in regions:
            valid = ((xx >= lo) & (xx < hi) & (q['height'] >= bottom)
                     & (q['height'] <= top) & np.isfinite(q['Q']) & np.isfinite(q['P']))
            weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], xx.shape)[valid]
            mean = lambda key: float(np.sum(q[key][valid]*weight) / np.sum(weight))
            region_rows.append(dict(date=str(day.date()), region=name,
                native_valid_cells=int(valid.sum()), Q_K_ms=mean('Q'), P_ms2=mean('P'),
                W_ms2=mean('W'), downward_cell_fraction=float(np.mean(q['Q'][valid]<0)),
                equatorward_cell_fraction=float(np.mean(q['P'][valid]<0))))

    cax = fig.add_axes([.944, .10, .016, .78])
    cb = fig.colorbar(artist, cax=cax)
    ticks = [0,50,150,300,vmax]
    labels = {value:f'{value:g}' for value in ticks}
    cb.set_ticks(ticks)
    cb.formatter = FuncFormatter(lambda value, pos: labels.get(value, f'{value:g}'))
    cb.update_ticks()
    cb.ax.tick_params(direction='in', labelsize=9, pad=2)
    cb.ax.set_title('m² s⁻²', fontsize=10, pad=7)
    fig.suptitle('Daily W1 EP-flux direction', y=.991, fontsize=16, weight='normal')
    fig.text(.5, .945, '27 Dec 2024–7 Jan 2025', ha='center', va='center', fontsize=12)
    validate_text_frame(fig)
    fig.savefig(FIGURE, dpi=600)
    plt.close(fig)
    with Image.open(FIGURE) as im:
        assert im.width == 4724 and abs(im.info['dpi'][0]-600) < .1
        print('FIGURE', str(FIGURE), im.size)
        im.verify()
    pd.DataFrame(cells).to_csv(OUT/'daily_W1_EP_12days_native_fields.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(region_rows).to_csv(OUT/'daily_W1_EP_12days_regional_checks.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(OUT/'daily_W1_EP_12days_fields.npz', dates=DAYS.values,
        latitude=lat, pressure_hPa=pressure, displayed_pressure_indices=level_indices,
        **{key: np.stack([q[key] for q in fields]) for key in
           ['Q', 'P', 'W', 'Fy', 'Fz', 'div', 'height', 'magnitude',
            'n2', 'background_U', 'background_qy']})
    settings = dict(dates=[str(day.date()) for day in DAYS], source=str(CACHE),
        data='MLS GPH and temperature native W1 coefficients; paired local-time branches',
        calculation='each day separately; same QG flux function as period comparison',
        width_cm=20, height_cm=18.8, dpi=600, layout='3 rows x 4 columns',
        pdf_generated=False, period_comparison_not_overwritten=True,
        temporal_smoothing=False, spatial_interpolation=False, missing_not_filled=True,
        latitude_limits=[20,80], altitude_limits_km=[20,90],
        arrow_colour='true unstretched reduced EP-vector magnitude; m^2/s^2; high-contrast blue-teal-green-ochre-red; PowerNorm gamma=0.5',
        arrow_colour_anchors=ARROW_COLOURS,arrow_shaft_width_inches=DENSE_SHAFT_POINTS/72,
        arrow_shaft_linewidth_points=DENSE_SHAFT_POINTS,
        arrow_head_mutation_scale_points=DENSE_HEAD_POINTS,
        arrow_head_size_independent_of_shaft_length=True,
        magnitude_maximum=maximum, magnitude_limits=[0,vmax],
        arrow_lengths='common square-root compressed display length with visible minimum; not a linear magnitude scale',
        arrow_minimum_inches=DENSE_LENGTH_BOUNDS[0],arrow_maximum_inches=DENSE_LENGTH_BOUNDS[1],
        arrow_display_reference=READABLE_REFERENCE,
        vertical_direction_display_factor=VERTICAL_DISPLAY_FACTOR,
        arrows_are_not_wave_packet_trajectories=True,
        arrows_near_altitude_frame_excluded_to_prevent_clipping=True,
        displayed_native_pressure_hPa=pressure[level_indices].tolist(), arrows_per_day=arrow_counts,
        original_01hPa_layer_restored=True,
        rectangles=False, rectangle_labels=False, contours=False,
        rectangles_by_date={},
        grey_background='finite daily total QG n2<0 at c=0; same diagnostic as period figure',
        refractive_index_background_points=5,negative_n2_overlay_alpha=.35,
        grey_regions_are_not_verified_reflection_surfaces=True,
        invalid_n2_not_drawn=True,white_regions_not_always_valid_propagation=True,
        arrow_zorder=5,grey_zorder=.5,vertical_scale_text_removed=True,
        daily_maps_do_not_separate_upward_and_downward_flux=True)
    (OUT/'daily_W1_EP_12days_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf8')
    print('MAXIMUM', maximum, 'COLOUR_LIMIT', vmax, 'ARROWS', arrow_counts)
    print(pd.DataFrame(region_rows).to_string(index=False))


if __name__ == '__main__':
    main()
