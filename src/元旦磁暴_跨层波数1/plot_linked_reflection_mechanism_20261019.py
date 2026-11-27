"""Joint Spatio-Temporal Evidence Linking Downward Wave Flux (Fig. 1 Row 2)
to Mesospheric Wave Refraction & Reflection Arrows (Fig. 2).

Clean, minimal-text publication version:
- One time-height row and two rows of daily sections, with the legend above.
- Two focus regions marked with yellow and green solid boxes, without letter tags.
- Detailed descriptions moved to the figure caption.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.patheffects as pe
from matplotlib.colors import SymLogNorm, PowerNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Import verified core modules
import plot_daily_W1_EP_directions_20261010 as old
import plot_W1_three_bands_four_rows_20261007 as band_source
from build_acp2018_event_figures_20261010 import extract, flux, background
from ep_flux_display_20261011 import (
    reduced_components, reference_arrow_cmap, draw_arrows, validate_text_frame
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'acp2018_event_figures_20261010'
SOURCE = OUT / 'MLS_W123_fullheight_native.npz'
PREVIOUS = OUT / 'daily_W1_EP_12days_fields.npz'
PREVIOUS_SETTINGS = OUT / 'daily_W1_EP_12days_settings.json'

FIG1_DIR = ROOT / 'W1_three_bands_four_rows_Dec27_Jan10_20261007'
FIG1_REV = FIG1_DIR / 'revision_07_amplitude_450_png_only'

DAYS = pd.date_range('2024-12-27', '2025-01-10')
BANDS = [(20, 40), (40, 60), (60, 80)]
DIRECTION_VERTICAL_BOOST = 4.0
# Descriptive focus regions, not significance or diagnosed reflection boundaries.
FOCUS_REGIONS = [
    dict(band=(40, 60), start='2025-01-03', end='2025-01-10',
         height=(53.0, 64.0), latitude=(40.3, 59.7), color='#e2b000'),
    dict(band=(60, 80), start='2025-01-05', end='2025-01-09',
         height=(55.0, 88.0), latitude=(60.3, 79.5), color='#00a65a'),
]
FOCUS_LINEWIDTH = 2.0


def edges(x):
    x = np.asarray(x, dtype=float)
    return np.r_[x[0] - (x[1] - x[0]) / 2.0, (x[:-1] + x[1:]) / 2.0, x[-1] + (x[-1] - x[-2]) / 2.0]


def rotate_and_style_arrows(ax, artist_orig, artist_boost, lat, height, P, W, keep,
                            highlight_downward=True, shaft_pts=1.05):
    """Rotate FancyArrowPatches to boosted physical inclination while preserving true
    magnitude colour, and distinctly style downward-pointing arrows (W < 0) in vivid cyan-blue.
    """
    trans = ax.transData
    inv_trans = trans.inverted()
    _, vert_red, _ = reduced_components(P, W, lat)
    v_kept = vert_red[keep]
    h_kept = height[keep]

    if len(artist_orig.fixed_head_patches) == len(artist_boost.fixed_head_patches):
        for p_orig, p_boost, vz, hz in zip(
            artist_orig.fixed_head_patches, artist_boost.fixed_head_patches, v_kept, h_kept
        ):
            try:
                posA0, posB0 = p_orig._posA_posB
                posAb, posBb = p_boost._posA_posB
                pA0 = np.asarray(trans.transform(posA0), dtype=float)
                pB0 = np.asarray(trans.transform(posB0), dtype=float)
                mid0 = 0.5 * (pA0 + pB0)
                len0 = np.hypot(*(pB0 - pA0))

                pAb = np.asarray(trans.transform(posAb), dtype=float)
                pBb = np.asarray(trans.transform(posBb), dtype=float)
                vec_b = pBb - pAb
                len_b = np.hypot(*vec_b)
                if len0 > 1e-6 and len_b > 1e-6:
                    scale_len = 1.18 if (highlight_downward and vz < 0 and hz >= 48.0) else 1.0
                    unit_b = vec_b / len_b
                    new_pA = mid0 - 0.5 * len0 * scale_len * unit_b
                    new_pB = mid0 + 0.5 * len0 * scale_len * unit_b
                    p_orig.set_positions(tuple(inv_trans.transform(new_pA)), tuple(inv_trans.transform(new_pB)))
            except Exception:
                pass

            if highlight_downward and (vz < 0) and (hz >= 48.0):
                p_orig.set_facecolor('#00e5ff')
                p_orig.set_edgecolor('#00e5ff')
                p_orig.set_linewidth(shaft_pts + 0.20)
                p_orig.set_zorder(7)
                p_orig.set_path_effects([
                    pe.withStroke(linewidth=shaft_pts + 1.35, foreground='#002b49'),
                    pe.Normal()
                ])
            else:
                p_orig.set_linewidth(shaft_pts)
                p_orig.set_zorder(5)
                p_orig.set_path_effects([
                    pe.withStroke(linewidth=shaft_pts + 0.22, foreground='#1f272d'),
                    pe.Normal()
                ])

    for p_b in getattr(artist_boost, 'fixed_head_patches', []):
        p_b.remove()
    try:
        artist_boost.remove()
    except Exception:
        pass


def extend_lower_native_heat(d, dates, lo, hi, existing_height, existing_heat):
    """Append lower MLS native layers, retaining every existing heat-flux value.

    The time-height panels use the joint-fit coefficient and the original
    geostrophic heat-flux convention, not the paired-LT arrow diagnostic.
    """
    it = dates.get_indexer(DAYS)
    assert (it >= 0).all()
    ids = np.flatnonzero((d['lat'] >= lo) & (d['lat'] < hi))
    lat = d['lat'][ids]
    weights = np.cos(np.deg2rad(lat))
    z = np.take(d['cz'][0, 0, ..., 0], it, axis=0)[:, ids]
    t = np.take(d['ct'][0, 0, ..., 0], it, axis=0)[:, ids]
    t0 = np.take(d['t0'][0, 0], it, axis=0)[:, ids]
    z0 = np.take(d['z0'][0, 0], it, axis=0)[:, ids]
    zstar = band_source.H * np.log(1000.0 / d['pressure'])
    stability = band_source.R / band_source.H * (
        np.gradient(t0, zstar, axis=-1, edge_order=2)
        + band_source.R / band_source.CP * t0 / band_source.H
    )
    factor = 0.5 * band_source.G / (
        2 * band_source.OM * np.sin(np.deg2rad(lat))
        * band_source.A * np.cos(np.deg2rad(lat))
    )
    native_heat = band_source.average(
        np.where(stability > 0, factor[None, :, None] * np.imag(z * t.conj()), np.nan),
        weights, 1,
    )
    native_height = np.mean(band_source.average(z0, weights, 1), axis=0) / 1000.0
    # Include the native cell just below 20 km so its genuine upper portion can
    # be clipped at the requested axis boundary, without extrapolating values.
    lower = (np.isfinite(native_height) & (native_height >= 18.0)
             & (native_height < np.min(existing_height) - 0.02))
    assert lower.any(), 'No lower native MLS layers available.'
    height = np.r_[native_height[lower], existing_height]
    heat = np.concatenate([native_heat[:, lower], existing_heat], axis=1)
    assert np.all(np.diff(height) > 0)
    assert np.array_equal(heat[:, lower.sum():], existing_heat, equal_nan=True)
    return height, heat


def load_daily_and_band_data():
    """Load native daily EP fields and the 3-band Time-Height matrices from Fig. 1."""
    d = dict(np.load(SOURCE))
    dates = pd.DatetimeIndex(d['dates'])
    previous = dict(np.load(PREVIOUS))
    prior = json.loads(PREVIOUS_SETTINGS.read_text(encoding='utf8'))
    lat, p = d['lat'], d['pressure']

    levels = np.zeros(len(p), dtype=bool)
    levels[previous['displayed_pressure_indices']] = True
    levels[int(np.argmin(abs(p - 0.1)))] = False

    z, t, z0, t0 = extract(d, branch=-1, sector=0)
    daily_fields = []
    for day in DAYS:
        it = dates.get_loc(day)
        q = flux(z[it, ..., 0], t[it, ..., 0], z0[it], t0[it], lat, p, wave=1, points=3)
        q['height'] = z0[it] / 1000.0
        q['Zamp'] = np.abs(z[it, ..., 0])
        q['mer_reduced'], q['vert_reduced'], q['magnitude'] = reduced_components(q['P'], q['W'], lat)
        bg = background(z0[it], t0[it], lat, p, points=5)
        q['n2'], q['background_U'], q['background_qy'] = bg['n2'], bg['U'], bg['qy']
        daily_fields.append(q)

    band_list = []
    for idx, (lo, hi) in enumerate(BANDS):
        amp_csv = FIG1_DIR / f'amplitude_{lo}_{hi}.csv'
        heat_csv = (FIG1_REV / f'extended_net_proxy_{lo}_{hi}.csv'
                    if (FIG1_REV / f'extended_net_proxy_{lo}_{hi}.csv').exists()
                    else FIG1_DIR / f'net_proxy_{lo}_{hi}.csv')
        phase_csv = (FIG1_REV / f'extended_relative_phase_{lo}_{hi}.csv'
                     if (FIG1_REV / f'extended_relative_phase_{lo}_{hi}.csv').exists()
                     else FIG1_DIR / f'relative_phase_{lo}_{hi}.csv')

        if amp_csv.exists() and heat_csv.exists() and phase_csv.exists():
            df_a = pd.read_csv(amp_csv, index_col='date', parse_dates=True).reindex(DAYS)
            df_q = pd.read_csv(heat_csv, index_col='date', parse_dates=True).reindex(DAYS)
            df_p = pd.read_csv(phase_csv, index_col='date', parse_dates=True).reindex(DAYS)
            band_list.append(dict(
                band=f'{lo}°–{hi}°N', lo=lo, hi=hi,
                amp_z=df_a.columns.to_numpy(float), amp=df_a.to_numpy(),
                heat_z=df_q.columns.to_numpy(float), heat=df_q.to_numpy(),
                phase_z=df_p.columns.to_numpy(float), phase=df_p.to_numpy(),
                amp_limit=[450, 1800, 2500][idx]
            ))
        else:
            lm = (lat >= lo) & (lat < hi)
            w = np.cos(np.deg2rad(lat[lm]))[:, None]
            h_mean = np.nanmean(np.stack([q['height'][lm] for q in daily_fields]), axis=(0, 1))
            keep_z = (h_mean >= 35.0) & (h_mean <= 90.0)
            hz = h_mean[keep_z]
            hq = np.array([np.nansum(q['Q'][lm][:, keep_z] * w, axis=0) / np.sum(w) for q in daily_fields])
            ha = np.array([np.nansum(q['Zamp'][lm][:, keep_z] * w, axis=0) / np.sum(w) for q in daily_fields])
            band_list.append(dict(
                band=f'{lo}°–{hi}°N', lo=lo, hi=hi,
                amp_z=hz, amp=ha, heat_z=hz, heat=hq,
                phase_z=hz, phase=np.full_like(hq, np.nan),
                amp_limit=[450, 1800, 2500][idx]
            ))
        band_list[-1]['heat_z'], band_list[-1]['heat'] = extend_lower_native_heat(
            d, dates, lo, hi, band_list[-1]['heat_z'], band_list[-1]['heat']
        )

    return lat, p, levels, daily_fields, prior, band_list


def plot_linked_master_figure(lat, p, levels, daily_fields, prior, band_list):
    """Create the unified master figure with clean, non-overlapping titles and minimal labels."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.titlesize': 11, 'axes.labelsize': 10.5,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.linewidth': 0.85, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 600
    })

    figure_height_cm = 19.8
    fig = plt.figure(figsize=(20.0 / 2.54, figure_height_cm / 2.54))
    left, right = 0.057, 0.928
    colorbar_left, colorbar_width = 0.941, 0.011
    # Preserve the daily panels' physical height and arrow display mapping.
    daily_height = 5.1075 / figure_height_cm
    row_gap = 1.0 / figure_height_cm
    lower_bottom = 1.00 / figure_height_cm
    lower_top = lower_bottom + daily_height
    middle_bottom = lower_top + row_gap
    middle_top = middle_bottom + daily_height
    top_bottom = middle_top + row_gap
    top_top = top_bottom + 5.2 / figure_height_cm

    # No overall title; place the shared legend above all panels.
    fig.legend(
        handles=[
            Line2D([], [], color='#00c853', lw=1.4, ls='--', label='GPH W1 amp.'),
            Line2D([], [], color='#00e5ff', marker='>', markersize=7, lw=1.8,
                   markeredgecolor='#002b49', label='Downward vector ($F_z < 0$)'),
            Rectangle((0, 0), 1, 1, fc='#7a7a7a', alpha=0.45, ec='#404040', ls='--', label='$n^2 < 0$'),
            Line2D([], [], color='#8c1d40', lw=1.4, label='$\\bar{q}_y = 0$')
        ],
        loc='upper center', bbox_to_anchor=(0.5, 0.998), ncol=4, frameon=False,
        fontsize=11.2, handlelength=1.6, columnspacing=1.0, handletextpad=0.3,
        borderaxespad=0.15, borderpad=0.2
    )

    # =========================================================================
    # ROW 1: Fig. 1 Row 2 Net Vertical Wave Flux (Time-Altitude) + 50 km W1 Contours
    # =========================================================================
    gs_top = fig.add_gridspec(1, 3, left=left, right=right, bottom=top_bottom, top=top_top, wspace=0.15)
    heat_norm = SymLogNorm(linthresh=1.0, linscale=0.8, vmin=-200.0, vmax=200.0, base=10)
    heat_cmap = plt.get_cmap('RdBu_r')

    im_top = None
    for col, bdata in enumerate(band_list):
        ax = fig.add_subplot(gs_top[0, col])
        z_q = bdata['heat_z']
        keep_q = (z_q >= 40.0) & (z_q <= 90.0)
        im_top = ax.pcolormesh(
            edges(md.date2num(DAYS)), edges(z_q[keep_q]),
            bdata['heat'][:, keep_q].T,
            cmap=heat_cmap, norm=heat_norm, shading='flat', zorder=1
        )

        # GPH W1 Amplitude contours (green)
        z_a = bdata['amp_z']
        keep_a = (z_a >= 40.0) & (z_a <= 90.0)
        amp_levels = [320, 380] if col == 0 else ([1200, 1500] if col == 1 else [1600, 2100])
        cs_a = ax.contour(
            md.date2num(DAYS), z_a[keep_a], bdata['amp'][:, keep_a].T,
            levels=amp_levels, colors='#00e676', linewidths=1.3, linestyles='--', zorder=4
        )
        for coll in cs_a.collections if hasattr(cs_a, 'collections') else [cs_a]:
            coll.set_path_effects([pe.withStroke(linewidth=2.2, foreground='#083316'), pe.Normal()])
        ax.clabel(cs_a, fmt='%d', fontsize=8.0, inline=True)

        for region in FOCUS_REGIONS:
            if (bdata['lo'], bdata['hi']) != region['band']:
                continue
            x0 = md.date2num(pd.Timestamp(region['start'])) - 0.4
            x1 = md.date2num(pd.Timestamp(region['end'])) + 0.4
            low, high = region['height']
            ax.add_patch(Rectangle(
                (x0, low), x1-x0, high-low, fill=False, edgecolor=region['color'],
                linewidth=FOCUS_LINEWIDTH, linestyle='-', zorder=8,
            ))

        ax.axhline(50, color='#222222', ls=':', lw=0.9, zorder=3)
        ax.axvline(md.date2num(pd.Timestamp('2025-01-01')), color='black', ls='--', lw=1.0, zorder=5)
        ax.set(ylim=(40, 90), yticks=[40, 50, 60, 70, 80, 90])
        ax.set_xlim(md.date2num(DAYS[0]) - 0.5, md.date2num(DAYS[-1]) + 0.5)
        ax.set_xticks(pd.date_range('2024-12-28', '2025-01-09', freq='4D'))
        ax.xaxis.set_major_formatter(md.DateFormatter('%b %d'))
        if col == 0:
            ax.set_ylabel('approx. alt. (km)', labelpad=4)
        ax.tick_params(labelleft=(col == 0))
        ax.set_title(f'({chr(97 + col)}) {bdata["band"]}', loc='left', pad=5, weight='bold')

    cax_top = fig.add_axes([colorbar_left, top_bottom, colorbar_width, top_top-top_bottom])
    cb_top = fig.colorbar(im_top, cax=cax_top, ticks=[-200, -10, -1, 0, 1, 10, 200])
    cb_top.ax.set_yticklabels(['-200', '-10', '-1', '0', '1', '10', '200'])
    cb_top.ax.tick_params(labelsize=8.5, direction='in')
    cb_top.ax.set_title('$v\'_1T\'_1$\n(K m s⁻¹)', fontsize=8.5, pad=5)

    # =========================================================================
    # ROWS 2-3: Eight daily sections in chronological order, four per row.
    # =========================================================================
    key_dates = [
        ('2025-01-03', '(d) 3 Jan'),
        ('2025-01-04', '(e) 4 Jan'),
        ('2025-01-05', '(f) 5 Jan'),
        ('2025-01-06', '(g) 6 Jan'),
        ('2025-01-07', '(h) 7 Jan'),
        ('2025-01-08', '(i) 8 Jan'),
        ('2025-01-09', '(j) 9 Jan'),
        ('2025-01-10', '(k) 10 Jan')
    ]
    daily_grids = [
        fig.add_gridspec(1, 4, left=left, right=right, bottom=middle_bottom, top=middle_top, wspace=0.16),
        fig.add_gridspec(1, 4, left=left, right=right, bottom=lower_bottom, top=lower_top, wspace=0.16),
    ]

    vmin_arr, vmax_arr = prior['magnitude_limits']
    arr_norm = PowerNorm(gamma=0.5, vmin=vmin_arr, vmax=vmax_arr)
    arr_cmap = reference_arrow_cmap()

    artist_mid = None
    for idx, (d_str, d_title) in enumerate(key_dates):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(daily_grids[row][0, col])
        it = DAYS.get_loc(pd.Timestamp(d_str))
        q = daily_fields[it]
        h = np.nanmedian(q['height'], axis=0)

        ax.set_facecolor('white')

        grey = np.where(np.isfinite(q['n2']) & (q['n2'] < 0), 1.0, np.nan)
        ax.pcolormesh(lat, h, grey.T, cmap=ListedColormap(['#6e6e6e']),
                      vmin=0, vmax=1, shading='auto', alpha=0.42, zorder=0.5)
        n2_clean = np.where(np.isfinite(q['n2']), q['n2'], np.nan)
        if np.any(n2_clean < 0) and np.any(n2_clean > 0):
            ax.contour(lat, h, n2_clean.T, levels=[0.0], colors='#333333',
                       linewidths=0.7, linestyles='--', alpha=0.8, zorder=0.7)
        qy_clean = np.where(np.isfinite(q['background_qy']), q['background_qy'], np.nan)
        if np.any(qy_clean < 0) and np.any(qy_clean > 0):
            ax.contour(lat, h, qy_clean.T, levels=[0.0], colors='#8c1d40',
                       linewidths=1.05, linestyles='-', alpha=0.9, zorder=0.8)

        for lat_div in [40, 60]:
            ax.axvline(lat_div, color='#444444', ls='--', lw=0.75, alpha=0.65, zorder=1.2)
        ax.axhline(50, color='#222222', ls=':', lw=0.85, zorder=1.2)

        artist_mid, keep = draw_arrows(
            ax, lat, q['height'], q['P'], q['W'], levels, arr_cmap, arr_norm,
            readable=True, uniform_heads=True,
            length_bounds_inches=old.DENSE_LENGTH_BOUNDS,
            head_points=old.DENSE_HEAD_POINTS, shaft_points=old.DENSE_SHAFT_POINTS
        )
        artist_boost, _ = draw_arrows(
            ax, lat, q['height'],
            q['P'] * keep.astype(float),
            q['W'] * DIRECTION_VERTICAL_BOOST * keep.astype(float),
            levels, arr_cmap, arr_norm,
            readable=True, uniform_heads=True,
            length_bounds_inches=old.DENSE_LENGTH_BOUNDS,
            head_points=old.DENSE_HEAD_POINTS, shaft_points=old.DENSE_SHAFT_POINTS
        )
        rotate_and_style_arrows(ax, artist_mid, artist_boost, lat, q['height'], q['P'], q['W'], keep,
                                highlight_downward=True, shaft_pts=1.05)

        for region in FOCUS_REGIONS:
            if not pd.Timestamp(region['start']) <= pd.Timestamp(d_str) <= pd.Timestamp(region['end']):
                continue
            x0, x1 = region['latitude']
            low, high = region['height']
            ax.add_patch(Rectangle(
                (x0, low), x1-x0, high-low, fill=False, edgecolor=region['color'],
                linewidth=FOCUS_LINEWIDTH, linestyle='-', zorder=8,
            ))

        ax.set(xlim=(20, 80), ylim=(20, 90), xticks=[20, 40, 60, 80], yticks=[20, 40, 50, 60, 80, 90])
        if col == 0:
            ax.set_ylabel('approx. alt. (km)', labelpad=4)
        ax.tick_params(labelleft=(col == 0))
        if row == 1:
            ax.set_xlabel('Latitude (°N)', labelpad=2)
        ax.set_title(d_title, loc='left', pad=5, weight='bold', fontsize=10.5)

    cax_mid = fig.add_axes([colorbar_left, lower_bottom, colorbar_width, middle_top-lower_bottom])
    cb_mid = fig.colorbar(artist_mid, cax=cax_mid, ticks=[0, 50, 150, 300, 550])
    cb_mid.formatter = FuncFormatter(lambda v, pos: f'{v:g}')
    cb_mid.update_ticks()
    cb_mid.ax.tick_params(labelsize=8.5, direction='in')
    cb_mid.ax.set_title('$|\\mathbf{F}|$\n(m² s⁻²)', fontsize=8.5, pad=5)

    out_master = OUT / '05_Linked_Downward_Wave_and_Reflection_Mechanism.png'
    validate_text_frame(fig)
    # User preference (2026-10-20): export PNG only, without a companion PDF.
    fig.savefig(out_master, dpi=600)
    plt.close(fig)
    print(f'Successfully generated Clean Linked Master Figure:\n  {out_master}')


def main():
    lat, p, levels, daily_fields, prior, band_list = load_daily_and_band_data()
    plot_linked_master_figure(lat, p, levels, daily_fields, prior, band_list)


if __name__ == '__main__':
    main()
