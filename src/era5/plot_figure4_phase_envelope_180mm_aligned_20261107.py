"""Reorder the established Figure 4 without changing its diagnostics."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'plot_packages'), str(ROOT / 'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image
import cartopy.crs as ccrs
import plot_figure4_wave_vortex_reconstruction_20261027 as old
import figure4_full_depth_data_20261030 as full_depth

OUT = ROOT / 'results' / 'Figure4_phase_envelope_180mm_aligned_20261107'
W, H = 18., 14.2
# Crop the existing canvas directly, without scaling or repositioning panels.
CROP_LEFT, CROP_BOTTOM, CROP_TOP = .08 * (15.8/18), .025 * (15.8/18), .275 * (15.8/18)  # inches
EXPORT_BBOX = Bbox.from_extents(CROP_LEFT, CROP_BOTTOM,
                                (W * 15.8/18) / 2.54, (H * 15.8/18) / 2.54 - CROP_TOP)


def plot(waves, source, envelope, recon):
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    # Reuse the exact original panel renderers, data grids and axis transforms.
    edge, phase = old.top_row(fig, waves, source, envelope)
    phase_bar, edge_bar = fig.axes[2], fig.axes[3]
    phase.set_position([.080, .570, .285, .340])
    edge.set_position([.580, .570, .290, .340])
    for cax, x in [(phase_bar, .437), (edge_bar, .936)]:
        pos = cax.get_position()
        cax.set_position([x, .570, pos.width, .340])
    # Put the latitude ranges in the titles, avoiding two competing title rows.
    for ax in (phase, edge):
        for t in list(ax.texts):
            if t.get_text() in ('60–80°N', old.SECTOR_LABEL):
                t.remove()
    phase.set_title('', loc='left')
    edge.set_title('', loc='left')
    phase.set_title('W1 trough | 60–80°N', loc='center', pad=6)
    edge.set_title('Z envelope | 60–130°W', loc='center', pad=6)
    phase_bar.set_title('Lon. (°)', pad=9)
    edge_bar.set_title('Lat. (°N)', pad=9)
    for ax, letter in [(phase, 'a'), (edge, 'b')]:
        ax.text(-.025, 1.025, letter, transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='left', va='bottom')
        for side in ax.child_axes:
            if side.yaxis.get_label_position() == 'right':
                side.yaxis.labelpad = 4.0
            label = side.yaxis.get_label().get_text()
            if label:
                side.yaxis.get_label().set_text(label[0].upper() + label[1:])

    width = .195
    height = W * width / H
    row_shift = -.005
    c = fig.add_axes([.045, .160 + row_shift, width, height],
                     projection=ccrs.NorthPolarStereo(central_longitude=0))
    d = fig.add_axes([.295, .160 + row_shift, width, height],
                     projection=ccrs.NorthPolarStereo(central_longitude=0))
    e = fig.add_axes([.580, .152 + row_shift, .382, height + .010])
    for ax, pressure, letter, color in [
        (c, old.UPPER_PRESSURE, 'c', old.legacy.PURPLE),
        (d, old.LOWER_PRESSURE, 'd', old.legacy.BLUE),
    ]:
        n_axes, n_text = len(fig.axes), len(fig.texts)
        old.wave_envelope_map(fig, ax, source, recon[pressure], pressure, letter, color)
        for color_ax in fig.axes[n_axes:]:
            pos = color_ax.get_position()
            color_ax.set_position([pos.x0, .055, pos.width, pos.height])
        for label in fig.texts[n_text:]:
            x, y = label.get_position()
            label.set_position((x, .059))
        leg = ax.get_legend()
        handles = list(leg.legend_handles)
        labels = [item.get_text() for item in leg.get_texts()]
        leg.remove()
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.13),
                  ncol=2, frameon=False, fontsize=10, handlelength=.8,
                  handletextpad=.2, columnspacing=.35, borderaxespad=0, labelspacing=.2)
        for label in ax.texts:
            if label.get_text() == '40°N':
                label.set_position((90, 40))
                label.set_visible(True)
            elif label.get_text() == '60°N':
                label.set_position((175 if letter == 'c' else -120, 60))
                label.set_visible(True)
            elif label.get_text() in ('180°', '0°', '60°W', '120°W', '60°E', '120°E'):
                x, y = label.get_position()
                scale = 1.025 if label.get_text() in ('180°', '120°W', '120°E') else 1.065
                label.set_position((.5 + (x-.5)*scale, .5 + (y-.5)*scale))
        title = ax.get_title(loc='left')
        ax.set_title('', loc='left')
        ax.set_title(title[4:] if title.startswith(f'({letter}) ') else title,
                     loc='center', y=1.16, pad=2)
        ax.text(-.10, 1.16, letter, transform=ax.transAxes, fontsize=8,
                fontweight='bold', ha='left', va='bottom')
    series = recon[old.UPPER_PRESSURE]['series']
    for name, color, ls, width, label in [
        ('actual', '#202020', '-', 1.15, 'Actual Z envelope'),
        ('W0_W1', old.legacy.BLUE, '-', 1.0, 'W0 + W1'),
        ('W0_W1_W2', old.legacy.ORANGE, '--', 1.0, 'W0 + W1 + W2'),
    ]:
        e.plot(old.DAYS, series[name], color=color, ls=ls, lw=width, label=label)
    e.set_ylim(48, 72)
    e.set_yticks([50, 55, 60, 65, 70])
    e.set_ylabel('Latitude (°N)', labelpad=3)
    e.set_title('Z-edge reconstruction (3 hPa)', loc='center', pad=7)
    e_letter = e.text(-.025, 1.02, 'e', transform=e.transAxes, fontsize=8,
                      fontweight='bold', ha='left', va='bottom')
    old.clean.time_axis(e, '#c74336')
    e.grid(axis='y', color='.85', ls=':', lw=.6, zorder=-3)
    e.legend(loc='upper right', frameon=False, handlelength=1.1,
             handletextpad=.4, borderaxespad=.3, labelspacing=.3)
    # Text sizes for the current manuscript figure.
    colorbar_axes = [phase_bar, edge_bar] + fig.axes[-2:]
    for item in fig.findobj(match=Text):
        if item.get_text() in ('a', 'b', 'c', 'd', 'e') and item.get_fontweight() == 'bold':
            item.set_fontsize(11)
        else:
            item.set_fontsize(10)
            item.set_fontweight('normal')
        item.set_fontfamily('Arial')
    for barax in colorbar_axes:
        barax.tick_params(labelsize=9)
        barax.title.set_fontsize(9)
        barax.xaxis.label.set_fontsize(9)
        barax.yaxis.label.set_fontsize(9)
    # Units next to the two lower horizontal colorbars belong to the bars.
    for item in fig.texts:
        if item.get_text() == 'm': item.set_fontsize(9)
    for ax in (c, d):
        for item in ax.texts:
            if item.get_text() in ('40°N', '60°N'):
                item.set_fontsize(9)
            elif '°' in item.get_text():
                item.set_fontsize(10)
    for ax in (c, d, e):
        for item in ax.get_legend().get_texts(): item.set_fontsize(9)
    for item in phase.texts:
        if item.get_text() in ('MLS', 'ERA5'):
            item.set_fontsize(9)
    fig.set_size_inches(W/2.54, H/2.54)
    fig.canvas.draw()
    # Set the top edge of each horizontal colorbar exactly 1 mm below its
    # map legend, using final rendered positions rather than visual estimates.
    renderer = fig.canvas.get_renderer()
    for ax, barax in zip((c, d), colorbar_axes[-2:]):
        legbox = ax.get_legend().get_window_extent(renderer)
        top = (legbox.y0 - fig.dpi / 25.4) / fig.bbox.height
        pos = barax.get_position()
        barax.set_position([pos.x0, top-pos.height, pos.width, pos.height])
        for item in fig.texts:
            if item.get_text() == 'm' and abs(item.get_position()[0]-(ax.get_position().x1-.001)) < .01:
                item.set_y(top-pos.height/2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_shift = (c.title.get_window_extent(renderer).y0 -
                   e.title.get_window_extent(renderer).y0) / fig.bbox.height
    pos = e.get_position()
    new_top = pos.y1 + title_shift
    lower_edge = .115
    e.set_position([pos.x0, lower_edge, pos.width, new_top-lower_edge])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    print('tight_mm', np.round(np.array(fig.get_tightbbox(renderer).extents)*25.4,2),
          'canvas_mm', W*10,H*10)
    print('e_xlabel_mm', np.round(np.array(e.xaxis.label.get_window_extent(renderer).extents)/fig.dpi*25.4,2))
    print('bar_c_mm', np.round(np.array(colorbar_axes[-2].get_window_extent(renderer).extents)/fig.dpi*25.4,2))
    # Align only the e panel letter with c; preserve the position of panel e,
    # its title and its data exactly.
    c_letter = next(t for t in c.texts if t.get_text() == 'c' and t.get_fontweight() == 'bold')
    renderer = fig.canvas.get_renderer()
    dy = (c_letter.get_window_extent(renderer).y0 -
          e_letter.get_window_extent(renderer).y0) / e.bbox.height
    e_letter.set_y(e_letter.get_position()[1] + dy)
    fig.canvas.draw()
    for panel, bar in [(phase, phase_bar), (edge, edge_bar)]:
        for side in panel.child_axes:
            for label in [side.yaxis.label, *side.get_yticklabels()]:
                if label.get_visible() and label.get_text().strip():
                    box = label.get_window_extent(fig.canvas.get_renderer())
                    if box.x0 < 0 or box.x1 > fig.bbox.width:
                        raise ValueError('Clipped axis text: ' + label.get_text())
    # Fit rendered content into 180 mm with a 1 mm margin on every side.
    margin = 1 / 25.4
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Move only the bottom edge of panel e so that its horizontal axis
        # title is vertically centered with the two map colorbars.
        label_box = e.xaxis.label.get_window_extent(renderer)
        bar_box = colorbar_axes[-2].get_window_extent(renderer)
        dy = ((bar_box.y0 + bar_box.y1) - (label_box.y0 + label_box.y1)) / (2*fig.bbox.height)
        if abs(dy) > .0001:
            p = e.get_position()
            e.set_position([p.x0, p.y0+dy, p.width, p.height-dy])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        delta = 180 / 25.4 - (tight.width + 2 * margin)
        if abs(delta) < .0005:
            break
        fig.set_size_inches(fig.get_figwidth() + delta, fig.get_figheight(), forward=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    export_box = Bbox.from_extents(tight.x0-margin, tight.y0-margin,
                                   tight.x0-margin+180/25.4, tight.y1+margin)
    print('export_mm', round(export_box.width*25.4,2), round(export_box.height*25.4,2))
    assert export_box.height*25.4 <= 170
    file = OUT / 'Figure4_W1_phase_and_vortex_envelope.png'
    fig.savefig(file, dpi=600, bbox_inches=export_box, pad_inches=0)
    fig.savefig(file.with_suffix('.svg'), dpi=600, bbox_inches=export_box, pad_inches=0)
    plt.close(fig)
    return file


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old.OUT = OUT
    old.clean.OUT = OUT
    old.legacy.OUT = OUT
    old.shared.OUT = OUT
    old.shared.QA = {}
    old.shared.style()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
        'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'svg.fonttype': 'none'})
    source = np.load(old.shared.base.CACHE)
    waves = old.clean.load_vertical_waves(source)
    full_depth.all_phase(waves, old.DAYS, old.shared, OUT)
    envelope = old.load_sector_envelopes(source, waves)
    recon, score = old.reconstruction(source, waves, envelope)
    file = plot(waves, source, envelope, recon)
    with Image.open(file) as im:
        exported_size_mm = [round(v / 600 * 25.4, 2) for v in im.size]
    (OUT / 'figure_settings.json').write_text(json.dumps({
        'panel_order': ['W1 trough longitude', 'North American Z envelope',
                        '3 hPa W1 and Z envelope', '50 hPa W1 and Z envelope',
                        '3 hPa edge reconstruction'],
        'sector': '60–130°W', 'data_processing': 'original calculations and fields retained',
        'exported_size_mm': exported_size_mm, 'dpi': 600,
        'ordinary_text_pt': 10, 'panel_letters_pt': 11, 'colorbar_text_pt': 9,
        'legend_text_pt': 9, 'latitude_labels_cd_pt': 9, 'MLS_ERA5_labels_pt': 9,
        'source_script_sha256': hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
        'prior_qa': old.shared.QA}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('Saved:', file)


if __name__ == '__main__':
    main()
