"""Restyle Figure 2 using the existing Dec 28--30 chemistry baseline.

Produces the requested four-index version.
Kp retains its native 3-hour values; no data are smoothed or interpolated.
Red and grey dashed lines mark SSC and the end of the main phase, respectively.
No shaded interval or event-line legend is drawn.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.text import Text
from PIL import Image

import plot_figure2_Dec28_Jan11_20261022 as original

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'figure2_baseline_Dec28_Dec30_20261022'
OUT = HERE / 'figure2_indices_mainphase_20261101'
WIDTH, HEIGHT = 20., 19.75
DAYS = pd.date_range('2024-12-28', '2025-01-11')
MID = DAYS + pd.Timedelta(hours=12)
START, STOP = DAYS[0], DAYS[-1] + pd.Timedelta(days=1)
SSC = pd.Timestamp('2024-12-31 16:18')
COLORS = dict(Kp='#8b75ae', Dst='#c74d3d', AE='#ae7832', PCN='#287bb5',
              **{'40-60': '#0072B2', '60-80': '#D55E00'})
AXIS_LABEL_SIZE = 13.
INDEX_LINEWIDTH = 1.6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    files = ['night_ozone_temperature_native_daily.csv', 'MEPED_plotted_values.csv',
             'AE_PCN_hourly.csv', 'Kp_3hourly_GFZ.csv', 'Dst_hourly_OMNI.csv']
    fingerprints = {name: sha(SOURCE / name) for name in files}
    chem = pd.read_csv(SOURCE / files[0])
    particles = pd.read_csv(SOURCE / files[1])
    hourly = pd.read_csv(SOURCE / files[2], index_col=0, parse_dates=True)
    kp = pd.read_csv(SOURCE / files[3], parse_dates=['time'])
    dst = pd.read_csv(SOURCE / files[4], parse_dates=['time'])
    palette = json.loads((original.BASE / 'data_and_layout_checks.json').read_text(encoding='utf-8'))
    # Hourly OMNI records are located at the interval centre (:30). Preserve
    # their timestamps; use the right edge of the last minimum bin as the
    # display boundary. This is 17:00 UT, consistent with Jiao et al. (2025).
    event = dst[dst.time.between(SSC, pd.Timestamp('2025-01-02'))]
    minima = event[event.Dst_nT.eq(event.Dst_nT.min())]
    end = minima.time.max() + pd.Timedelta(minutes=30)
    assert end == pd.Timestamp('2025-01-01 17:00')
    assert len(kp) == 120 and len(dst) == len(hourly) == 360
    assert (kp.time.diff().dropna() == pd.Timedelta(hours=3)).all()
    for table in [chem, particles]:
        assert table.date.nunique() == 15
    return chem, particles, hourly, kp, dst, palette, fingerprints, minima, end


def ax_at(fig, x, y, w, h):
    return fig.add_axes([x/WIDTH, y/HEIGHT, w/WIDTH, h/HEIGHT])


def style(ax, title, ylabel='', label=False, field=False, wide=False, end=None):
    ax.set_xlim(START, STOP)
    # Label the original alternate days, with a tick at every daily-cell centre.
    ax.set_xticks(MID[::2])
    ax.set_xticks(MID, minor=True)
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %d' if wide else '%b\n%d'))
    ax.tick_params(which='major', direction='in', width=.65, length=4.0,
                   bottom=True, left=True, top=True, right=True,
                   labelsize=12, pad=3, labelbottom=label)
    ax.tick_params(axis='x', which='minor', direction='in', width=.55,
                   length=2.0, bottom=True, top=True)
    ax.set_title(title, loc='left', fontsize=14, fontweight='normal', pad=6.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, color='black', rotation=90,
                      ha='center', va='center')
        ax.yaxis.set_label_coords(-.063 if wide else (-.138 if field else -.160), .5)
    if label and not wide:
        ax.set_xlabel('UTC date | 2024–2025', fontsize=AXIS_LABEL_SIZE, labelpad=4)
    # The field panels need their ticks above pcolormesh, which otherwise
    # covers inward-facing x and y ticks when the axes are drawn below it.
    ax.set_axisbelow(not field)
    if not field:
        ax.grid(True, which='major', color='#9ca6b0', linewidth=.65, linestyle=':', alpha=.72)
    if wide:
        ax.grid(True, axis='x', which='minor', color='#9ca6b0', linewidth=.4, linestyle=':', alpha=.35)
    line = ax.axvline(SSC, color='#d62728', ls='--', lw=.85, zorder=10)
    line.set_gid('storm_sudden_commencement')
    line = ax.axvline(end, color='#777777', ls='--', lw=.85, zorder=10)
    line.set_gid('storm_main_phase_end')


def plot(chem, particles, hourly, kp, dst, palette, end, compact=False):
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 12,
        'axes.labelsize': AXIS_LABEL_SIZE, 'axes.linewidth': .65, 'axes.titleweight': 'normal',
        'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(WIDTH/2.54, HEIGHT/2.54))
    if compact:
        index_axes = [ax_at(fig, 1.70, 16.70, 18.02, 2.20),
                      ax_at(fig, 1.70, 13.65, 8.02, 2.20),
                      ax_at(fig, 11.70, 13.65, 8.02, 2.20)]
        names = ['Dst', 'AE', 'PCN']
        titles = ['Ring-current disturbance', 'Auroral activity', 'Polar-cap activity']
    else:
        index_axes = [ax_at(fig, x, y, 8.02, 2.20)
                      for y in [16.70, 13.65] for x in [1.70, 11.70]]
        names = ['Kp', 'Dst', 'AE', 'PCN']
        titles = ['Planetary activity', 'Ring-current disturbance', 'Auroral activity', 'Polar-cap activity']
    for i, (ax, name, title) in enumerate(zip(index_axes, names, titles)):
        if name == 'Kp':
            # Each point describes its three-hour UT interval and is placed at
            # the centre of that interval, exactly as in the original bars.
            ax.plot(kp.time + pd.Timedelta(hours=1.5), kp.Kp,
                    color=COLORS[name], lw=INDEX_LINEWIDTH, solid_capstyle='round')
            ax.set(ylim=(0, 9), yticks=[0, 3, 6, 9])
            ylabel = 'Kp'
        elif name == 'Dst':
            ax.plot(dst.time, dst.Dst_nT, color=COLORS[name], lw=INDEX_LINEWIDTH)
            ax.set(ylim=(-250, 50), yticks=[-200, -100, 0, 50])
            ax.axhline(0, color='.6', lw=.55, alpha=.6)
            ylabel = 'Dst (nT)'
        elif name == 'AE':
            ax.plot(hourly.index, hourly.aeValue, color=COLORS[name], lw=INDEX_LINEWIDTH)
            ax.set(ylim=(0, 1600), yticks=[0, 800, 1600])
            ylabel = 'AE (nT)'
        else:
            ax.plot(hourly.index, hourly.PCN, color=COLORS[name], lw=INDEX_LINEWIDTH)
            ax.set(ylim=(-2, 12), yticks=[0, 6, 12])
            ylabel = 'PCN (mV m⁻¹)'
        style(ax, f'({chr(97+i)}) {title}', ylabel, wide=compact and name=='Dst', end=end)

    eep = ax_at(fig, 1.70, 9.90, 18.02, 2.20)
    eep_index = len(index_axes)
    handles = []
    for band in ['40-60', '60-80']:
        for channel, ls, energy in [('e1', '-', 40), ('e3', '--', 287)]:
            group = particles[(particles.band == band) & (particles.variable == 'MEPED_'+channel)]
            values = group.set_index('date').reindex(DAYS.strftime('%Y-%m-%d')).value
            line, = eep.plot(MID, values, color=COLORS[band], ls=ls, lw=1.7,
                marker='o' if channel=='e1' else None, ms=3,
                label=f'{band.replace("-", "–")}°N, >{energy} keV')
            handles.append(line)
            np.testing.assert_array_equal(np.asarray(line.get_ydata()), values.to_numpy())
    values = particles.value.dropna()
    eep.set(yscale='log', ylim=(values.min()/1.5, values.max()*1.5), yticks=[1e2,1e3,1e4,1e5])
    style(eep, f'({chr(97+eep_index)}) EEP proxy (MEPED) | cm⁻² s⁻¹ sr⁻¹',
          'Electron flux', label=True, wide=True, end=end)
    legend = fig.legend(handles=handles, loc='lower right',
        bbox_to_anchor=(19.72/WIDTH,12.25/HEIGHT), ncol=2, frameon=False, fontsize=11,
        handlelength=1.6, handletextpad=.45, columnspacing=1., labelspacing=.30,
        borderpad=0, borderaxespad=0)
    field_axes = []
    for n, (band, field) in enumerate([('40-60','dO3'),('60-80','dO3'),('40-60','dT'),('60-80','dT')]):
        left, y = ([1.70,10.95][n%2], [5.55,1.65][n//2])
        ax = ax_at(fig, left, y, 7.60, 3.05)
        field_axes.append(ax)
        table = chem[chem.band.eq(band)]
        levels = table.groupby('pressure_hPa').reference_height_km.first().sort_values()
        values = table.pivot(index='date',columns='pressure_hPa',values=field).reindex(
            index=DAYS.strftime('%Y-%m-%d'),columns=levels.index).to_numpy()
        old_scale = next(s for s in palette['individual_panel_colour_scales'] if s['panel']==chr(102+n))
        lower,upper = old_scale['colour_limits']
        if field=='dO3':
            cmap=plt.get_cmap('BrBG').copy(); cmap.set_bad('white')
            norm=TwoSlopeNorm(vmin=lower,vcenter=0,vmax=upper)
        else:
            cmap=original.zero_white_cmap(lower,upper,'new_Figure2_'+band)
            norm=Normalize(vmin=lower,vmax=upper)
        height=levels.to_numpy()
        good_levels=table.groupby('pressure_hPa').dO3.count().gt(0)
        zupper=float(levels[good_levels.reindex(levels.index)].max())
        selected=np.isfinite(height)&(height>42)&(height<zupper+8)
        im=ax.pcolormesh(md.date2num(DAYS.append(pd.DatetimeIndex([STOP]))),
            original.edges(height[selected]),np.ma.masked_invalid(values[:,selected].T),
            shading='flat',norm=norm,cmap=cmap)
        title=f'({chr(98+eep_index+n)}) Night {"O₃" if field=="dO3" else "T"} | {band.replace("-","–")}°N'
        style(ax,title,'approx. alt. (km)' if n%2==0 else '',label=n>=2,field=True,end=end)
        ax.set_ylim(50,zupper)
        ax.set_yticks([z for z in [50,60,70,80,90] if z<=zupper])
        ax.tick_params(labelleft=n%2==0)
        bar=fig.colorbar(im,cax=ax_at(fig,left+7.60+.25,y,.20,3.05),
                        ticks=old_scale['colourbar_ticks'],extend='neither')
        bar.ax.tick_params(direction='in',labelsize=10,pad=2,width=.6,length=2.5)
        bar.ax.set_title('ΔO₃ (%)' if field=='dO3' else 'ΔT (K)',fontsize=10,pad=8,fontweight='normal')
        # Daily labels remain centred in their original daily cells.
        coords=im.get_coordinates()
        np.testing.assert_allclose((coords[0,:-1,0]+coords[0,1:,0])/2,md.date2num(MID))

    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    clipped=[]
    ghost=set()
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not min(limits)<=tick.get_loc()<=max(limits): ghost.update([tick.label1,tick.label2])
    for text in fig.findobj(Text):
        if text in ghost or not text.get_visible() or not text.get_text().strip(): continue
        bb=text.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:
            clipped.append(text.get_text())
    assert not clipped,clipped
    lb=legend.get_window_extent(renderer)
    assert lb.y0>eep.bbox.y1 and lb.y1<index_axes[-1].bbox.y0
    assert not lb.overlaps(eep._left_title.get_window_extent(renderer))
    for ax in index_axes+[eep]+field_axes:
        if ax.get_ylabel(): assert ax.yaxis.label.get_fontsize()==AXIS_LABEL_SIZE
        if ax.get_xlabel(): assert ax.xaxis.label.get_fontsize()==AXIS_LABEL_SIZE
        assert not ax.patches, 'No event shading should be present'
        assert ax.get_legend() is None, 'No event-line legend should be present'
        events = {line.get_gid(): line for line in ax.lines if line.get_gid()}
        assert set(events)=={'storm_sudden_commencement','storm_main_phase_end'}
        assert all(line.get_linestyle()=='--' for line in events.values())
        for key, timestamp in [('storm_sudden_commencement', SSC), ('storm_main_phase_end', end)]:
            np.testing.assert_allclose(md.date2num(events[key].get_xdata()),md.date2num([timestamp,timestamp]))
        shown_ticks=np.sort(np.r_[ax.get_xticks(),ax.get_xticks(minor=True)])
        np.testing.assert_allclose(shown_ticks,md.date2num(MID))
    for ax in field_axes:
        assert ax.xaxis.get_zorder()>ax.collections[0].get_zorder()
        assert ax.yaxis.get_zorder()>ax.collections[0].get_zorder()
    for ax in index_axes:
        assert ax.lines[0].get_linewidth()==INDEX_LINEWIDTH
    stem='Figure2_three_indices' if compact else 'Figure2_four_indices'
    path=OUT/(stem+'.png')
    fig.savefig(OUT/(stem+'_preview.png'),dpi=150)
    fig.savefig(path,dpi=600)
    with Image.open(path) as image: pixels=list(image.size)
    assert abs(pixels[0]-WIDTH/2.54*600)<2
    plt.close(fig)
    return dict(path=str(path),pixels=pixels,width_cm=WIDTH,height_cm=HEIGHT,dpi=600,
                indices=names,clipped_text=clipped)


def main():
    OUT.mkdir(exist_ok=True)
    chem,particles,hourly,kp,dst,palette,hashes,minima,end=load()
    figures=[plot(chem,particles,hourly,kp,dst,palette,end,compact=False)]
    for name,value in hashes.items(): assert sha(SOURCE/name)==value
    methods=dict(figures=figures,all_x_y_axis_titles_pt=AXIS_LABEL_SIZE,
        geomagnetic_index_linewidth_pt=INDEX_LINEWIDTH,
        Kp_display='unsmoothed native 3-hour values; lines join interval centres',
        EEP_band_colours={key:COLORS[key] for key in ['40-60','60-80']},
        SSC_UTC=str(SSC),main_phase_end_UTC=str(end),
        event_markers='red dashed SSC; grey dashed main-phase end; no shading and no event legend',
        shaded_interval=False,event_line_legend=False,
        hourly_Dst_minimum_nT=float(minima.Dst_nT.iloc[0]),
        Dst_equal_minimum_centres_UTC=minima.time.astype(str).tolist(),
        main_phase_end_does_not_define_end_of_all_external_forcing=True,
        old_80percent_recovery_line_removed=True,
        baseline='2024-12-28 through 2024-12-30; unchanged',
        all_input_values_and_missing_masks_unchanged=True,no_smoothing=True,no_gap_filling=True,
        daily_values_and_ticks_centred_UTC_hour=12,PDF_generated=False,input_sha256=hashes,
        daily_tick_lines_all_panels=True,date_labels_every_days=2,
        major_tick_length_pt=4.0,unlabelled_daily_minor_tick_length_pt=2.0,
        field_ticks_drawn_above_colour_mesh=True,
        identical_event_times_across_all_panels=True,
        sources={
            'SSC':'https://www.usgs.gov/programs/geomagnetism/science/severe-magnetic-storm-december-31-2024',
            'event_paper':'https://doi.org/10.1029/2025GL116542',
            'storm_phase_definition':'https://doi.org/10.1029/2008JA013228',
            'Kp_definition':'https://doi.org/10.1029/2020SW002641',
            'AE_definition':'https://www.ngdc.noaa.gov/geomag/indices/ae.html',
            'PCN_definition':'https://www.space.dtu.dk/english/research/scientific_data_and_models/world_data_center_for_geomagnetism/pcn-index',
            'PC_energy_proxy':'https://doi.org/10.1002/2014JA019940'})
    (OUT/'data_and_layout_checks.json').write_text(json.dumps(methods,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'图2修改与指数时段说明.md').write_text('''# 图2修改与指数、时段说明

本次使用figure2_baseline_Dec28_Dec30_20261022中的原始绘图表，不重新下载或重算臭氧、温度、粒子和地磁指数。

- 所有横纵轴标题统一为13；子图标题、刻度与色标范围延续原图。
- 保留Kp、Dst、AE、PCN四个指数；四条指数折线线宽统一为1.6 pt。
- 四指数版的Kp改为原生3小时数据折线，点位于各3小时区间中点，不插值、不平滑；Kp不是小时瞬时观测。
- EEP：40–60°N为蓝色#0072B2，60–80°N为橙红色#D55E00；能道仍以实线/虚线区分。
- 红色虚线为USGS报道的SSC（2024-12-31 16:18 UTC）；灰色虚线为主相结束（2025-01-01 17:00 UTC）。去除浅金色阴影，两条事件线均不设置图例。
- 九个面板使用完全相同的红、灰事件时刻和时间范围。EEP面板横跨两列，面板宽度不同造成事件线在画布上的横向位置不同，事件时间本身一致。
- 每个面板均在每日12 UTC绘制向内的日刻度线，日期标签仍只标原来的隔日日期。下两行将坐标轴层级置于填色之上，显示横纵轴刻度线。
- 日期标签对应的主刻度线长4pt，其间无标签的逐日次刻度线长2pt。
- 本地OMNI小时Dst最低值为−212 nT，等低值样本在2025-01-01 15:30和16:30 UTC；数据时间为小时区间中点，最后一段的右边界为17:00 UTC。
- 主相结束时刻也与Jiao et al. (2025)对本次磁暴的描述一致。他们还指出1月1日09–17 UTC为Dst快速下降阶段。
- 文献常以Dst下降开始至谷值定义主相；SSC与主相开始可以相隔一个初相。本图红线标记SSC，灰线标记主相结束。
- 原图1月3日“Dst恢复80%”竖线已删除。恢复期的AE、PCN和EEP仍可增强，17 UTC之后不等于所有外部输入停止。
- 按用户要求保留四个指数：Dst交代磁暴强度和阶段，AE交代极光电集流活动，PCN交代北半球极区能量输入背景，Kp提供全球地磁活动背景。
- PCN是能量输入相关代理，不等于本研究区域的焦耳加热率；AE也不是电子能量沉降的直接测量。实际电子通量由MEPED代理面板展示。
- 臭氧和温度基准仍为2024-12-28至30日，各本征气压层分别计算；缺测不填补，75 km附近已有掩码保留。
- PNG宽20 cm，600 dpi；不导出PDF。旧脚本与旧图保留。

## 方法来源

- Jiao et al. (2025), https://doi.org/10.1029/2025GL116542 ：本次元旦磁暴的Dst/Kp图和时间描述。
- Zhang et al. (2008), https://doi.org/10.1029/2008JA013228 ：磁暴主相及多谷值事件的定义。
- Matzka et al. (2021), https://doi.org/10.1029/2020SW002641 ：Kp定义与原生3小时时段。
- USGS： https://www.usgs.gov/programs/geomagnetism/science/severe-magnetic-storm-december-31-2024
- AE： https://www.ngdc.noaa.gov/geomag/indices/ae.html
- PCN： https://www.space.dtu.dk/english/research/scientific_data_and_models/world_data_center_for_geomagnetism/pcn-index
- Troshichev et al. (2014), https://doi.org/10.1002/2014JA019940 ：PC指数与太阳风输入的关系。
''',encoding='utf-8')
    print(json.dumps(methods,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
