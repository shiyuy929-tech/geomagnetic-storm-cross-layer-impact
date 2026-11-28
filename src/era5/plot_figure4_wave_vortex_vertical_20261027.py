"""Multilevel W1 longitude and fixed-Z envelopes, with independent PV maps.

Daily native pressure levels are displayed as discrete cells, restricted to
levels valid on every event day. Each upper panel has its own vertical range.
No interpolation of longitude, no normalization, and PNG-only export.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'plot_packages'), str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, LinearSegmentedColormap
import cartopy.crs as ccrs
import plot_figure4_wave_pv_boundary_20261027 as previous

shared = previous.shared
DAYS, PRE, EVENT = shared.DAYS, shared.PRE, previous.EVENT
OUT = ROOT/'results/Figure4_wave_vortex_vertical_20261027'
OLD_OUT = ROOT/'results/Figure4_wave_PV_boundary_20261027'
PHASE_MIN_AMPLITUDE_M = 50.
MIN_DISPLAY_HEIGHT_KM = 10.


def trough_colormap():
    """Warm cyclic key, distinct from the blue-green Z-envelope color scale."""
    colors = ['#6c3b8f','#b74383','#e98fab','#fff1d6',
              '#f2be63','#cf733e','#6c3b8f']
    cmap = LinearSegmentedColormap.from_list('trough_purple_rose_gold',colors,N=257)
    np.testing.assert_allclose(cmap(0.),cmap(1.),rtol=0,atol=1e-12)
    return cmap


def band_mean(a, lat, lower=60, upper=80, axis=-1):
    """Use the same native 5-degree centers as the MLS/ERA5 overlap check."""
    centers = np.arange(lower+2.5, upper, 5.)
    ids = np.array([np.argmin(abs(lat-x)) for x in centers])
    np.testing.assert_allclose(lat[ids], centers)
    return np.average(np.take(a, ids, axis=axis),
                      weights=np.cos(np.deg2rad(centers)), axis=axis)


def load_vertical_waves(source):
    p, lat, lon = source['pressure'], source['lat'], source['lon']
    z = source['z']
    assert np.isfinite(z).all(), 'Do not display an incomplete daily pressure layer'
    np.testing.assert_array_equal(source['time'].astype('datetime64[D]'),
                                  DAYS.to_numpy(dtype='datetime64[D]'))
    coefficient = 2*np.mean(z*np.exp(-1j*np.deg2rad(lon))[None,None,None,:], axis=-1)
    era = band_mean(coefficient, lat)
    height_era = band_mean(z.mean(axis=-1), lat)[PRE].mean(axis=0)/1000
    # Independent cross-check of the phase convention and latitude averaging.
    common = pd.read_csv(shared.COMMON)
    errors = []
    for j, pressure in enumerate(p):
        ref = common[(common.band == '60-80') & (common.pressure_hPa == pressure)]
        if len(ref):
            ref = ref.set_index('date').loc[DAYS.strftime('%Y-%m-%d')]
            expected = ref.ERA5_C_real.to_numpy()+1j*ref.ERA5_C_imag.to_numpy()
            np.testing.assert_allclose(era[:,j], expected, rtol=1e-9, atol=1e-9)
            errors.append(float(np.max(abs(era[:,j]-expected))))

    m = np.load(shared.MLS)
    days_m = m['dates'].astype('datetime64[D]')
    ids = [np.flatnonzero(days_m == np.datetime64(day.date()))[0] for day in DAYS]
    # MLS stores a+i*b for a*cos(lambda)+b*sin(lambda).
    all_c = np.conj(np.take(m['cz'][0,0,...,0], ids, axis=0))
    cm = band_mean(all_c, m['lat'], axis=1)
    hm = band_mean(np.take(m['z0'][0,0], ids, axis=0), m['lat'], axis=1)
    hm = hm[PRE].mean(axis=0)/1000
    anchor = int(np.argmin(abs(hm-50)))
    keep = np.flatnonzero((m['pressure'] >= m['pressure'][anchor]*.999)
                         & (m['pressure'] < 1))
    # Run the existing MLS sector/day-night phase-sensitivity screens per level.
    sector = np.conj(np.take(m['cz'][0,1:,...,0], ids, axis=1))
    sector = band_mean(sector, m['lat'], axis=2)
    branch = np.conj(np.take(m['cz'][1:3,0,...,0], ids, axis=1))
    branch = band_mean(branch, m['lat'], axis=2)
    sector_error = np.nanpercentile(abs(shared.wrap(np.angle(sector*cm.conj()[None],deg=True))),95,axis=0)
    branch_error = np.nanmax(abs(shared.wrap(np.angle(branch*cm.conj()[None],deg=True))),axis=0)
    m_good = (abs(cm)>=PHASE_MIN_AMPLITUDE_M) & (sector_error<=30) & (branch_error<=45)

    pressure = np.r_[p, m['pressure'][keep]]
    height = np.r_[height_era, hm[keep]]
    coefficients = np.c_[era, cm[:,keep]]
    valid = np.c_[abs(era)>=PHASE_MIN_AMPLITUDE_M, m_good[:,keep]]
    sources = np.array(['ERA5']*len(p)+['MLS']*len(keep))
    order = np.argsort(height)
    pressure, height, sources = pressure[order], height[order], sources[order]
    coefficients, valid = coefficients[:,order], valid[:,order]
    assert np.all(np.diff(height)>0)
    longitude = shared.wrap(180-np.angle(coefficients,deg=True))
    longitude[~valid] = np.nan
    assert np.all((-180<=longitude[np.isfinite(longitude)]) & (longitude[np.isfinite(longitude)]<180))
    records = []
    for j in range(len(height)):
        for i, day in enumerate(DAYS):
            records.append(dict(date=str(day.date()),source=sources[j],pressure_hPa=float(pressure[j]),
                reference_height_km=float(height[j]),latitude_band='60-80N',
                C_real=float(coefficients[i,j].real),C_imag=float(coefficients[i,j].imag),
                amplitude_m=float(abs(coefficients[i,j])),phase_valid=bool(valid[i,j]),
                W1_trough_longitude_degE=float(longitude[i,j])))
    pd.DataFrame(records).to_csv(OUT/'W1_trough_multilevel_daily.csv',index=False,encoding='utf-8-sig')
    shared.QA['multilevel_wave'] = dict(
        band='62.5,67.5,72.5,77.5 N; cosine-weighted complex coefficients',
        variable='absolute daily geopotential height, not climatological anomalies',
        convention='Z1=Re(C exp(i lambda)); longitude=wrap(180-arg(C))',
        ERA5_pressure_hPa=p.tolist(),MLS_pressure_hPa=m['pressure'][keep].astype(float).tolist(),
        reference_height='Dec28-30 zonal-mean GPH in the same 60-80N band /1000; fixed display coordinate, not instantaneous geometric altitude',
        longitude_colormap='cyclic purple-magenta-rose-cream-gold-orange-purple; identical endpoint colors; discrete native cells; no phase interpolation',
        amplitude_screen_m=PHASE_MIN_AMPLITUDE_M,
        MLS_screen='each cell: leave-one-sector p95<=30 degrees and day/night max<=45 degrees',
        no_MLS_anchor_gate_on_ERA5=True,
        screened_cells_by_pressure={f'{pp:g}':int((~valid[:,j]).sum()) for j,pp in enumerate(pressure)},
        overlap_coefficient_max_error_m=max(errors),
        vertical_interpolation=False)
    return dict(p=pressure,height=height,sources=sources,coefficients=coefficients,
                valid=valid,longitude=longitude,era_p=p,era_height=height_era)


def latitude_band_sensitivity(source):
    """Compare a fixed band across levels, without changing the main diagnostic."""
    lat,lon,p=source['lat'],source['lon'],source['pressure']
    c=2*np.mean(source['z']*np.exp(-1j*np.deg2rad(lon))[None,None,None,:],axis=-1)
    coefficients={f'{lo}-{hi}':band_mean(c,lat,lo,hi) for lo,hi in [(40,60),(50,70),(60,80)]}
    daily=[];summary=[]
    for band,cc in coefficients.items():
        for j,pp in enumerate(p):
            good=abs(cc[:,j])>=PHASE_MIN_AMPLITUDE_M
            phase=shared.wrap(180-np.angle(cc[:,j],deg=True))
            ref=coefficients['60-80'][:,j]
            both=good & (abs(ref)>=PHASE_MIN_AMPLITUDE_M)
            error=shared.wrap(np.angle(cc[:,j]*ref.conj(),deg=True))
            for i,day in enumerate(DAYS):
                daily.append(dict(date=str(day.date()),latitude_band=band,pressure_hPa=float(pp),
                    C_real=float(cc[i,j].real),C_imag=float(cc[i,j].imag),
                    amplitude_m=float(abs(cc[i,j])),phase_valid=bool(good[i]),
                    trough_longitude_degE=float(phase[i]) if good[i] else np.nan))
            summary.append(dict(latitude_band=band,pressure_hPa=float(pp),
                valid_days=int(good.sum()),
                pre_mean_field_trough_degE=float(shared.wrap(180-np.angle(cc[PRE,j].mean(),deg=True))),
                Jan07_09_mean_field_trough_degE=float(shared.wrap(180-np.angle(cc[EVENT,j].mean(),deg=True))),
                paired_valid_days_vs_60_80=int(both.sum()),
                median_absolute_phase_difference_vs_60_80_deg=float(np.median(abs(error[both]))) if both.any() else np.nan))
    pd.DataFrame(daily).to_csv(OUT/'W1_latitude_band_sensitivity_daily.csv',index=False,encoding='utf-8-sig')
    result=pd.DataFrame(summary)
    result.to_csv(OUT/'W1_latitude_band_sensitivity_summary.csv',index=False,encoding='utf-8-sig')
    shared.QA['latitude_band_choice']=dict(
        main='60-80N at every level; unchanged from prior MLS upper-wave analysis',
        reference='Chandra et al. (2026), doi:10.1038/s41612-026-01418-4: 100hPa anomaly harmonics in 60-80N; not a requirement to use this band at all levels',
        alternatives=['40-60N','50-70N'],
        boundary_note='Band for GPH Fourier coefficients is not a prescribed PV-edge latitude; Z-envelope extraction uses the full 20-90N field',
        selection_note='Alternatives compared on the same days and levels, without selecting a band to maximize the apparent event response')
    return result


def load_z_envelopes(source, waves):
    """Extend existing daily-map Z proxies consistently, not as material PV edges."""
    lat, lon, p = source['lat'], source['lon'], source['pressure']
    weights = shared.base.cell_weights(lat,len(lon))
    sector = (lon>=-130)&(lon<=-60)
    edge_mean = np.full((len(DAYS),len(p)),np.nan)
    rows, summary, boundaries = [], [], {}
    for j,pressure in enumerate(p):
        ref_lat = 50 if pressure>=300 else 60
        threshold = float(source['z'][PRE,j,np.flatnonzero(lat==ref_lat)[0]].mean())
        edges = np.full((len(DAYS),len(lon)),np.nan)
        for i,field in enumerate(source['z'][:,j]):
            smooth = shared.base.smooth_map(field,.4)
            core,fraction = shared.region.main_component(smooth<=threshold,weights)
            if core[0].any():
                raise ValueError(f'{pressure:g} hPa {DAYS[i]} low-Z envelope touches input 20N limit')
            edges[i] = shared.region.boundary_latitude(smooth,core,threshold,lat)
            n = int(np.isfinite(edges[i,sector]).sum())
            if n>=int(np.ceil(sector.sum()*.8)):
                edge_mean[i,j] = np.nanmean(edges[i,sector])
            rows.append(dict(date=str(DAYS[i].date()),pressure_hPa=float(pressure),
                reference_height_km=float(waves['era_height'][j]),reference_latitude_N=ref_lat,
                fixed_Z_threshold_m=threshold,NA_mean_edge_latitude_degN=float(edge_mean[i,j]),
                valid_sector_meridians=n,total_sector_meridians=int(sector.sum()),
                retained_component_area_fraction=float(fraction)))
        boundaries[float(pressure)] = edges
        pre, post = edge_mean[PRE,j].mean(),edge_mean[EVENT,j].mean()
        summary.append(dict(pressure_hPa=float(pressure),reference_height_km=float(waves['era_height'][j]),
                            pre_mean_latitude_degN=float(pre),Jan07_09_latitude_degN=float(post),
                            southward_shift_deg=float(pre-post)))
    assert np.isfinite(edge_mean).all(), 'A missing envelope needs explicit treatment'
    assert np.all((edge_mean>=40)&(edge_mean<=80)), 'Expand the latitude color scale; never silently saturate it'
    # The exact original two PV boundaries remain unchanged in the lower row.
    # Do not replace these Z-envelope numbers by PV values or mix their scales.
    pd.DataFrame(rows).to_csv(OUT/'Z_envelope_multilevel_daily.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'Z_envelope_multilevel_periods.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(OUT/'Z_envelope_daily_by_longitude.npz',time=DAYS.to_numpy(dtype='datetime64[D]'),
                        pressure_hPa=p,latitude=lat,longitude=lon,
                        edge_latitude=np.stack([boundaries[float(pp)] for pp in p],axis=1))
    shared.QA['multilevel_Z_envelope'] = dict(
        definition='largest longitude-periodic connected low-Z component; fixed pre-period Z threshold per pressure level',
        threshold='Dec28-30 mean zonal-mean GPH at 60N for p<=150hPa; 50N for p>=300hPa, matching the original daily vortex maps',
        statistic='mean of linearly interpolated equatorward boundary latitude across 130-60W; not sector minimum',
        smoothing_sigma_grid=.4,resolution_degrees=2.5,
        sector_meridians=int(sector.sum()),minimum_valid_meridians=int(min(x['valid_sector_meridians'] for x in rows)),
        touches_southern_limit=False,material_PV_boundary=False,
        threshold_reference_change_hPa=[150,300],
        interpretation='absolute latitude; different Z thresholds are layer-specific proxies and need not form one material 3D boundary')
    return edge_mean,pd.DataFrame(summary)


def cell_edges(centers):
    centers = np.asarray(centers,dtype=float)
    return np.r_[centers[0]-(centers[1]-centers[0])/2,
                 (centers[:-1]+centers[1:])/2,
                 centers[-1]+(centers[-1]-centers[-2])/2]


def time_axis(ax,onset_color):
    ax.set_xlim(DAYS[0]-pd.Timedelta(hours=12),DAYS[-1]+pd.Timedelta(hours=12))
    ticks = pd.date_range('2024-12-29','2025-01-10',freq='3D')
    ax.set_xticks(ticks);ax.set_xticklabels([d.strftime('%b\n%d') for d in ticks])
    ax.set_xlabel('UTC date | 2024–2025',labelpad=3)
    ax.axvline(shared.ONSET,color=onset_color,ls='--',lw=.9,zorder=7)
    ax.tick_params(top=True,right=True,length=3,pad=2)


def heat_axis(ax,yedges):
    ax.set_ylim(yedges[0],yedges[-1])
    ticks=np.arange(10,51,5)
    ax.set_yticks(ticks[(ticks>=yedges[0])&(ticks<=yedges[-1])])
    ax.set_ylabel('Reference height (km)',labelpad=4)
    time_axis(ax,'white')


def add_colorbar(fig,ax,mesh,ticks,labels,title):
    pos=ax.get_position()
    ca=fig.add_axes([pos.x1+.012,pos.y0,.011,pos.height])
    cb=fig.colorbar(mesh,cax=ca,ticks=ticks)
    if labels is not None: cb.ax.set_yticklabels(labels)
    cb.ax.set_title(title,fontsize=9,pad=5)
    cb.ax.tick_params(labelsize=9,length=2,direction='in',pad=2)
    return cb


def plot(waves,source,pv,pv_edges,budget,edge_mean):
    height_cm=14.8
    fig=plt.figure(figsize=(20/2.54,height_cm/2.54))
    a=fig.add_axes([.067,.547,.371,.375])
    b=fig.add_axes([.595,.547,.323,.375])
    map_width=.19;map_height=20*map_width/height_cm
    c=fig.add_axes([.037,.112,map_width,map_height],projection=ccrs.NorthPolarStereo(central_longitude=0))
    d=fig.add_axes([.294,.112,map_width,map_height],projection=ccrs.NorthPolarStereo(central_longitude=0))
    e=fig.add_axes([.585,.112,.40,map_height])
    xe=cell_edges(mdates.date2num(DAYS.to_pydatetime()))
    full_ye=cell_edges(waves['height'])
    selected_a=np.flatnonzero(waves['valid'].all(axis=0)&(waves['height']>=MIN_DISPLAY_HEIGHT_KM))
    assert len(selected_a)>1 and np.all(np.diff(selected_a)==1), 'Do not bridge an invalid interior height'
    assert np.isfinite(waves['longitude'][:,selected_a]).all()
    ye_a=full_ye[selected_a[0]:selected_a[-1]+2]
    phase_cmap=trough_colormap()
    phase=a.pcolormesh(xe,ye_a,waves['longitude'][:,selected_a].T,
                       cmap=phase_cmap,norm=Normalize(-180,180),shading='flat',rasterized=True)
    longitude_labels=['180°','90°W','0°','90°E','180°']
    add_colorbar(fig,a,phase,[-180,-90,0,90,180],longitude_labels,'Longitude')
    # Show the NA longitude interval beside, rather than over, the cyclic color key.
    bar=fig.axes[-1]
    bar.plot([-.45,-.45],[-130,-60],color=previous.GREEN,lw=2,clip_on=False)
    a.set_title('(a) W1 trough longitude',loc='left',pad=15)
    a.text(0,1.015,'60–80°N',transform=a.transAxes,fontsize=9,ha='left',va='bottom')
    heat_axis(a,ye_a)

    # Panel b has its own ERA5-only height range; never add MLS-only empty rows.
    order_b=np.argsort(waves['era_height'])
    height_b=waves['era_height'][order_b]
    full_yb=cell_edges(height_b)
    selected_b=np.flatnonzero(np.isfinite(edge_mean[:,order_b]).all(axis=0)&(height_b>=MIN_DISPLAY_HEIGHT_KM))
    assert len(selected_b)>1 and np.all(np.diff(selected_b)==1), 'Do not bridge an invalid interior envelope height'
    original_b=order_b[selected_b]
    ye_b=full_yb[selected_b[0]:selected_b[-1]+2]
    assert np.isfinite(edge_mean[:,original_b]).all()
    lat_cmap=plt.get_cmap('YlGnBu_r').copy()
    mesh=b.pcolormesh(xe,ye_b,edge_mean[:,original_b].T,cmap=lat_cmap,
                     norm=Normalize(40,80),shading='flat',rasterized=True)
    add_colorbar(fig,b,mesh,[40,50,60,70,80],None,'Latitude (°N)')
    b.set_title('(b) N. American Z envelope',loc='left',pad=15)
    b.text(0,1.015,'130–60°W',transform=b.transAxes,fontsize=9,ha='left',va='bottom')
    heat_axis(b,ye_b)
    era_indices=selected_a[waves['sources'][selected_a]=='ERA5']
    seam=full_ye[era_indices[-1]+1]
    a.axhline(seam,color='.25',ls='--',lw=.7,zorder=6)
    text_box=dict(facecolor='white',alpha=.78,edgecolor='none',pad=.8)
    a.text(.025,(seam-ye_a[0]+.8)/(ye_a[-1]-ye_a[0]),'MLS',transform=a.transAxes,
           fontsize=8.5,ha='left',va='bottom',bbox=text_box)
    a.text(.025,(seam-ye_a[0]-.8)/(ye_a[-1]-ye_a[0]),'ERA5',transform=a.transAxes,
           fontsize=8.5,ha='left',va='top',bbox=text_box)
    centers={str(th):previous.draw_map(ax,source,pv[th],pv_edges[th],th,letter,col)
             for th,ax,letter,col in [(1200,c,'c',previous.PURPLE),(450,d,'d',previous.BLUE)]}
    for key,col,ls,label in [('observed','.15','-','PV rate'),('total',previous.PURPLE,'--','Advection'),
                             ('w1',previous.BLUE,'-','W1 wind'),('w2',previous.ORANGE,'-.','W2 wind')]:
        e.plot(DAYS,budget[key]/1000,color=col,ls=ls,lw=1.55,label=label)
    e.axhline(0,color='.4',lw=.7)
    e.set_ylabel('PV tendency\n'+r'($10^3$ PVU day$^{-1}$)',labelpad=3)
    e.set_title('(e) PV tendency',loc='left',pad=20)
    time_axis(e,'#c74336')
    e.grid(axis='y',color='.85',ls=':',lw=.6,zorder=-3)
    e.text(.99,1.16,'1200 K',transform=e.transAxes,ha='right',va='bottom',fontsize=9,clip_on=False)
    e.legend(loc='upper right',bbox_to_anchor=(.99,.99),ncol=2,fontsize=8.7,frameon=False,
             columnspacing=.6,handlelength=1.15,handletextpad=.35,borderaxespad=.25)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in [a,b,c,d,e]:
        box=ax._left_title.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
            raise RuntimeError(f'Panel title clipped: {ax._left_title.get_text()}')
    shared.QA['display_vertical_coordinate']=dict(
        coordinate='fixed Dec28-30 reference geopotential height of the 60-80N band at each native pressure',
        native_pressure_hPa=waves['p'].tolist(),reference_height_km=waves['height'].tolist(),
        panel_a=dict(pressure_hPa=waves['p'][selected_a].tolist(),reference_height_km=waves['height'][selected_a].tolist(),
                     cell_edges_km=ye_a.tolist(),all_days_phase_valid=True),
        panel_b=dict(pressure_hPa=source['pressure'][original_b].tolist(),reference_height_km=height_b[selected_b].tolist(),
                     cell_edges_km=ye_b.tolist(),all_days_boundary_valid=True),
        independent_panel_y_ranges=True,MLS_ERA5_seam_km=float(seam),
        reflection_window_markers=False,reflection_window_shading=False,
        instantaneous_height_inferred=False,pressure_level_interpolation=False)
    assert not a.patches and not b.patches and not e.patches, 'Reflection patches must be absent'
    shared.save(fig,'Figure4_wave_vortex_vertical.png')
    return centers


def write_notes(summary):
    rounded=summary.round(2)
    table='| '+' | '.join(rounded.columns)+' |\n'
    table+='| '+' | '.join(['---']*len(rounded.columns))+' |\n'
    table+='\n'.join('| '+' | '.join(f'{x:g}' for x in row)+' |'
                     for row in rounded.itertuples(index=False,name=None))
    text=f'''# 多高度图：图注、读图方法与计算说明

**图4. 2024年12月28日至2025年1月11日行星波槽位与北美环流包络的垂直结构，以及平流层PV边界和水平输送诊断。**

（a）60°–80°N纬度带位势高度W1槽位经度的时间–高度分布。仅显示研究期15天全部通过相位有效性筛查、且参考高度不低于10 km的原生层：ERA5 1、2、3、5、7、10、20、30、50、70、100和150 hPa，上方加入MLS约50 km及其下方相邻原生层。最低保留层150 hPa参考高度约12.8 km；方块下边界约10.7 km。300和500 hPa存在W1振幅不足50 m的日期，整层不再画出；原始数据仍保留在CSV中。相位筛查不插值补值。使用紫—玫红—浅粉—米白—金黄—橙—紫的循环色标，与图（b）的蓝绿色系区分；−180°和180°端点颜色完全相同。色标旁绿色短线标示130°–60°W。

（b）北美扇区（130°–60°W）固定低位势高度包络外边界的平均实际纬度。仅显示参考高度不低于10 km且15天全部有效的ERA5层，即150–1 hPa（参考高度约12.8–44.3 km）；图（b）采用自己的纵轴范围和刻度，未添加MLS层或上方空行。图中每层阈值取12月28–30日60°N纬向平均位势高度的均值，并在整个事件内固定。保留完整20°–90°N资料场内最大经度周期连通低高度区，在每条经线上线性插值最南外边界，然后对北美扇区29条经线求均值；边界不受60°–80°N纬度限制。颜色表示绝对纬度，不是温度，也不是相对基准的变化。该包络是环流几何代理，不是物质PV边界。未显示的300/500 hPa指标仍按原50°N参考定义保存于CSV。

（a、b）纵坐标均以各原生气压层在12月28–30日60°–80°N带内纬向平均位势高度的余弦加权均值（km）作为固定参考高度，但两图各自显示完整有效层的范围；并非将每日气压面当成固定几何高度。每个方块对应一天和一个原生气压层，没有进行时间或垂直相位插值，未跨越无效层连色。日期标在每天方块中心。图（a）上方水平虚线分隔MLS与ERA5。白色竖虚线为12月31日16:18 UTC；第一行不再标反射期框线。

（c、d）原1200 K和450 K的PV主体平均外边界及平均质心，灰色虚线/空心点为12月28–30日，彩色实线/实心点为1月7–9日。先逐日提取固定PV阈值主体和各经线最南外边界，再平均边界纬度；质心对每日面积质心单位向量等权球面平均。与旧版计算完全一致。这两幅是PV诊断，与图（b）的Z包络是不同定义，不要求数值相同。

（e）沿用原1200 K的6小时PV水平输送诊断，未重新改变数值。黑线为欧拉局地PV倾向，紫色虚线为总水平平流倾向；蓝、橙线分别为W1、W2水平风作用于完整PV梯度的分项。它们不是波能量转移，也不是边界法向速度；未显示的均流和其他波数保留于CSV，水平项不构成闭合物理PV收支。不再标反射期阴影或框线，红色竖虚线仍为磁暴开始。

## 60°–80°N的选择及敏感性

图（a）的纬带选择直接衔接此前MLS高纬行星波结果，也借鉴Chandra et al. (2026), https://doi.org/10.1038/s41612-026-01418-4 中对100 hPa、60°–80°N高度异常W1/W2振幅和相位的分析。该论文的范围不等于极涡边界的固定位置，不能据此认定所有高度都必须用同一纬带。本文使用原始日位势高度，没有照搬其气候态异常或7日平滑。

为检验纬带依赖，本次另输出40°–60°N、50°–70°N和60°–80°N在相同日期和气压层上的复系数相位比较及有效日数。主图暂保持60°–80°N，以便比较同一纬带的垂直结构；不能在不同高度无说明地切换纬带，也不按事件响应最大化挑选纬带。图（b）的完整场包络提取与图（a）的纬带平均是不同操作。若研究重点转到中纬度槽和天气，应同时分析40°–60°N，不能由60°–80°N单一纬带代表全部环流。

## 本次图应回答的问题

1. 同一60°–80°N纬带，不同高度的W1槽位是否共同转到北美，而非只比较两个预选高度。
2. 实际低高度包络向南伸展出现在哪些层、何时出现，以及是否连续或同步。图（b）本身没有提供垂直传播方向或因果归因。
3. 原有PV图核实上层与下层几何响应不同，不用Z包络替代PV边界；PV输送诊断仍只用于已声明的开放收支。

## 各气压层的参考高度及Z包络结果

下表保留全部已计算气压层的绝对纬度及事前/事件差值；主图（b）只画约10 km以上的完整层，且没有画成相对量。

{table}

正southward_shift_deg表示1月7–9日的扇区平均边界纬度低于12月28–30日；不同层参考纬度不同，不据此拼成唯一的三维物质边界。

## 输出文件

- `Figure4_wave_vortex_vertical.png`：20 cm宽、600 dpi，未生成PDF。
- `W1_trough_multilevel_daily.csv`：原生层、来源、固定参考高度、复系数、振幅、相位及有效标志。
- `Z_envelope_multilevel_daily.csv`：固定阈值、参考纬度、逐日绝对边界纬度、经线覆盖率。
- `Z_envelope_multilevel_periods.csv`：各层基准期和1月7–9日均值。
- `Z_envelope_daily_by_longitude.npz`：逐日、各气压层、各经线的实际边界纬度。
- `W1_latitude_band_sensitivity_daily.csv`、`W1_latitude_band_sensitivity_summary.csv`：三个纬带的逐日相位、有效性与汇总比较。
- `method_checks.json`：源数据哈希、相位复系数交叉检查、显示层次和导出尺寸。

原始两高度版本及底部PV诊断保留；本脚本和对应多高度图随本次修改更新。
'''
    (OUT/'图注与结果说明.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous.OUT=OUT;shared.OUT=OUT;shared.QA={};shared.style()
    # Preserve the smaller axis lettering requested for the existing figure.
    plt.rcParams.update({'axes.titlesize':11.5,'axes.labelsize':10.5,
                         'xtick.labelsize':10,'ytick.labelsize':10})
    source,pv=shared.load_pv()
    pv_edges={th:previous.daily_outer_boundaries(source,pv[th],th) for th in [1200,450]}
    shared.QA['PV_maps_consistency']=shared.QA.pop('panel_b_maps_consistency')
    for periods in shared.QA['PV_maps_consistency'].values():
        for values in periods.values():
            values['daily_PV_indicator_period_mean_degN']=values.pop('panel_b_period_mean_degN')
    shared.QA['PV_method']['map']='Daily outer boundaries averaged by longitude; daily spherical centroids averaged as unit vectors. Unchanged from previous PV figure.'
    waves=load_vertical_waves(source)
    band_sensitivity=latitude_band_sensitivity(source)
    envelope,summary=load_z_envelopes(source,waves)
    budget_file=OLD_OUT/'PV_1200K_boundary_transport_daily.csv'
    budget=pd.read_csv(budget_file,index_col='date',parse_dates=True).reindex(DAYS)
    assert np.isfinite(budget[['observed','total','w1','w2']].to_numpy()).all()
    np.testing.assert_allclose(budget.total,budget.w0+budget.w1+budget.w2+budget.other,atol=1e-8)
    budget.to_csv(OUT/budget_file.name,index_label='date',encoding='utf-8-sig')
    old_checks=json.loads((OLD_OUT/'method_checks.json').read_text(encoding='utf-8'))
    shared.QA['PV_transport']=old_checks['PV_transport']
    centers=plot(waves,source,pv,pv_edges,budget,envelope)
    shared.QA['map_centers']=centers
    for th in ['1200','450']:
        for new,old in zip(centers[th],old_checks['map_centers'][th]):
            for coordinate in ['centroid_lat','centroid_lon']:
                np.testing.assert_allclose(new[coordinate],old[coordinate],atol=1e-12)
    shared.QA['windows']={'baseline':'2024-12-28/30','reflection_highlight_displayed':False,'map':'2025-01-07/09'}
    shared.QA['sources']=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                          for path in [shared.base.CACHE,shared.MLS,shared.COMMON,budget_file,Path(__file__)]]
    shared.QA['unchanged_lower_row']={'PV_centroids_match_previous':True,'PV_budget_loaded_without_recalculation':True}
    (OUT/'method_checks.json').write_text(json.dumps(shared.QA,ensure_ascii=False,indent=2),encoding='utf-8')
    write_notes(summary)
    print(summary.round(3).to_string(index=False),flush=True)
    print('Latitude sensitivity: 40-60N versus 60-80N',flush=True)
    print(band_sensitivity[(band_sensitivity.latitude_band=='40-60') & band_sensitivity.pressure_hPa.isin([1,3,10,50,100,150,300,500])].round(2).to_string(index=False),flush=True)
    print('Saved:',OUT,flush=True)


if __name__=='__main__':
    main()
