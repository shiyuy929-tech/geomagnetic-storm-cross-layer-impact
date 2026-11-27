"""Wave-phase / Z-envelope diagnostics and same-level harmonic reconstruction.

Preserves the clean multilevel figure. Reconstruction quantifies circulation
geometry, not an independently forced experiment or a causal percentage.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'plot_packages'),str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, NullLocator
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import plot_figure4_wave_vortex_vertical_20261027 as clean

shared=clean.shared
legacy=clean.previous
DAYS,PRE,EVENT=clean.DAYS,clean.PRE,clean.EVENT
OUT=ROOT/'results/Figure4_wave_vortex_reconstruction_20261027'
UPPER_PRESSURE,LOWER_PRESSURE=3,50
SECTOR_WEST,SECTOR_EAST=-130.,-60.
SECTOR_LABEL='60–130°W'
PC=ccrs.PlateCarree()


def sector_mask(lon):
    return (lon>=SECTOR_WEST)&(lon<=SECTOR_EAST)


def load_sector_envelopes(source,waves):
    # Preserve the established contour extraction; only change the meridians
    # over which the resulting daily outer boundaries are averaged.
    clean.load_z_envelopes(source,waves)
    with np.load(OUT/'Z_envelope_daily_by_longitude.npz') as cache:
        edges=cache['edge_latitude']
        lon=cache['longitude']
    select=sector_mask(lon)
    assert np.isfinite(edges[:,:,select]).all(), 'New sector needs complete daily boundaries'
    envelope=edges[:,:,select].mean(axis=-1)
    table=pd.read_csv(OUT/'Z_envelope_multilevel_daily.csv')
    for j,pressure in enumerate(source['pressure']):
        rows=table.pressure_hPa.eq(pressure)
        assert rows.sum()==len(DAYS)
        table.loc[rows,'NA_mean_edge_latitude_degN']=envelope[:,j]
        table.loc[rows,'valid_sector_meridians']=int(select.sum())
        table.loc[rows,'total_sector_meridians']=int(select.sum())
    table['sector_west_degE']=SECTOR_WEST
    table['sector_east_degE']=SECTOR_EAST
    table.to_csv(OUT/'Z_envelope_multilevel_daily.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame([dict(pressure_hPa=float(pressure),
        reference_height_km=float(waves['era_height'][j]),
        pre_mean_latitude_degN=float(envelope[PRE,j].mean()),
        Jan07_09_latitude_degN=float(envelope[EVENT,j].mean()),
        southward_shift_deg=float(envelope[PRE,j].mean()-envelope[EVENT,j].mean()))
        for j,pressure in enumerate(source['pressure'])])
    summary.to_csv(OUT/'Z_envelope_multilevel_periods.csv',index=False,encoding='utf-8-sig')
    shared.QA['multilevel_Z_envelope'].update(
        statistic=f'mean equatorward boundary latitude across {SECTOR_LABEL}; not sector minimum',
        sector_longitude_bounds_degE=[SECTOR_WEST,SECTOR_EAST],
        sector_meridians=int(select.sum()),minimum_valid_meridians=int(select.sum()),
        contour_extraction_unchanged=True)
    return envelope


def mean_edge(field,threshold,lat,lon,weights):
    core,fraction=shared.region.main_component(field<=threshold,weights)
    if core[0].any():
        raise ValueError('The low-Z component reaches the southern data limit')
    edge=shared.region.boundary_latitude(field,core,threshold,lat)
    sector=sector_mask(lon)
    if not np.isfinite(edge[sector]).all():
        raise ValueError(f'Reconstructed edges must use all valid {SECTOR_LABEL} sector meridians')
    return float(edge[sector].mean()),edge,core,fraction


def wave_fields(field,lon):
    """Z0, Z1, Z2 on each latitude; Fourier coefficients retain their phase."""
    lam=np.deg2rad(lon)
    z0=np.broadcast_to(field.mean(-1,keepdims=True),field.shape).copy()
    pieces=[2*np.real(np.mean(field*np.exp(-1j*k*lam),axis=-1,keepdims=True)
                      *np.exp(1j*k*lam)) for k in [1,2]]
    spectrum=np.fft.rfft(field,axis=-1)
    spectrum[...,3:]=0
    independent=np.fft.irfft(spectrum,n=len(lon),axis=-1)
    np.testing.assert_allclose(z0+pieces[0]+pieces[1],independent,rtol=0,atol=1e-9)
    return z0,*pieces


def reconstruction(source,waves,envelope):
    lat,lon,p=source['lat'],source['lon'],source['pressure']
    weights=shared.base.cell_weights(lat,len(lon))
    result={};rows=[];metrics=[];sensitivity=[]
    for pressure in [UPPER_PRESSURE,LOWER_PRESSURE]:
        j=np.flatnonzero(p==pressure)[0]
        threshold=float(source['z'][PRE,j,np.flatnonzero(lat==60)[0]].mean())
        actual=np.stack([shared.base.smooth_map(x,.4) for x in source['z'][:,j]])
        z0,z1,z2=wave_fields(actual,lon)
        fields={'actual':actual,'W0':z0,'W0_W1':z0+z1,'W0_W1_W2':z0+z1+z2}
        series={name:[] for name in fields}
        edges=[];geometries=[]
        for name,full in fields.items():
            for i,field in enumerate(full):
                value,edge,core,fraction=mean_edge(field,threshold,lat,lon,weights)
                series[name].append(value)
                rows.append(dict(date=str(DAYS[i].date()),pressure_hPa=pressure,reconstruction=name,
                    fixed_threshold_m=threshold,reference_latitude_N=60,NA_mean_latitude_degN=value,
                    valid_sector_meridians=int(sector_mask(lon).sum()),
                    sector_west_degE=SECTOR_WEST,sector_east_degE=SECTOR_EAST,
                    retained_component_area_fraction=float(fraction)))
                if name=='actual':
                    if not np.isfinite(edge).all():
                        raise ValueError('Period-map radial averages need full longitude coverage')
                    edges.append(edge)
                    geometries.append(shared.geometry(core,lat,lon,weights))
        series=pd.DataFrame(series,index=DAYS)
        # Panel a and panel e use identical source, smoothing, threshold and edge statistic.
        np.testing.assert_allclose(series.actual,envelope[:,j],rtol=0,atol=1e-10)
        height=float(waves['era_height'][j])
        for name in series:
            pre=float(series[name][PRE].mean());event=float(series[name][EVENT].mean())
            error=series[name]-series.actual
            denominator=float(np.sum((series.actual-series.actual.mean())**2))
            metrics.append(dict(pressure_hPa=pressure,reference_height_km=height,reconstruction=name,
                pre_mean_latitude_degN=pre,Jan07_09_mean_latitude_degN=event,
                southward_shift_deg=pre-event,RMSE_deg=float(np.sqrt(np.mean(error**2))),
                mean_bias_deg=float(error.mean()),
                geometric_R2=float(1-np.sum(error**2)/denominator)))
        # Same reconstruction experiment with nearby reference contours, for diagnostic robustness.
        for ref in [55,65]:
            threshold_other=float(source['z'][PRE,j,np.flatnonzero(lat==ref)[0]].mean())
            values={name:np.array([mean_edge(f,threshold_other,lat,lon,weights)[0] for f in full])
                    for name,full in fields.items()}
            for name,value in values.items():
                sensitivity.append(dict(pressure_hPa=pressure,reference_latitude_N=ref,reconstruction=name,
                    fixed_threshold_m=threshold_other,pre_mean_latitude_degN=float(value[PRE].mean()),
                    Jan07_09_mean_latitude_degN=float(value[EVENT].mean()),
                    southward_shift_deg=float(value[PRE].mean()-value[EVENT].mean()),
                    RMSE_deg=float(np.sqrt(np.mean((value-values['actual'])**2)))))
        edges=np.stack(edges)
        for title,select in [('Dec28-30',PRE),('Jan7-9',EVENT)]:
            sector=sector_mask(lon)
            np.testing.assert_allclose(edges[select].mean(0)[sector].mean(),series.actual[select].mean(),
                                       rtol=0,atol=1e-10)
        result[pressure]=dict(series=series,edges=edges,geometry=pd.DataFrame(geometries),
            threshold=threshold,height=height,W1_mean=z1[EVENT].mean(0),W1_daily=z1)
        pd.DataFrame(edges,index=DAYS,columns=[f'{x:g}' for x in lon]).to_csv(
            OUT/f'Z_{pressure}hPa_daily_edge_by_longitude.csv',index_label='date',encoding='utf-8-sig')
    frame=pd.DataFrame(rows);score=pd.DataFrame(metrics)
    frame.to_csv(OUT/'edge_reconstruction_daily.csv',index=False,encoding='utf-8-sig')
    score.to_csv(OUT/'edge_reconstruction_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(OUT/'edge_threshold_sensitivity.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(OUT/'same_level_reconstructed_fields.npz',time=DAYS.to_numpy(dtype='datetime64[D]'),
        latitude=lat,longitude=lon,pressure_hPa=np.array([UPPER_PRESSURE,LOWER_PRESSURE]),
        W1_height_component=np.stack([result[k]['W1_daily'] for k in [UPPER_PRESSURE,LOWER_PRESSURE]],axis=1))
    shared.QA['reconstruction']=dict(pressures_hPa=[UPPER_PRESSURE,LOWER_PRESSURE],
        field='daily ERA5 geopotential height, lightly smoothed with sigma 0.4 grid cells',
        definition='Z0(latitude,time) plus observed longitudinal Fourier components Z1 and Z2 at every latitude',
        thresholds='same fixed Dec28-30 60N zonal-mean Z threshold for actual and all reconstructions',
        phase_and_amplitude='both retained; the wave is not a prescribed northward or southward wind',
        boundary=f'same largest periodic low-Z component and interpolated outer edge, {SECTOR_LABEL} mean',
        sector_longitude_bounds_degE=[SECTOR_WEST,SECTOR_EAST],
        sector_meridians=int(sector_mask(lon).sum()),
        actual_matches_panel_a=True,period_maps_match_daily_metric=True,
        W0='observed daily zonal mean, time-varying; not held at its pre-period value',
        independent_FFT_check=True,causal_experiment=False,
        percentage_attribution=False,nearby_reference_latitudes=[55,60,65])
    return result,score


def pressure_altitude_axes(ax,heights,pressures,edges,panel):
    """Keep the field's reference-height coordinate, calibrate both side scales."""
    logp_from_height=interp1d(heights,np.log(pressures),bounds_error=False,
                            fill_value='extrapolate',assume_sorted=True)
    height_from_logp=interp1d(np.log(pressures)[::-1],heights[::-1],bounds_error=False,
                            fill_value='extrapolate',assume_sorted=True)
    def height_to_pressure(value):
        return np.exp(np.clip(logp_from_height(np.asarray(value)),-700,700))
    def pressure_to_height(value):
        return height_from_logp(np.log(np.maximum(np.asarray(value),1e-12)))
    radius_km=shared.base.RADIUS/1000
    def height_to_altitude(value):
        value=np.asarray(value)
        return radius_km*value/(radius_km-value)
    def altitude_to_height(value):
        value=np.asarray(value)
        return radius_km*value/(radius_km+value)
    np.testing.assert_allclose(pressure_to_height(height_to_pressure(heights)),heights,atol=1e-10,rtol=0)
    np.testing.assert_allclose(altitude_to_height(height_to_altitude(heights)),heights,atol=1e-10,rtol=0)
    ax.set_ylim(edges[0],edges[-1]);ax.set_yticks([]);ax.set_ylabel('')
    clean.time_axis(ax,'#c74336')
    ax.tick_params(axis='y',left=False,right=False,labelleft=False,labelright=False)
    left=ax.secondary_yaxis('left',functions=(height_to_pressure,pressure_to_height))
    choices=np.array([1000,500,150,50,10,3,1,.5])
    positions=pressure_to_height(choices)
    choices=choices[(positions>=edges[0])&(positions<=edges[-1])]
    left.set_yticks(choices);left.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    left.yaxis.set_minor_locator(NullLocator())
    left.set_ylabel('pressure (hPa)',labelpad=3,fontsize=12.2)
    left.tick_params(axis='y',direction='in',labelsize=11,length=3,pad=2,
                     left=True,right=False,labelleft=True,labelright=False)
    right=ax.secondary_yaxis('right',functions=(height_to_altitude,altitude_to_height))
    low,high=height_to_altitude(np.array([edges[0],edges[-1]]))
    ticks=np.arange(0,56,5);ticks=ticks[(ticks>=low)&(ticks<=high)]
    right.set_yticks(ticks);right.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    right.yaxis.set_minor_locator(NullLocator())
    right.set_ylabel('approx. alt. (km)',labelpad=1.5,fontsize=11.2)
    right.tick_params(axis='y',direction='in',labelsize=10.5,length=3,pad=1,
                      left=False,right=True,labelleft=False,labelright=True)
    # SecondaryAxis defaults to zorder 0, below the pcolormesh (zorder 1).
    # Raise the whole axes, not just individual tick artists, so inward ticks
    # are drawn over the field while the transparent axes leave it visible.
    for side in [left,right]:
        side.set_zorder(15)
        side.patch.set_visible(False)
        side.tick_params(axis='y',width=.8,color='#202020')
    shared.QA.setdefault('dual_vertical_axes',{})[panel]=dict(
        left='pressure (hPa); native pressure/reference-height calibration, linear in log(p) between native centers',
        right='approx. alt. (km); approximate reference geometric altitude above sea level, z=R*H/(R-H)',
        reference='Dec28-30 zonal-mean GPH in 60-80N; not instantaneous geometric altitude',
        native_pressure_hPa=pressures.tolist(),native_reference_GPH_km=heights.tolist(),
        native_reference_altitude_km=height_to_altitude(heights).tolist(),
        plotted_reference_height_bounds_km=edges.tolist(),
        pressure_ticks_hPa=choices.tolist(),altitude_ticks_km=ticks.tolist(),
        roundtrip_check=True,field_resampling=False,secondary_axes_zorder=15,
        secondary_axes_background=False,inward_ticks_drawn_above_field=True)


def top_colorbar(fig,ax,image,ticks,labels,title,left):
    bar=clean.add_colorbar(fig,ax,image,ticks,labels,title)
    bar.ax.set_title(title,fontsize=11,pad=8)
    bar.ax.tick_params(labelsize=10)
    pos=bar.ax.get_position()
    bar.ax.set_position([left,pos.y0,pos.width,pos.height])
    return bar


def top_row(fig,waves,source,envelope):
    """Show the Z-envelope evolution first, then the W1 trough longitude."""
    # Keep each panel's outer footprint and colorbar fixed; reserve internal
    # width for the newly requested right-hand altitude scale.
    # Keep the top edge fixed while shortening the upper panels from below;
    # the bottom panels and the overall canvas keep their existing positions.
    a=fig.add_axes([.088,.572,.318,.350])
    b=fig.add_axes([.594,.572,.277,.350])
    xe=clean.cell_edges(mdates.date2num(DAYS.to_pydatetime()))
    full_y=clean.cell_edges(waves['height'])
    # Restore the original lower display limit without changing finite phases.
    selected=np.flatnonzero(waves['height']>=clean.MIN_DISPLAY_HEIGHT_KM)
    assert np.all(np.diff(selected)==1)
    ya=full_y[selected[0]:selected[-1]+2]
    image=b.pcolormesh(xe,ya,waves['longitude'][:,selected].T,cmap=clean.trough_colormap(),
                      norm=Normalize(-180,180),shading='flat',rasterized=True)
    bar=top_colorbar(fig,b,image,[-180,-90,0,90,180],['180°','90°W','0°','90°E','180°'],'Longitude',.932)
    bar.ax.plot([-.45,-.45],[SECTOR_WEST,SECTOR_EAST],color=legacy.GREEN,lw=2,clip_on=False)
    b.set_title('(b) W1 trough longitude',loc='left',pad=20)
    b.text(0,1.015,'60–80°N',transform=b.transAxes,fontsize=11,ha='left',va='bottom')
    pressure_altitude_axes(b,waves['height'][selected],waves['p'][selected],ya,'b')
    order=np.argsort(waves['era_height']);height=waves['era_height'][order]
    full_yb=clean.cell_edges(height)
    use=np.flatnonzero(height>=clean.MIN_DISPLAY_HEIGHT_KM)
    assert np.all(np.diff(use)==1)
    yb=full_yb[use[0]:use[-1]+2]
    visible_envelope=envelope[:,order[use]]
    vmin=min(40.,10*np.floor(np.nanmin(visible_envelope)/10))
    vmax=max(80.,10*np.ceil(np.nanmax(visible_envelope)/10))
    image=a.pcolormesh(xe,yb,np.ma.masked_invalid(envelope[:,order[use]].T),cmap='YlGnBu_r',norm=Normalize(vmin,vmax),
                      shading='flat',rasterized=True)
    top_colorbar(fig,a,image,np.arange(vmin,vmax+1,10),None,'Latitude (°N)',.458)
    a.set_title('(a) N. American Z envelope',loc='left',pad=20)
    a.text(0,1.015,SECTOR_LABEL,transform=a.transAxes,fontsize=11,ha='left',va='bottom')
    pressure_altitude_axes(a,height[use],source['pressure'][order[use]],yb,'a')
    era=selected[waves['sources'][selected]=='ERA5'];seam=full_y[era[-1]+1]
    b.axhline(seam,color='.25',ls='--',lw=.7,zorder=6)
    b.text(.025,(seam-ya[0]+.8)/(ya[-1]-ya[0]),'MLS',transform=b.transAxes,
           fontsize=10.5,ha='left',va='bottom')
    b.text(.025,(seam-ya[0]-.8)/(ya[-1]-ya[0]),'ERA5',transform=b.transAxes,
           fontsize=10.5,ha='left',va='top')
    return a,b


def wave_envelope_map(fig,ax,source,diagnostic,pressure,letter,color):
    # The established renderer averages daily outer boundaries, with no change
    # of operation order. Explicit units avoid conflating pressure with theta.
    centers=legacy.draw_map(ax,source,diagnostic,diagnostic['edges'],pressure,letter,color,
                            surface_unit='hPa',fill_core=False,
                            coordinate_fontsize=10.5,coordinate_color='black')
    for line in list(ax.lines):
        if line.get_color()==legacy.GREEN and line.get_linestyle()==':':
            line.remove()
    ax.set_title(ax.get_title(loc='left'),loc='left',pad=14,fontsize=13.5)
    ax.get_legend().set_bbox_to_anchor((.5,-.07))
    for text in ax.get_legend().get_texts():
        text.set_fontsize(10.5)
    for text in ax.texts:
        if '°' in text.get_text():
            text.set_color('black')
            text.set_fontsize(10.5)
            if text.get_text() in ['40°N','60°N']:
                text.set_bbox(None)
    for line in ax.lines:
        if len(line.get_xdata())==len(source['lon'])+1:
            line.set_path_effects([path_effects.Stroke(linewidth=2.5,foreground='white'),path_effects.Normal()])
    field=diagnostic['W1_mean']
    step=250 if abs(field).max()>1000 else 100
    limit=float(np.ceil(abs(field).max()/step)*step)
    cyclic,lon=add_cyclic_point(field,coord=source['lon'])
    image=ax.pcolormesh(lon,source['lat'],cyclic,transform=PC,cmap='RdBu_r',
                       norm=Normalize(-limit,limit),shading='auto',alpha=.70,zorder=0,rasterized=True)
    # Compact independent scales preserve the much weaker lower-level wave.
    pos=ax.get_position()
    ca=fig.add_axes([pos.x0+.012,.038,pos.width-.024,.008])
    bar=fig.colorbar(image,cax=ca,orientation='horizontal',ticks=[-limit,0,limit])
    bar.ax.tick_params(labelsize=10,direction='in',length=2,pad=2)
    fig.text(pos.x1-.001,.042,'m',fontsize=10.5,ha='left',va='center')
    shared.QA.setdefault('map_fields',{})[str(pressure)]=dict(
        shading='Jan7-9 mean observed GPH W1 component at each latitude; red=ridge, blue=trough',
        units='m',color_limits_m=[-limit,limit],
        sector_meridian_markers_displayed=False,
        edges='equal-day means of daily actual fixed-Z outer boundaries, not reconstructed boundaries',
        centroids=centers,coordinate=f'{pressure} hPa')
    return centers


def plot(waves,source,envelope,recon):
    height_cm=14.8
    fig=plt.figure(figsize=(20/2.54,height_cm/2.54))
    a,b=top_row(fig,waves,source,envelope)
    width=.205;height=20*width/height_cm
    c=fig.add_axes([.032,.112,width,height],projection=ccrs.NorthPolarStereo(central_longitude=0))
    d=fig.add_axes([.275,.112,width,height],projection=ccrs.NorthPolarStereo(central_longitude=0))
    e=fig.add_axes([.570,.112,.415,height])
    for ax,pressure,letter,color in [(c,UPPER_PRESSURE,'c',legacy.PURPLE),(d,LOWER_PRESSURE,'d',legacy.BLUE)]:
        wave_envelope_map(fig,ax,source,recon[pressure],pressure,letter,color)
    series=recon[UPPER_PRESSURE]['series']
    for name,color,style,width,label in [
        ('actual','#202020','-',2.15,'Actual Z envelope'),
        ('W0_W1',legacy.BLUE,'-',1.45,'W0 + W1'),
        ('W0_W1_W2',legacy.ORANGE,'--',1.5,'W0 + W1 + W2')]:
        e.plot(DAYS,series[name],color=color,ls=style,lw=width,label=label)
    e.set_ylim(48,72);e.set_yticks([50,55,60,65,70])
    e.set_ylabel('Latitude (°N)',labelpad=3)
    e.set_title(f'(e) Z-edge reconstruction ({UPPER_PRESSURE} hPa)',loc='left',pad=14)
    clean.time_axis(e,'#c74336')
    e.grid(axis='y',color='.85',ls=':',lw=.6,zorder=-3)
    e.legend(loc='upper right',fontsize=10.5,frameon=False,handlelength=1.7,
             handletextpad=.4,borderaxespad=.3,labelspacing=.3)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for parent,colorbar_left in [(a,.458),(b,.932)]:
        for side in parent.child_axes:
            if side.get_zorder()<=max(item.get_zorder() for item in parent.collections):
                raise RuntimeError('Secondary-axis ticks are hidden below the heatmap')
            visible_marks=[tick.tick1line if side.yaxis.get_label_position()=='left' else tick.tick2line
                           for tick in side.yaxis.get_major_ticks()]
            if not all(mark.get_visible() and mark.get_markersize()>0 for mark in visible_marks):
                raise RuntimeError('A requested vertical-axis tick is missing')
            for text in [side.yaxis.label,*side.get_yticklabels()]:
                if text.get_visible() and text.get_text():
                    box=text.get_window_extent(renderer)
                    if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
                        raise RuntimeError('Clipped secondary-axis text: '+text.get_text())
                    if side.yaxis.get_label_position()=='right' and box.x1>=colorbar_left*fig.bbox.width:
                        raise RuntimeError('Altitude axis overlaps the adjacent colorbar')
    for ax in [a,b,c,d,e]:
        box=ax._left_title.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
            raise RuntimeError('Clipped panel title: '+ax._left_title.get_text())
    shared.QA['figure_semantics']=dict(
        first_row='same multilevel fields, with independent p(hPa) left / reference altitude z(km) right axes',
        panel_a=f'{SECTOR_LABEL} mean North American Z-envelope latitude',panel_b='W1 trough longitude',
        panel_b_colourbar_sector_bounds_degE=[SECTOR_WEST,SECTOR_EAST],
        onset_lines='same red dashed Dec31 16:18 UTC marker in panels a, b and e',
        lower_maps='same-pressure W1 GPH shading plus actual low-Z envelopes; these are not PV boundaries',
        line_panel='3hPa actual and reconstructed absolute edge latitude; no inversion of north/south',
        percent_claim=False,wind_advection_claim=False,temperature_shading=False,
        physical_width_cm=20,dpi=600,PDF_export=False)
    shared.QA['layout_update']=dict(
        upper_panel_positions={name:list(ax.get_position().bounds) for name,ax in [('a',a),('b',b)]},
        lower_panel_positions={name:list(ax.get_position().bounds) for name,ax in [('c',c),('d',d),('e',e)]},
        lower_title_pad_pt=14,map_legend_anchor_y=-.07,map_latitude_label_background=False,
        top_colorbar_left_positions=[.458,.932],
        top_colorbar_tick_fontsize=10,top_colorbar_title_pad_pt=8,
        data_source_label_background=False,
        upper_panels_shortened_with_fixed_top=True,
        upper_bottom_raised_fraction=.025,
        upper_column_gap_increase_cm=.24,lower_section_gap_increase_cm=.20,
        secondary_axis_text_clipped=False,secondary_axis_colorbar_overlap=False,
        inward_secondary_ticks_visible=True)
    shared.QA['restored_start_display']=dict(
        all_native_ERA5_levels_displayed=False,ERA5_source_bottom_hPa=float(source['pressure'].max()),
        minimum_display_reference_height_km=float(clean.MIN_DISPLAY_HEIGHT_KM),
        displayed_ERA5_bottom_hPa=float(np.max(waves['p'][(waves['sources']=='ERA5')&(waves['height']>=clean.MIN_DISPLAY_HEIGHT_KM)])),
        minimum_height_gate=True,all_days_valid_gate=False,phase_amplitude_gate=False,
        sector_phase_sensitivity_gate=False,MLS_daynight_display_gate=False,
        finite_phase_cells=int(np.isfinite(waves['longitude']).sum()),
        total_phase_cells=int(waves['longitude'].size),
        latitude_color_limits_degN=[float(np.min(a.collections[0].get_clim())),float(np.max(a.collections[0].get_clim()))])
    shared.save(fig,'Figure4_wave_vortex_reconstruction.png')


def markdown_table(frame):
    rounded=frame.round(3)
    header='| '+' | '.join(rounded.columns)+' |\n'
    header+='| '+' | '.join(['---']*len(rounded.columns))+' |\n'
    return header+'\n'.join('| '+' | '.join(str(x) for x in row)+' |'
                            for row in rounded.itertuples(index=False,name=None))


def notes(score):
    upper=score[score.pressure_hPa==UPPER_PRESSURE]
    lower=score[score.pressure_hPa==LOWER_PRESSURE]
    text=f'''# W1/W2与北美环流边界：波数重构对比

**图4. 2024年12月28日至2025年1月11日行星波槽位、北美环流包络及位势高度波数重构。**

（a）北美扇区（130°–60°W）实际低位势高度包络平均外边界纬度的时间–高度分布，只显示约10 km以上完整有效的ERA5层，各层阈值在12月28–30日确定并固定。（b）60°–80°N位势高度W1槽位经度的时间–高度分布；ERA5 150–1 hPa和MLS约47–50 km只保留15天全部有效的层，填色是经度，不是振幅，没有新增等值线或反射期框。两图分别使用自己的双纵轴：左侧pressure (hPa)为气压，右侧approx. alt. (km)为海拔参考标尺。左轴按原生层气压与基准期60°–80°N带内参考位势高度H的对应关系标定，原生层之间对log(p)线性换算；右轴用球形地球近似z=R H/(R−H)、R=6371 km，将同一参考位势高度换为海拔。两侧标尺均非当日各经纬度的瞬时几何高度，没有重新采样填色数据；两图的有效高度范围各自独立。日期标在每日方块中心，图（a、b、e）统一以红色竖虚线标示12月31日16:18 UTC。

（c、d）分别为3 hPa（参考高度约36.6 km）和50 hPa（约19.5 km）的1月7–9日平均位势高度W1空间分量，红色为正高度脊、蓝色为负高度槽，单位m；两个小颜色条各自对称定标。它们不是温度图，不能把蓝色直接称为寒潮。灰色虚线/空心点为12月28–30日、彩色实线/实心点为1月7–9日的实际低位势高度包络与面积质心；先逐日提取各经线外边界再等权平均，质心对每日单位向量等权球面平均。绿色点线标示130°W和60°W。

排版保留（c–e）标题下移6 pt、（c、d）图例上移约0.12 cm及纬度标签无白底的设置。缩小左右部分间距：图（b）左边界左移0.60 cm并相应增宽，图（e）左边界左移0.50 cm并相应增宽，右边界保持不动；图（d）整体左移0.20 cm，使两个圆稍微靠近，图（c）位置不变。各图的纵向位置和高度保持不变。第一行两个颜色条沿用此前右移约0.16 cm的位置。左右双纵轴的刻度线向内，整个双轴置于填色图层之上，避免刻度线被填色遮住。

（e）3 hPa同一固定阈值下北美扇区边界的三条曲线：实际完整位势高度场包络（黑线）、W0+W1重构包络（蓝线）、W0+W1+W2重构包络（橙色虚线），均为绝对纬度。每个纬度独立对经度做傅里叶分解，W0为当日纬向平均，W1/W2同时保留当日振幅和相位；不使用图（b）的一个纬带平均系数去替代整个二维场。对完整和重构场使用完全相同的空间平滑（sigma=0.4个原网格）、固定阈值、最大周期连通区、线性外边界插值和29条扇区经线。图（e）黑线逐日与图（a）3 hPa层的数值完全一致，图（c）的时段边界也按相同操作顺序计算。

## 对原建议的必要修正

- 10 hPa与1200 K不是同一个面。在本资料里10 hPa基准参考高度约29.0 km；1200 K对应的气压、高度随时空变化，原PV图平均高度约36 km。本图上层明确选3 hPa约36.6 km，下层选50 hPa约19.5 km，避免混用压力面和等熵面。
- 原1200 K、450 K的PV图仍保存在旧版。本版（c、d）改为与GPH波数重构一致的气压面低Z包络，明确属于环流几何代理，不能标成物质PV边界。独立PV诊断应另行引用。
- W0只在经度上对称，它依然随纬度和日期变化；在固定高度阈值下，其边界不必保持60°N不动。实际W0曲线保存在CSV中，没有额外挤入主图。
- 这是对同一资料场的几何分解：重构相近说明低波数结构能复现边界形态，不等于独立施加W1强迫后获得的响应。没有把重构边界称为“仅由W1驱动”，没有预设85%或任何因果贡献率。
- W1/W2的边界位置不可线性相加。加入W2后可以抵消或增强W1产生的几何偏移，必须用重构结果判断，不能由两者振幅大小预先断言。
- 本图没有新画风矢量，不能单凭脊槽填色宣称“槽后北风把边界拉南”。若要证明动力驱动，还需边界输送/法向速度与更完整收支、受控强迫试验。

## 3 hPa重构结果

{markdown_table(upper)}

`southward_shift_deg`=12月28–30日均值减1月7–9日均值，正值表示该扇区平均边界更偏南。RMSE比较15天的绝对纬度，`geometric_R2`只是同一资料场的几何重构拟合分数，不是因果解释率。

## 50 hPa对照

{markdown_table(lower)}

下层没有出现上层同等幅度的南伸，不能据此图写成整层同步下移。附近55°N和65°N参考阈值的敏感性结果另存，不根据拟合优度挑选主图层次或阈值。

## 方法依据

- Chandra et al. (2026), *Regional structure of the polar vortex via planetary wave phase diagnostics and its link to cold air outbreaks*, https://doi.org/10.1038/s41612-026-01418-4 。参考W1/W2振幅、相位和区域结构分开诊断的思路；本图使用原始日高度谐波，不照搬该文的气候态异常和7日平滑。
- Seviour et al. (2013), *A practical method to identify displaced and split stratospheric polar vortex events*, https://doi.org/10.1002/grl.50927 。参考用固定高度轮廓和几何指标描述极涡结构的思路；本文采用事件基准阈值和扇区平均外边界，是明确说明的个例几何诊断，并非完整复现其10 hPa事件分类。

## 文件

- `Figure4_wave_vortex_reconstruction.png`：20 cm宽、600 dpi，PNG，无PDF。
- `edge_reconstruction_daily.csv`：完整场与W0、W0+W1、W0+W1+W2逐日绝对边界纬度。
- `edge_reconstruction_summary.csv`：两层的基准/事件均值、南伸量、误差。
- `edge_threshold_sensitivity.csv`：55°N/65°N参考阈值的补充计算。
- `Z_3hPa_daily_edge_by_longitude.csv`、`Z_50hPa_daily_edge_by_longitude.csv`：地图和时间序列共同使用的逐日各经线外边界。
- `same_level_reconstructed_fields.npz`：两个气压层逐日W1空间分量。
- `method_checks.json`：资料来源哈希、方法、数值一致性与导出检查。

保留的旧版脚本：`plot_figure4_wave_vortex_vertical_20261027.py`。旧版PV诊断与本版Z包络不是同一边界定义。
'''
    first=text.index('（a）')
    lower=text.index('（c、d）')
    updated='''（a）北美扇区（130°–60°W）固定低位势高度包络平均外边界纬度的时间–高度分布，恢复扩展到近地面之前的显示范围：显示参考高度约10 km以上的原ERA5层，本地原网格对应150–1 hPa（最低原生层中心参考位势高度约12.8 km，色块下缘约10.7 km）。所显示各层均采用12月28–30日60°N纬向平均GPH作为各层固定阈值；300 hPa及以下不同参考纬度的包络不再出现在图a中。沿用原有0.4格点空间平滑、最大周期连通低Z区和固定北美扇区的统计规则。

（b）60°–80°N位势高度W1槽位经度，也恢复为原ERA5 150–1 hPa及上方MLS约47–50 km的显示范围。仅恢复高度范围，不重新施加50 m振幅、删扇区、昼夜相位或整层15天有效的显示筛选：高度范围内所有有限复系数均显示。原始MLS观测质量控制及既有谐波拟合不变。弱振幅时相位不稳定，显示出数值不表示传播方向可靠；未做相位插值或补值。

（a、b）左轴pressure (hPa)，右轴approx. alt. (km)，不再显示近地层刻度；两图分别使用自己的显示高度范围。左轴按事件前60°–80°N纬带参考GPH与原生气压的对应关系标定，右轴海拔为球形近似换算，不是当日各经纬度瞬时高度。日期刻度仍位于每日色块中心，红色竖虚线为12月31日16:18 UTC，不显示反射期阴影。图a色标根据恢复后的实际展示数据确定，为40°–80°N。图c、d仍为3 hPa及50 hPa，图e仍为3 hPa重构，三者的数据和排版均不变。

'''
    text=text[:first]+updated+text[lower:]
    text=text.replace('130°–60°W',SECTOR_LABEL)
    text=text.replace('绿色点线标示130°W和60°W。','两幅圆图不绘制北美扇区经度标线。')
    text=text.replace('和29条扇区经线',f"和{shared.QA['reconstruction']['sector_meridians']}条扇区经线")
    text=text.replace('图c、d仍为3 hPa及50 hPa，图e仍为3 hPa重构，三者的数据和排版均不变。',
        f'图c、d仍为3 hPa及50 hPa，包络、质心及W1填色不变，不画绿色扇区经线；图e仍为3 hPa重构，并与图a统一采用{SECTOR_LABEL}扇区平均。')
    (OUT/'图注与重构结果说明.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    clean.OUT=OUT;legacy.OUT=OUT;shared.OUT=OUT;shared.QA={};shared.style()
    plt.rcParams.update({'axes.titlesize':13.5,'axes.labelsize':12.5,
                         'xtick.labelsize':12,'ytick.labelsize':12})
    import figure4_full_depth_data_20261030 as full_depth
    # Use the original native grid so the lower color-cell edge also matches
    # the pre-extension figure, rather than introducing 200/250-hPa cells.
    source=np.load(shared.base.CACHE)
    waves=clean.load_vertical_waves(source)
    full_depth.all_phase(waves,DAYS,shared,OUT)
    envelope=load_sector_envelopes(source,waves)
    recon,score=reconstruction(source,waves,envelope)
    plot(waves,source,envelope,recon)
    shared.QA['windows']={'baseline':'2024-12-28/30','maps':'2025-01-07/09','reflection_highlight':False}
    shared.QA['sources']=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                          for path in [shared.base.CACHE,shared.MLS,shared.COMMON,Path(clean.__file__),Path(legacy.__file__),Path(__file__),Path(full_depth.__file__)]]
    (OUT/'method_checks.json').write_text(json.dumps(shared.QA,ensure_ascii=False,indent=2),encoding='utf-8')
    notes(score)
    print(score.round(4).to_string(index=False),flush=True)
    print('Saved:',OUT,flush=True)


if __name__=='__main__':
    main()
