"""Figure 5: Vertical structure of harmonic height changes (ΔZ_W1, ΔZ_W2-4),
regional 500 hPa trough budget, and surface weather context.

Key consistency fix:
- Panels (a) and (b) now display the event geopotential height CHANGES (ΔZ_W1 and
  ΔZ_W2-4: Jan 7-9 minus Dec 28-30 baseline), matching the exact baseline-difference
  definition used in Panel (c) (ΔZ500 terms) and Panel (d) (ΔZ500 map).
- Removing the stationary pre-storm background ridge reveals the continuous vertical
  connection of the negative height change (trough deepening) over eastern North
  America (100-60W) and positive height change (ridge amplification) over western
  North America (150-110W) from the stratosphere down to 500 hPa.
"""
from pathlib import Path
import json
import hashlib
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'plot_packages'),str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator,FormatStrFormatter,NullLocator
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
from PIL import Image
import analyze_upper_vortex_followup_20261029 as checked
import plot_era5_daily_vortex_boundaries as boundary_base
import analyze_vortex_regional_extension as boundary_region

WAVE=ROOT.parent/'planetary_waves'
OUT=ROOT/'results/Figure5_wave_structure_weather_20261030'
ALIGNED=WAVE/'aligned_daily_crosslevel_20261024/aligned_daily_fields.npz'
SURFACE=WAVE/'troposphere_coupling_20261023/surface_temperature_anomaly_to_Jan31.npz'
BOUNDARY=ROOT/'results/ERA5_W1_vortex_hypothesis_test/daily_boundary_and_centroid_validation.csv'
SNOW_DAILY=ROOT/'results/Figure5_cold_air_weather_evidence_20261028/observations/ghcn_daily_snow_tmin_20250108_12.csv'
COAST=ROOT/'cartopy_data/shapefiles/natural_earth/physical/ne_110m_coastline.shp'
ADMIN=WAVE/'news_storm_map_20261024/natural_earth/ne_50m_admin_1_states_provinces.shp'
WAVE_VERT_CACHE=OUT/'wavenumber_vertical_deltaZ_1_500hPa_Jan07_09.npz'
DAYS=pd.date_range('2024-12-28','2025-01-11')
PRE=np.asarray(DAYS<='2024-12-30')
MAIN=np.asarray((DAYS>='2025-01-07')&(DAYS<='2025-01-09'))
SNOW=np.asarray((DAYS>='2025-01-09')&(DAYS<='2025-01-11'))
CROSS_LAT_BAND=(40.,65.)
NA_TROUGH_BOX=(-100.,-65.,35.,55.)
G=9.80665
ONSET=pd.Timestamp('2024-12-31 16:18')
WIDTH,HEIGHT=20.,14.9
PC=ccrs.PlateCarree()
CHECKS={}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def component(field,lon,modes):
    assert np.isfinite(field).all()
    out=np.zeros_like(field,dtype=float)
    lam=np.deg2rad(lon)
    for k in modes:
        c=2*np.mean(field*np.exp(-1j*k*lam),axis=-1,keepdims=True)
        out+=np.real(c*np.exp(1j*k*lam))
    return out

def box_mean(field,lat,lon,box=NA_TROUGH_BOX):
    return checked.spatial_mean(field,lat,lon,box)

def load_vertical():
    # Reconstruct from source daily fields using the EXACT SAME baseline subtraction
    # (Jan 7-9 mean minus Dec 28-30 pre-storm mean) as panels (c) and (d)!
    checked.END=np.datetime64('2025-01-11')
    times,days,p,lat,lon,fields=checked.load()
    np.testing.assert_array_equal(days,DAYS.values.astype('datetime64[D]'))
    assert len(times)==4*len(DAYS)
    np.testing.assert_allclose(np.diff(lon),2.5,rtol=0,atol=1e-8)
    assert len(lon)==144 and len(np.unique(lon))==144
    sp=checked.read_sp(times,lat,lon)
    usep=(p>=1)&(p<=500)
    p=p[usep]
    z=checked.daily(fields['z'][:,usep].astype(float)/G)
    latmask=(lat>=CROSS_LAT_BAND[0])&(lat<=CROSS_LAT_BAND[1])
    assert (sp[:,latmask]>=p.max()*100).all(), 'Subterrain samples in upper-panel latitude band'
    assert np.isfinite(z[:,:,latmask]).all()
    w=np.cos(np.deg2rad(lat[latmask]))
    profile=np.average(z[:,:,latmask],weights=w,axis=2)
    # Subtract Dec 28-30 baseline so panels (a, b) show ΔZ_k (event change) just like (c, d)!
    delta_profile=profile[MAIN].mean(0)-profile[PRE].mean(0)
    original={name:component(delta_profile,lon,modes) for name,modes in
              [('w1',[1]),('w2',[2]),('w34',[3,4]),('w234',[2,3,4])]}
    spectrum=np.fft.rfft(delta_profile,axis=-1)
    independent=np.zeros_like(spectrum);independent[:,2:5]=spectrum[:,2:5]
    np.testing.assert_allclose(original['w234'],np.fft.irfft(independent,n=len(lon),axis=-1),
                               rtol=0,atol=1e-9)
    h=profile[PRE].mean(axis=(0,2))/1000
    alt=6371*h/(6371-h)
    assert (np.diff(alt)<0).all()
    out=dict(pressure_hPa=p,ref_alt_km=alt,lon=lon,time=DAYS.values,
             period=np.array(['2025-01-07','2025-01-09']),
             baseline=np.array(['2024-12-28','2024-12-30']),
             latitude_band=np.array(CROSS_LAT_BAND),
             daily_zonal_profiles_m=profile,
             delta_zonal_profile_m=delta_profile)
    for name,values in original.items():
        out[name+'_m']=values
    np.savez_compressed(WAVE_VERT_CACHE,**out)
    CHECKS.update(upper_native_pressure_hPa=p.tolist(),upper_display_pressure_hPa=[1,500],
        upper_latitude_band=list(CROSS_LAT_BAND),upper_period=['2025-01-07','2025-01-09'],
        upper_baseline=['2024-12-28','2024-12-30'],
        upper_all_samples_above_ground=True,upper_old_cache_read=False,
        independent_FFT_reconstruction_check=True,
        upper_sources_sha256={str(path):digest(path) for path in set(checked.INPUTS)},
        top500_daily=z[:,int(np.flatnonzero(p==500)[0])])
    return out

def load_snow():
    d=pd.read_csv(SNOW_DAILY,keep_default_na=False)
    d=d[d.element.eq('SNOW')&d.observation_date.between('2025-01-09','2025-01-11')]
    valid=d.usable.astype(str).str.lower().eq('true')&d.qflag.eq('')
    rows=[]
    for station,group in d.groupby('station_id'):
        accepted=group[valid.loc[group.index]]
        if accepted.observation_date.nunique()!=3:continue
        value=pd.to_numeric(accepted.value,errors='coerce')
        if not value.notna().all():continue
        row=group.iloc[0][['station_id','station_name','latitude','longitude']].to_dict()
        row['snowfall_accumulation_mm']=float(value.sum())
        row['observation_days_valid']=3
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'plotted_snow_observations_Jan09_11.csv',index=False,encoding='utf-8-sig')
    CHECKS['snow_complete_stations']=len(result)
    CHECKS['snow_positive_stations']=int((result.snowfall_accumulation_mm>0).sum())
    return result[result.snowfall_accumulation_mm>0]

def load_data():
    with np.load(ALIGNED) as a:f={key:a[key] for key in a.files}
    lat,lon=f['lat'],f['lon']
    np.testing.assert_array_equal(f['time'],DAYS.values.astype('datetime64[D]'))
    assert len(lon)==144 and len(np.unique(lon))==144
    assert np.allclose(np.diff(lon),2.5)
    z=np.asarray(f['z500'],float)
    dz=z-z[PRE].mean(0)
    CHECKS['aligned_deltaZ_max_error_m']=float(np.nanmax(abs(dz-f['delta_z500'])))
    assert CHECKS['aligned_deltaZ_max_error_m']<.01
    waves={name:component(z,lon,modes) for name,modes in
           [('W1',[1]),('W2',[2]),('W34',[3,4]),('W234',[2,3,4])]}
    waves['W0']=np.broadcast_to(z.mean(-1,keepdims=True),z.shape).copy()
    waves['W5plus']=z-waves['W0']-waves['W1']-waves['W234']
    changes={name:values-values[PRE].mean(0) for name,values in waves.items()}
    closure=dz-sum(changes[name] for name in ['W0','W1','W234','W5plus'])
    CHECKS['Z500_pointwise_decomposition_error_m']=float(np.max(abs(closure)))
    assert CHECKS['Z500_pointwise_decomposition_error_m']<1e-8
    ts=pd.DataFrame(dict(date=DAYS.strftime('%Y-%m-%d'),
                        Total_dZ500_m=box_mean(dz,lat,lon)))
    for name,values in changes.items():ts[name+'_dZ500_m']=box_mean(values,lat,lon)
    ts['W0_W5plus_dZ500_m']=ts.W0_dZ500_m+ts.W5plus_dZ500_m
    np.testing.assert_allclose(ts.Total_dZ500_m,ts.W0_dZ500_m+ts.W1_dZ500_m+
                               ts.W234_dZ500_m+ts.W5plus_dZ500_m,atol=1e-8)
    ts.to_csv(OUT/'NA_Z500_trough_wave_contributions.csv',index=False,encoding='utf-8-sig')
    reference=pd.read_csv(BOUNDARY)
    thresholds=reference[reference.pressure_hPa.eq(500)].threshold_gpm.unique()
    assert len(thresholds)==1
    threshold=float(thresholds[0])
    order=np.argsort(lat);ordered_lat=lat[order]
    weights=boundary_base.cell_weights(ordered_lat,len(lon))
    edges=[]
    for zz in z[:,order]:
        smoothed=boundary_base.smooth_map(zz,.4)
        area,_=boundary_region.main_component(smoothed<=threshold,weights)
        edges.append(boundary_region.boundary_latitude(smoothed,area,threshold,ordered_lat))
    edges=np.asarray(edges)
    with np.load(SURFACE) as sf:
        sd=sf['time'].astype('datetime64[D]')
        sid=[int(np.flatnonzero(sd==day)[0]) for day in DAYS.values.astype('datetime64[D]')]
        slon=(sf['lon']+180)%360-180
        so=np.argsort(slon)
        surface=sf['anomaly_K'][sid][...,so]
        slat=sf['lat']
    valid_mean=lambda a:np.divide(np.nansum(a,axis=0),np.isfinite(a).sum(0),
        out=np.full(a.shape[1:],np.nan),where=np.isfinite(a).sum(0)>0)
    support=np.isfinite(f['advection850'][MAIN]).all(axis=0)
    support&=np.isfinite(f['u850'][MAIN]).all(axis=0)&np.isfinite(f['v850'][MAIN]).all(axis=0)
    maps=dict(lat=lat,lon=lon,s_lat=slat,s_lon=slon[so],
        dz500=dz[MAIN].mean(0),adv850=np.where(support,valid_mean(f['advection850'][MAIN]),np.nan),
        u850=np.where(support,valid_mean(f['u850'][MAIN]),np.nan),v850=np.where(support,valid_mean(f['v850'][MAIN]),np.nan),
        surface=surface[SNOW].mean(0),edge_pre=edges[PRE].mean(0),
        edge_main=edges[MAIN].mean(0),snow=load_snow())
    CHECKS.update(main_period_mean_terms_m={name:float(ts.loc[MAIN,name+'_dZ500_m'].mean())
        for name in ['Total','W0','W1','W2','W34','W234','W5plus']},
        Z500_envelope_threshold_m=threshold)
    return maps,ts

def axes(fig,x,y,w,h,projection=None):
    return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT],projection=projection)

def edges(c):
    c=np.asarray(c)
    return np.r_[c[0]-(c[1]-c[0])/2,(c[:-1]+c[1:])/2,c[-1]+(c[-1]-c[-2])/2]

def heading(ax,title,subtitle):
    ax.set_title(title+'\n'+subtitle,loc='left',fontsize=12,fontweight='normal',pad=6)

def section_style(ax,vert,left=False,right=False):
    p,alt=vert['pressure_hPa'],vert['ref_alt_km']
    ax.set(xlim=(-180,-30),ylim=(np.log(500),np.log(1)),xticks=[-160,-100,-40])
    ax.set_xticklabels(['160°W','100°W','40°W'])
    pt=np.array([1,3,10,30,100,300,500])
    ax.set_yticks(np.log(pt));ax.set_yticklabels([f'{x:g}' for x in pt])
    ax.tick_params(direction='in',top=True,right=not right,labelleft=left,pad=2,length=3)
    ax.set_xlabel('Longitude',labelpad=3)
    if left:ax.set_ylabel('pressure (hPa)',labelpad=3)
    if right:
        forward=interp1d(np.log(p),alt,bounds_error=False,fill_value='extrapolate')
        inverse=interp1d(alt[::-1],np.log(p)[::-1],bounds_error=False,fill_value='extrapolate')
        secondary=ax.secondary_yaxis('right',functions=(forward,inverse))
        ticks=np.array([5,10,20,30,40,45])
        ticks=ticks[(ticks>=alt.min())&(ticks<=alt.max())]
        secondary.set_yticks(ticks);secondary.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        secondary.yaxis.set_minor_locator(NullLocator())
        secondary.set_ylabel('')
        secondary.tick_params(direction='in',labelsize=10.5,pad=2,length=3)
        secondary.set_zorder(15);secondary.patch.set_visible(False)
        ax.text(1.045,1.01,'km',transform=ax.transAxes,ha='left',va='bottom',fontsize=10.5)

def map_style(ax,left):
    ax.set_extent([-140,-50,22,82],crs=PC);ax.set_aspect('auto')
    ax.set_facecolor('#e7e7e7')
    ax.add_geometries(shapereader.Reader(COAST).geometries(),PC,facecolor='none',
                     edgecolor='#444444',lw=.5,zorder=5)
    states=[r.geometry for r in shapereader.Reader(ADMIN).records()
            if r.attributes['admin'] in ['United States of America','Canada','Mexico']]
    ax.add_geometries(states,PC,facecolor='none',edgecolor='#888888',lw=.3,zorder=4)
    ax.set_xticks([-120,-90,-60],crs=PC);ax.set_yticks([30,45,60,75],crs=PC)
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.tick_params(direction='in',top=True,right=True,labelleft=left,length=3,pad=2,labelsize=12)
    ax.gridlines(xlocs=[-120,-90,-60],ylocs=[30,45,60,75],color='#777777',
                 linewidth=.45,linestyle=':',alpha=.5,zorder=6)

def right_colorbar(fig,ax,artist,title,ticks,gap_cm=.18):
    pos=ax.get_position()
    ca=fig.add_axes([pos.x1+gap_cm/WIDTH,pos.y0+.25/HEIGHT,.17/WIDTH,pos.height-1.25/HEIGHT])
    cb=fig.colorbar(artist,cax=ca,orientation='vertical',ticks=ticks)
    if title=='m':
        cb.ax.set_title(title,fontsize=10.4,fontweight='normal',pad=4)
    else:
        cb.set_label(title.replace('\n',' '),fontsize=9.5,labelpad=2)
    cb.ax.tick_params(direction='in',labelsize=9.7,pad=1.8,length=2.7)
    return cb

def section_colorbar(fig,ax,image,values,lon,panel):
    shown=values[:,(lon>=-180)&(lon<=-30)]
    low,high=float(np.min(shown)),float(np.max(shown))
    step=100. if max(abs(low),abs(high))>=1000 else 50.
    vmin,vmax=float(step*np.floor(low/step)),float(step*np.ceil(high/step))
    assert vmin<0<vmax
    image.set_norm(TwoSlopeNorm(vcenter=0,vmin=vmin,vmax=vmax))
    def middle(value):
        magnitude=10**np.floor(np.log10(abs(value)/2))
        return np.sign(value)*np.round(abs(value)/2/magnitude)*magnitude
    ticks=[vmin,middle(vmin),0,middle(vmax),vmax]
    cb=right_colorbar(fig,ax,image,'m',ticks,gap_cm=.20 if panel=='a' else .75)
    return cb

def plot_all(m,vert,ts):
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,
        'axes.labelsize':12,'axes.titlesize':12,'axes.titleweight':'normal',
        'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':.7,
        'xtick.direction':'in','ytick.direction':'in'})
    fig=plt.figure(figsize=(WIDTH/2.54,HEIGHT/2.54))
    # Mathtext subscripts make the two-line section titles slightly taller.
    # Reserve top clearance while preserving the canvas, font sizes and row bottoms.
    top_panel_height=4.7
    a=axes(fig,1.30,8.9,4.50,top_panel_height)
    b=axes(fig,7.50,8.9,4.50,top_panel_height)
    c=axes(fig,14.15,8.9,4.70,top_panel_height)
    for ax,key,title,left,right in [(a,'w1_m',r'(a) $\Delta Z_{\mathrm{W1}}$ structure',True,False),
                                  (b,'w234_m',r'(b) $\Delta Z_{\mathrm{W2-4}}$ structure',False,True)]:
        im=ax.pcolormesh(edges(vert['lon']),edges(np.log(vert['pressure_hPa'])),vert[key],
            cmap='RdBu_r',shading='flat')
        ax.contour(vert['lon'],np.log(vert['pressure_hPa']),vert[key],levels=[0],
                   colors='#222222',linewidths=.7)
        section_style(ax,vert,left,right)
        heading(ax,title,'Jan 7–9 | 40–65°N')
        section_colorbar(fig,ax,im,vert[key],vert['lon'],'a' if left else 'b')
    c.axvline(ONSET,color='#c74336',ls='--',lw=.9)
    c.axhline(0,color='#777777',lw=.65)
    for key,label,color,ls,lw in [
        ('Total','Total','#1a1a1a','-',2.0),('W1','W1','#167da3','-',1.55),
        ('W234','W2–4','#d97813','--',1.6),
        ('W0_W5plus',r'$W_0+W_{\geq5}$','#777777','-.',1.25)]:
        c.plot(DAYS,ts[key+'_dZ500_m'],label=label,color=color,ls=ls,lw=lw)
    quantities=ts[[x+'_dZ500_m' for x in ['Total','W1','W234','W0_W5plus']]].to_numpy()
    c.set_ylim(25*np.floor(quantities.min()/25)-5,25*np.ceil(quantities.max()/25)+25)
    c.set_xlim(DAYS[0]-pd.Timedelta(hours=12),DAYS[-1]+pd.Timedelta(hours=12))
    c.set_xticks(pd.date_range('2024-12-29','2025-01-10',freq='3D'))
    c.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
    c.set_xlabel('UTC date | 2024–2025',labelpad=3)
    c.set_ylabel('')
    c.tick_params(direction='in',top=True,right=True,labelleft=False,labelright=True,pad=2,length=3)
    c.grid(axis='y',color='#cccccc',ls=':',lw=.5)
    heading(c,r'(c) $\Delta Z_{500}$ terms (m)','100–65°W, 35–55°N')
    c.legend(loc='upper right',ncol=2,fontsize=10.8,frameon=False,handlelength=1.35,
             columnspacing=.7,handletextpad=.35,borderpad=.15)
    d=axes(fig,1.10,1.2,4.50,5.0,PC)
    e=axes(fig,7.50,1.2,4.50,5.0,PC)
    f=axes(fig,13.90,1.2,4.50,5.0,PC)
    for ax,left in [(d,True),(e,False),(f,False)]:map_style(ax,left)
    im=d.pcolormesh(m['lon'],m['lat'],m['dz500'],transform=PC,shading='auto',
                    cmap='RdBu_r',norm=TwoSlopeNorm(0,-240,240),zorder=2)
    d.plot(m['lon'],m['edge_pre'],color='#555555',ls='--',lw=1.35,transform=PC,zorder=8)
    d.plot(m['lon'],m['edge_main'],color='#7b3294',lw=1.7,transform=PC,zorder=9)
    heading(d,'(d) Trough and envelope','Jan 7–9 | 500 hPa')
    right_colorbar(fig,d,im,'ΔZ500\n(m)',[-240,-120,0,120,240])
    d.legend(handles=[Line2D([],[],color='#555555',ls='--',lw=1.3,label='Dec 28–30'),
                      Line2D([],[],color='#7b3294',lw=1.7,label='Jan 7–9')],
             loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,
             fontsize=10.7,handlelength=1.2,handletextpad=.3,columnspacing=.7,borderpad=0,borderaxespad=0)
    im=e.pcolormesh(m['lon'],m['lat'],m['adv850'],transform=PC,shading='auto',
                    cmap='RdBu_r',norm=TwoSlopeNorm(0,-10,10),zorder=2)
    iy=np.flatnonzero((m['lat']>=22)&(m['lat']<=82))[::2]
    ix=np.flatnonzero((m['lon']>=-140)&(m['lon']<=-50))[::2]
    qu=e.quiver(m['lon'][ix],m['lat'][iy],m['u850'][np.ix_(iy,ix)],m['v850'][np.ix_(iy,ix)],
                transform=PC,scale=180,width=.0034,headwidth=3.4,color='#222222',zorder=8)
    e.quiverkey(qu,.36,-.14,10,r'10 m s$^{-1}$',labelpos='E',fontproperties={'size':11},coordinates='axes')
    heading(e,'(e) Cold-air transport','Jan 7–9 | 850 hPa')
    right_colorbar(fig,e,im,r'$-\mathbf{V}\cdot\nabla T$'+'\n'+r'(K day$^{-1}$)',[-10,-5,0,5,10])
    im=f.pcolormesh(m['s_lon'],m['s_lat'],m['surface'],transform=PC,shading='auto',
                    cmap='RdBu_r',norm=TwoSlopeNorm(0,-12,12),zorder=2)
    snow_points=f.scatter(m['snow'].longitude,m['snow'].latitude,s=20,marker=(6,2,0),
               color='#f7f9fa',linewidths=.6,transform=PC,zorder=12)
    snow_points.set_path_effects([pe.Stroke(linewidth=1.15,foreground='#50606b'),pe.Normal()])
    heading(f,'(f) Surface cold & snow','Jan 9–11 | 2 m temperature')
    right_colorbar(fig,f,im,r'$\Delta T_{2m}$'+'\n(K)',[-12,-6,0,6,12])
    f.legend(handles=[Line2D([],[],color='#f7f9fa',marker=(6,2,0),mec='#f7f9fa',mew=.6,
                             ms=np.sqrt(20),ls='none',label='Snow: Jan 9–11',
                             path_effects=[pe.Stroke(linewidth=1.15,foreground='#50606b'),pe.Normal()])],
             loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False,fontsize=11,
             handlelength=.8,handletextpad=.3,borderpad=0,borderaxespad=0)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clipped=[]
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            box=text.get_window_extent(renderer)
            if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
                clipped.append(text.get_text())
    assert not clipped,('Text outside canvas',clipped)
    path=OUT/'Figure5_wave_structure_weather.png'
    fig.savefig(path,dpi=600,facecolor='white')
    fig.savefig(OUT/'Figure5_wave_structure_weather_preview.png',dpi=150,facecolor='white')
    plt.close(fig)
    return path

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    maps,ts=load_data()
    vert=load_vertical()
    path=plot_all(maps,vert,ts)
    print('Main-period Z500 terms (m):',CHECKS['main_period_mean_terms_m'],flush=True)
    print('Saved:',path,flush=True)

if __name__=='__main__':main()
