"""Two spatially explicit storm / propagation diagnostic figures, PNG only.

Search windows are prescribed from the observed response boxes, not detected
reflectors. No missing heating/vertical wind, causal lag or wave ray is imposed.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.text import Text
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'reflector_attribution_20261020'))
from build_acp2018_event_figures_20261010 import extract, flux
from analyze_candidates import fields
from ep_flux_display_20261011 import draw_arrows

DAYS = pd.date_range('2024-12-27', '2025-01-10')
MID = DAYS + pd.Timedelta(hours=12)
PRE = DAYS <= '2024-12-30'
POST = (DAYS >= '2025-01-06') & (DAYS <= '2025-01-09')
ONSET = pd.Timestamp('2024-12-31 16:18')
XTICKS = pd.to_datetime(['2024-12-28','2025-01-01','2025-01-05','2025-01-09'])
BLUE, RED, PURPLE = '#0072b2', '#c53b2b', '#7b3294'
GREEN, GOLD, CYAN, DARK = '#009e73', '#d6a000', '#00d9ef', '#36464f'
HASHES = {}
SERIES, COVERAGE, DIAGNOSTICS = [], [], []
CONFIG = {
    'mid': dict(name='Midlatitude', target=(40,60), support=(40,60),
                latlim=(35,70), zlim=(45,85), thermal_zlim=(50,90),
                candidate=(40,60,60,75), response=(40,60,53,64), color=GOLD),
    'high': dict(name='High-latitude', target=(60,80), support=(60,70),
                 latlim=(50,80), zlim=(45,100), thermal_zlim=(50,100),
                 candidate=(60,80,80,95), response=(60,80,55,88), color=GREEN),
}


def source(rel):
    path = ROOT/rel
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load():
    bg = dict(np.load(source('two_region_diagnostics_20261021/background/background_native_Dec27_Jan20.npz')))
    ids = pd.DatetimeIndex(bg['dates']).get_indexer(DAYS)
    assert (ids >= 0).all()
    result = {k: (v[ids] if v.ndim == 3 else v) for k,v in bg.items() if k != 'dates'}
    raw = dict(np.load(source('W1_three_bands_Dec27_Jan20_20261016/MLS_W123_fullheight_Dec27_Jan20.npz')))
    it = pd.DatetimeIndex(raw['dates']).get_indexer(DAYS)
    cz,ct,z0,t0 = [v[it] for v in extract(raw)]
    f = [flux(cz[i,...,0],ct[i,...,0],z0[i],t0[i],raw['lat'],raw['pressure'],points=3) for i in range(len(DAYS))]
    result['P'], result['W'] = [np.stack([row[k] for row in f]) for k in ['P','W']]
    np.testing.assert_allclose(result['Fz'],result['W']*np.cos(np.deg2rad(raw['lat']))[None,:,None],equal_nan=True,atol=1e-15)
    # LT branch spread is a sampling sensitivity, not a confidence interval.
    result['LT'] = []
    for branch in [1,2]:
        _,_,zz,tt = [v[it] for v in extract(raw,branch=branch)]
        r = [fields(zz[i],tt[i],raw['lat'],raw['pressure'],points=5) for i in range(len(DAYS))]
        result['LT'].append({k:np.stack([row[k] for row in r]) for k in ['qy','Uz','thermal_shear']})
    idx = pd.read_csv(source('PCN_AE_three_newyear_cases_20261017/AE_PCN_2025_Dec27_Jan20_1min.csv'))
    idx['UTC_time'] = pd.to_datetime(idx.UTC_time,utc=True).dt.tz_localize(None)
    result['indices'] = idx.set_index('UTC_time')[['aeValue','PCN']].resample('1h').mean()
    result['particles'] = pd.read_csv(source('two_region_diagnostics_20261021/forcing/paired_fixed_daily.csv'),parse_dates=['date'])
    result['chemistry'] = pd.read_csv(source('storm_two_path_test_20261020/MLS_night_O3_T_reference.csv'),parse_dates=['date'])
    result['arrow_levels'] = np.load(source('acp2018_event_figures_20261010/daily_W1_EP_12days_fields.npz'))['displayed_pressure_indices']
    return result


def latmean(a, data, band, name=None):
    mask = (data['latitude']>=band[0])&(data['latitude']<band[1])
    weights = np.cos(np.deg2rad(data['latitude'][mask]))
    good = np.isfinite(a[:,mask]).all(axis=(0,1))
    out = np.full((len(DAYS), a.shape[-1]),np.nan)
    out[:,good] = np.average(a[:,mask][:,:,good],weights=weights,axis=1)
    height = np.average(data['reference_height_km'][mask],weights=weights,axis=0)
    if name is not None:
        for j in np.flatnonzero((height>=40)&(height<=100)):
            COVERAGE.append(dict(variable=name,requested_band=f'{band[0]}-{band[1]}',
                pressure_hPa=data['pressure_hPa'][j],height_km=height[j],
                complete_requested_band=bool(good[j]),
                available_latitudes=';'.join(map(str,data['latitude'][mask][np.isfinite(a[:,mask,j]).all(0)]))))
    return out,height


def chemistry(data, band):
    key=f'{band[0]}-{band[1]}'
    c=data['chemistry']
    c=c[(c.band==key)&(c['mode']=='night')&(c['case']=='reference')]
    pressure=np.sort(c.pressure_hPa.unique())[::-1]
    a={k:c.pivot(index='date',columns='pressure_hPa',values=k).reindex(index=DAYS,columns=pressure).to_numpy()
       for k in ['temperature_K','ozone_ppmv','height_km']}
    dT=a['temperature_K']-a['temperature_K'][PRE].mean(0)
    dO3=100*(a['ozone_ppmv']/a['ozone_ppmv'][PRE].mean(0)-1)
    dO3[:,~(a['ozone_ppmv'][PRE]>.05).all(0)]=np.nan
    return dict(dT=dT,dO3=dO3,height=a['height_km'][PRE].mean(0),pressure=pressure)


def record(region,panel,variable,values,height=None,pressure=None,**meta):
    for it,date in enumerate(DAYS):
        row=dict(region=region,panel=panel,variable=variable,date=str(date.date()),**meta)
        if np.asarray(values).ndim==1:
            SERIES.append(dict(**row,value=float(values[it])))
        else:
            for ip in range(values.shape[1]):
                SERIES.append(dict(**row,height_km=float(height[ip]),pressure_hPa=float(pressure[ip]),value=float(values[it,ip])))


def edges(x):
    x=np.asarray(x)
    return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]


def time_style(ax,label=False):
    ax.set_xlim(DAYS[0],DAYS[-1]+pd.Timedelta(days=1))
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.axvline(ONSET,color='#a23d56',ls='--',lw=.8,zorder=10)
    ax.axvline(pd.Timestamp('2025-01-04'),color='.45',ls=':',lw=.7,zorder=10)
    ax.tick_params(direction='in',top=True,right=True,labelbottom=label)
    if label: ax.set_xlabel('UTC date | 2024–2025',labelpad=2)


def height_style(ax,zlim,right=False):
    ax.set_ylim(*zlim)
    ax.set_yticks(np.arange(np.ceil(zlim[0]/10)*10,zlim[1]+1,10))
    ax.tick_params(direction='in',top=True,right=True,labelleft=not right)
    if not right: ax.set_ylabel('approx. alt. (km)',labelpad=2)
    ax.axhline(50,color='.35',ls=':',lw=.65,zorder=8)
    ax.set_facecolor('#e9e9e9')


def time_mesh(ax,values,h,zlim,norm,cmap='RdBu_r'):
    select=np.isfinite(h)&(h>zlim[0]-8)&(h<zlim[1]+8)
    hh=h[select]
    image=ax.pcolormesh(mdates.date2num(DAYS.append(pd.DatetimeIndex([DAYS[-1]+pd.Timedelta(days=1)]))),
                       edges(hh),np.ma.masked_invalid(values[:,select].T),cmap=cmap,norm=norm,shading='flat')
    finite=np.isfinite(values[:,select]).all(0)
    if finite.any():
        highest=hh[finite].max()
        if highest<zlim[1]:
            ax.axhspan(highest,zlim[1],color='#e9e9e9',zorder=2)
    return image


def search_rectangle(ax,conf,time_axis=False):
    lo,hi,bot,top=conf['candidate']
    left=mdates.date2num(DAYS[0]) if time_axis else lo
    width=15 if time_axis else hi-lo
    ax.add_patch(Rectangle((left,bot),width,top-bot,fill=False,ec=conf['color'],lw=1.4,ls='--',zorder=9))


def draw_chem(ax,data,band,conf,right,title,region):
    c=chemistry(data,band)
    im=time_mesh(ax,c['dT'],c['height'],conf['thermal_zlim'],TwoSlopeNorm(vmin=-15,vcenter=0,vmax=15))
    for level,ls in [(-25,'--'),(25,'-')]:
        if np.nanmin(c['dO3'])<level<np.nanmax(c['dO3']):
            cs=ax.contour(mdates.date2num(MID),c['height'],np.ma.masked_invalid(c['dO3'].T),levels=[level],
                          colors=[PURPLE],linewidths=1.0,linestyles=ls,corner_mask=False,zorder=4)
            ax.clabel(cs,fmt={level:f'{level:+d}%'},fontsize=10,inline_spacing=2)
    height_style(ax,conf['thermal_zlim'],right)
    time_style(ax)
    ax.set_title(title,loc='left',pad=4)
    if band==conf['target']: search_rectangle(ax,conf,True)
    for key in ['dT','dO3']:
        record(region,title[1],key,c[key],c['height'],c['pressure'],band=f'{band[0]}-{band[1]}')
    return im


def mean_complete(a,select):
    return np.where(np.isfinite(a[select]).all(0),a[select].mean(0),np.nan)


def draw_background(ax,data,conf,select,right):
    lat,h=data['latitude'],data['reference_height_km']
    u=mean_complete(data['U'],select)
    u[~np.isfinite(data['U'][PRE|POST]).all(0)]=np.nan
    p,w=[mean_complete(data[k],select) for k in ['P','W']]
    common=np.isfinite(data['P'][PRE|POST]).all(0)&np.isfinite(data['W'][PRE|POST]).all(0)
    p[~common]=np.nan;w[~common]=np.nan
    vertices,indices=[],[]
    for il,la in enumerate(lat):
        for ip in range(1,len(data['pressure_hPa'])-1):
            hz=h[il,ip-1:ip+2]
            if np.isfinite(hz).all() and np.isfinite(u[il,ip]):
                vertices.append([(la-2.5,(hz[0]+hz[1])/2),(la+2.5,(hz[0]+hz[1])/2),
                                 (la+2.5,(hz[1]+hz[2])/2),(la-2.5,(hz[1]+hz[2])/2)])
                indices.append((il,ip))
    im=PolyCollection(vertices,array=np.asarray([u[ix] for ix in indices]),cmap='PuOr_r',
                      norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=80),edgecolors='none',zorder=0)
    ax.add_collection(im)
    n=data['n2'][select]
    negative=np.isfinite(n).all(0)&((n<0).mean(0)>=.5)
    missing=~np.isfinite(n).all(0)
    for mask,hatch in [(negative,'///'),(missing,'...')]:
        vv=[v for v,ix in zip(vertices,indices) if mask[ix]]
        ax.add_collection(PolyCollection(vv,facecolors='none',edgecolors='.45',linewidths=0,hatch=hatch,zorder=2))
    if np.array_equal(select,POST):
        before=data['n2'][PRE]
        newly_negative=(np.isfinite(before).all(0)&~(before<0).any(0)
                        &np.isfinite(n).all(0)&((n<0).sum(0)>=2))
        vv=[v for v,ix in zip(vertices,indices) if newly_negative[ix]]
        ax.add_collection(PolyCollection(vv,facecolors='none',edgecolors='#202020',linewidths=.9,zorder=5))
    for a,col,ls in [(u,'#5e3c99','-'),(mean_complete(data['qy'],select),'#a23b54','--')]:
        if np.nanmin(a)<0<np.nanmax(a):
            ax.contour(np.broadcast_to(lat[:,None],h.shape),h,np.ma.masked_invalid(a),levels=[0],colors=[col],linewidths=.9,linestyles=ls,zorder=4)
    keep=np.zeros(len(data['pressure_hPa']),dtype=bool)
    keep[data['arrow_levels']]=True
    keep[np.argmin(abs(data['pressure_hPa']-.1))]=False
    keep&=(np.nanmedian(h,axis=0)>=conf['zlim'][0])&(np.nanmedian(h,axis=0)<=conf['zlim'][1])
    lo,hi=conf['latlim'];p[(lat<lo)|(lat>hi)]=np.nan;w[(lat<lo)|(lat>hi)]=np.nan
    ax.set_xlim(lo,hi)
    height_style(ax,conf['zlim'],right)
    ax.set_xticks(np.arange(np.ceil(lo/10)*10,hi+1,10))
    artist,chosen=draw_arrows(ax,lat,h,p,w,keep,readable=True,uniform_heads=True,
        length_bounds_inches=(.105,.105),head_points=2.8,shaft_points=.85)
    for patch,v in zip(artist.fixed_head_patches,w[chosen]):
        down=v<0
        patch.set_facecolor(CYAN if down else DARK)
        patch.set_edgecolor(CYAN if down else DARK)
        patch.set_path_effects([pe.withStroke(linewidth=1.65,foreground='#00354f' if down else 'white'),pe.Normal()])
    search_rectangle(ax,conf)
    x0,x1,z0,z1=conf['response']
    ax.add_patch(Rectangle((x0,z0),x1-x0,z1-z0,fill=False,ec=conf['color'],lw=1.4,zorder=9))
    ax.set_xlabel('Latitude (°N)',labelpad=2)
    return im


def profile_change(values):
    return values[POST].mean(0)-values[PRE].mean(0)


def draw_profiles(ax,data,conf,region):
    band=conf['support']
    results={}
    h=None
    means={key:latmean(data[key],data,band,key)[0] for key in ['qy','qmer','qvert']}
    shared=np.logical_and.reduce([np.isfinite(a).all(0) for a in means.values()])
    for key,col,ls,label in [('qy',DARK,'-','Total'),('qmer',BLUE,'--','Meridional'),('qvert',RED,'-.','Vertical')]:
        a,h=latmean(data[key],data,band)
        a[:,~shared]=np.nan
        delta=profile_change(a)*1e11
        results[key]=delta
        # Keep the neighboring native level across the display edge so a
        # valid profile is clipped at the axis, rather than ending too early.
        shown=shared&(h>=conf['zlim'][0]-6)&(h<=conf['zlim'][1]+6)
        ax.plot(delta[shown],h[shown],color=col,ls=ls,lw=1.4,marker='o' if key=='qy' else None,ms=2.5,label=label)
        record(region,'f',key,a,h,data['pressure_hPa'],band=f'{band[0]}-{band[1]}')
    np.testing.assert_allclose(results['qy'],results['qmer']+results['qvert'],equal_nan=True,atol=1e-12)
    spreads=[]
    for setting in data['LT']:
        a,_=latmean(setting['qy'],data,band)
        spreads.append(profile_change(a)*1e11)
    both=np.stack(spreads)
    known=np.isfinite(both).all(0)&shared
    ax.fill_betweenx(h,np.where(known,both.min(0),np.nan),np.where(known,both.max(0),np.nan),color='.5',alpha=.18,zorder=0)
    ax.axvline(0,color='.55',lw=.7)
    height_style(ax,conf['zlim'],True)
    ax.set_facecolor('white')
    valid=np.isfinite(results['qy'])
    if valid.any() and h[valid].max()<conf['zlim'][1]:
        ax.axhspan(h[valid].max(),conf['zlim'][1],color='#e9e9e9',zorder=-1)
    ax.set_xlabel('Δqᵧ (10⁻¹¹ m⁻¹ s⁻¹)',labelpad=2)
    ax.set_title(f'(f) PV-gradient contributions\n{band[0]}–{band[1]}°N | Jan 6–9 − pre',loc='left',pad=4)
    ax.legend(loc='upper right',fontsize=9.5,frameon=False,handlelength=1.5,labelspacing=.2)
    return h,results


def draw_shear(ax,data,conf,region):
    band=conf['support']
    actual,h=latmean(data['Uz'],data,band,'Uz')
    predicted,_=latmean(data['thermal_shear'],data,band,'thermal_shear')
    common=np.isfinite(actual).all(0)&np.isfinite(predicted).all(0)
    actual[:,~common]=np.nan;predicted[:,~common]=np.nan
    displayed=common&(h>=conf['zlim'][0]-6)&(h<=conf['zlim'][1]+6)
    for mask,col,label in [(PRE,'.4','Dec 27–30'),(POST,'#d47215','Jan 6–9')]:
        ax.plot(actual[mask].mean(0)[displayed],h[displayed],color=col,lw=1.5,marker='o',ms=2.4,label=label)
        ax.plot(predicted[mask].mean(0)[displayed],h[displayed],color=col,lw=1.3,ls='--')
    ax.axvline(0,color='.55',lw=.65)
    height_style(ax,conf['zlim'])
    ax.set_facecolor('white')
    if displayed.any() and h[displayed].max()<conf['zlim'][1]:
        ax.axhspan(h[displayed].max(),conf['zlim'][1],color='#e9e9e9',zorder=-1)
    ax.set_xlabel('∂U/∂z* (m s⁻¹ km⁻¹)',labelpad=2)
    ax.set_title(f'(i) Shear and thermal wind\n{band[0]}–{band[1]}°N',loc='left',pad=4)
    ax.legend(loc='lower left',fontsize=9.5,frameon=False,handlelength=1.4,labelspacing=.2)
    for name,a in [('actual_shear',actual),('thermal_shear',predicted)]:
        record(region,'i',name,a,h,data['pressure_hPa'],band=f'{band[0]}-{band[1]}')
    return actual,predicted,h


def draw_wave(ax,data,conf,region):
    fz,h=latmean(data['Fz'],data,conf['target'],'Fz')
    amplitude,_=latmean(data['amplitude_W1'],data,conf['target'],'amplitude_W1')
    im=time_mesh(ax,fz,h,conf['zlim'],SymLogNorm(linthresh=.02,linscale=.7,vmin=-2,vmax=2))
    finite=np.isfinite(amplitude).all(0)&np.isfinite(fz).all(0)
    levels=[1200,1600] if region=='mid' else [1600,2400]
    cs=ax.contour(mdates.date2num(MID),h[finite],amplitude[:,finite].T,levels=levels,colors=[GREEN],linestyles='--',linewidths=1.1)
    ax.clabel(cs,fmt='%d',fontsize=9.5,inline_spacing=2)
    height_style(ax,conf['zlim'],True)
    time_style(ax,True)
    _,_,lo,hi=conf['response']
    ax.axhline(lo,color=conf['color'],lw=1.0)
    ax.axhline(hi,color=conf['color'],lw=1.0)
    ax.set_title(f'(j) W1 vertical EP flux\n{conf["target"][0]}–{conf["target"][1]}°N',loc='left',pad=4)
    record(region,'j','Fz_reduced',fz,h,data['pressure_hPa'],band=f'{conf["target"][0]}-{conf["target"][1]}')
    record(region,'j','GPH_W1_amplitude',amplitude,h,data['pressure_hPa'])
    return im


def finish(fig,path,axes,twins):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    # Matplotlib retains locator ticks outside view limits; their Text objects
    # exist but Axis.draw does not render them. Exclude only those ghost labels.
    inactive_tick_labels=set()
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            low,high=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-9<=tick.get_loc()<=high+1e-9:
                    inactive_tick_labels.update([tick.label1,tick.label2])
    outside=[]
    for t in fig.findobj(Text):
        if t not in inactive_tick_labels and t.get_visible() and t.get_text().strip():
            b=t.get_window_extent(renderer)
            if b.width>0 and b.height>0 and (b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5):outside.append(t.get_text())
    assert not outside, f'Clipped text: {outside}'
    decorations=list(fig.texts)+list(fig.legends)
    for ax in fig.axes:
        decorations += [ax._left_title,ax.title,ax._right_title,ax.xaxis.label,ax.yaxis.label]
        if ax.get_legend() is not None:decorations.append(ax.get_legend())
    collisions=[]
    for i,a in enumerate(decorations):
        if isinstance(a,Text) and not a.get_text().strip():continue
        for b in decorations[i+1:]:
            if isinstance(b,Text) and not b.get_text().strip():continue
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                collisions.append((str(a),str(b)))
    assert not collisions, f'Decoration overlaps: {collisions}'
    fig.savefig(path,dpi=600)
    fig.savefig(path.with_name(path.stem+'_preview.png'),dpi=140)
    with Image.open(path) as im:size=list(im.size)
    assert abs(size[0]-20/2.54*600)<2
    plt.close(fig)
    return dict(path=str(path),pixels=size,width_cm=20,height_cm=29,dpi=600,clipped_text=outside,decoration_overlaps=collisions)


def make(data,region):
    conf=CONFIG[region]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12,'axes.titleweight':'normal',
        'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':.8,
        'xtick.direction':'in','ytick.direction':'in','hatch.linewidth':.5,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(20/2.54,29/2.54))
    left,right,width=.092,.532,.338
    rows=[(.866,.075),(.664,.139),(.475,.127),(.263,.127),(.055,.120)]
    axes=np.array([[fig.add_axes([x,y,width,h]) for x in [left,right]] for y,h in rows])
    fig.suptitle(f'{conf["name"]} W1 and storm-time background',x=.50,y=.989,fontsize=16,fontweight='normal')
    twins=[]
    a,b=axes[0]
    idx=data['indices']
    ae,=a.plot(idx.index,idx.aeValue,color=DARK,lw=.8)
    a.set_ylim(0,1800);a.set_yticks([0,800,1600]);a.set_ylabel('AE (nT)',labelpad=2)
    twin=a.twinx();twins.append(twin)
    pc,=twin.plot(idx.index,idx.PCN,color=BLUE,lw=.8,alpha=.75)
    twin.set_ylim(-1,14);twin.set_yticks([0,6,12]);twin.tick_params(direction='in',colors=BLUE,labelsize=10)
    twin.set_ylabel('PCN (mV m⁻¹)',color=BLUE,fontsize=10,labelpad=2)
    a.set_title('(a) Geomagnetic activity',loc='left',pad=4)
    a.legend([ae,pc],['AE','PCN'],loc='upper left',ncol=2,fontsize=10,frameon=False,handlelength=1.1,columnspacing=.7,borderpad=.1)
    particle=data['particles']
    handles=[]
    for band,col in [('40-60',GOLD),('60-80',GREEN)]:
        for channel,ls in [('e1','-'),('e3','--')]:
            sub=particle[(particle.band==band)&(particle.channel==channel)&(particle.telescope==0)].set_index('date').reindex(DAYS)
            line,=b.plot(MID,sub.flux,color=col,ls=ls,lw=1.1,marker='o' if channel=='e1' else None,ms=2)
            if channel=='e1':handles.append(line)
            record(region,'b','MEPED_'+channel,sub.flux.to_numpy(),band=band)
    b.set_yscale('log');b.set_ylim(1,2e6);b.set_yticks([1,1e2,1e4,1e6]);b.yaxis.tick_right();b.tick_params(labelsize=10,direction='in')
    b.set_title('(b) Directional electron flux',loc='left',pad=4)
    b.legend(handles,['40–60°N','60–80°N'],loc='upper center',bbox_to_anchor=(.5,-.07),ncol=2,fontsize=9.5,frameon=False,handlelength=1.2,labelspacing=.15,columnspacing=.8,borderpad=.1)
    b.text(.03,.93,'>40 keV —\n>287 keV --',transform=b.transAxes,va='top',fontsize=9.5)
    b.text(1.19,.5,'cm⁻² s⁻¹ sr⁻¹',transform=b.transAxes,rotation=90,ha='center',va='center',fontsize=10)
    time_style(a);time_style(b)

    if region=='mid':
        tc=draw_chem(axes[1,0],data,(60,80),conf,False,'(c) Polar temperature and ozone\n60–80°N',region)
        draw_chem(axes[1,1],data,(40,60),conf,True,'(d) Local temperature and ozone\n40–60°N',region)
        # Polar source range is separate from the midlatitude search layer.
        axes[1,0].axhline(65,color=GREEN,lw=.8,ls=':');axes[1,0].axhline(85,color=GREEN,lw=.8,ls=':')
        cb=fig.colorbar(tc,cax=fig.add_axes([.891,rows[1][0],.012,rows[1][1]]),ticks=[-15,0,15])
        cb.ax.set_title('ΔT\n(K)',fontsize=10,pad=4)
    else:
        tc=draw_chem(axes[1,0],data,(60,80),conf,False,'(c) Polar temperature and ozone\n60–80°N',region)
        cb=fig.colorbar(tc,cax=fig.add_axes([.445,rows[1][0],.012,rows[1][1]]),ticks=[-15,0,15])
        cb.ax.set_title('ΔT\n(K)',fontsize=10,pad=4)
        u,hh=latmean(data['U'],data,conf['support'],'U')
        im=time_mesh(axes[1,1],u,hh,conf['thermal_zlim'],TwoSlopeNorm(vmin=-40,vcenter=0,vmax=80),'PuOr_r')
        cs=axes[1,1].contour(mdates.date2num(MID),hh,np.ma.masked_invalid(u.T),levels=[0],colors=['#5e3c99'],linewidths=1.0)
        height_style(axes[1,1],conf['thermal_zlim'],True);time_style(axes[1,1]);search_rectangle(axes[1,1],conf,True)
        axes[1,1].set_title('(d) Upper wind and reversal\n60–70°N support',loc='left',pad=4)
        cb=fig.colorbar(im,cax=fig.add_axes([.891,rows[1][0],.012,rows[1][1]]),ticks=[-40,0,40,80])
        cb.ax.set_title('$U_g$\n(m s⁻¹)',fontsize=10,pad=4)
        record(region,'d','U_geostrophic',u,hh,data['pressure_hPa'],band='60-70')

    gradient,h=latmean(data['T_gradient'],data,conf['support'],'T_gradient')
    shear,_=latmean(data['Uz'],data,conf['support'])
    dg=gradient-gradient[PRE].mean(0);ds=shear-shear[PRE].mean(0)
    e=axes[2,0]
    gi=time_mesh(e,dg,h,conf['zlim'],TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12))
    cs=e.contour(mdates.date2num(MID),h,np.ma.masked_invalid(ds.T),levels=[-2,-1,1,2],colors='.2',linewidths=.8)
    e.clabel(cs,fmt='%+g',fontsize=9,inline_spacing=2)
    height_style(e,conf['zlim']);time_style(e);search_rectangle(e,conf,True)
    e.set_title(f'(e) Thermal gradient and shear\n{conf["support"][0]}–{conf["support"][1]}°N support',loc='left',pad=4)
    cb=fig.colorbar(gi,cax=fig.add_axes([.445,rows[2][0],.012,rows[2][1]]),ticks=[-12,0,12])
    cb.ax.set_title('ΔTᵧ',fontsize=10,pad=4)
    cb.ax.set_ylabel('K / 1000 km',fontsize=9,labelpad=1)
    record(region,'e','delta_T_gradient',dg,h,data['pressure_hPa'])
    record(region,'e','delta_actual_shear',ds,h,data['pressure_hPa'])
    draw_profiles(axes[2,1],data,conf,region)

    for j,select in enumerate([PRE,POST]):
        ui=draw_background(axes[3,j],data,conf,select,j==1)
        axes[3,j].set_title('(g) Wind and EP direction\nDec 27–30' if j==0 else '(h) Wind and EP direction\nJan 6–9',loc='left',pad=4)
    cb=fig.colorbar(ui,cax=fig.add_axes([.891,rows[3][0],.012,rows[3][1]]),ticks=[-40,0,40,80],extend='both')
    cb.ax.set_title('$U_g$\n(m s⁻¹)',fontsize=10,pad=4)
    fig.legend(handles=[Patch(fc='white',ec='.4',hatch='///',label='n² < 0'),Patch(fc='white',ec='.4',hatch='...',label='n² incomplete'),
        Line2D([],[],color='#5e3c99',label='$U_g = 0$'),Line2D([],[],color='#a23b54',ls='--',label='qᵧ = 0'),
        Line2D([],[],color=CYAN,marker='>',markeredgecolor='#00354f',label='Downward EP'),
        Patch(fc='none',ec='#202020',lw=.9,label='New n² < 0')],
        loc='center',bbox_to_anchor=(.5,.651),ncol=6,fontsize=9,frameon=False,handlelength=1.1,handletextpad=.35,columnspacing=.7)
    actual,predicted,hh=draw_shear(axes[4,0],data,conf,region)
    fi=draw_wave(axes[4,1],data,conf,region)
    cb=fig.colorbar(fi,cax=fig.add_axes([.891,rows[4][0],.012,rows[4][1]]),ticks=[-2,-.02,0,.02,2],extend='both')
    cb.ax.set_yticklabels(['−2','−.02','0','.02','2'])
    cb.ax.set_title('$\\widetilde F_z$\n(m² s⁻²)',fontsize=9.5,pad=4)
    for ax in fig.axes:ax.tick_params(direction='in')
    lo,hi=conf['candidate'][2:]
    matched=(hh>=lo)&(hh<=hi)&np.isfinite(actual).all(0)&np.isfinite(predicted).all(0)
    difference=actual[:,matched]-predicted[:,matched]
    DIAGNOSTICS.append(dict(region=region,target_band=conf['target'],background_band=conf['support'],
        search_window=conf['candidate'],response_window=conf['response'],
        matched_shear_levels=int(matched.sum()),shear_actual_minus_thermal_RMSE=float(np.sqrt(np.mean(difference**2))),
        causal_attribution_established=False,vertical_turning_surface_established=False))
    return finish(fig,OUT/f'{"01" if region=="mid" else "02"}_{region}_regional_mechanism_tests.png',axes,twins)


def main():
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    data=load()
    outputs=[make(data,r) for r in ['mid','high']]
    spatial=[]
    profiles=[]
    for region,conf in CONFIG.items():
        lo,hi=conf['latlim']
        for il in np.flatnonzero((data['latitude']>=lo)&(data['latitude']<hi)):
            for ip in np.flatnonzero((data['reference_height_km'][il]>=conf['zlim'][0])&(data['reference_height_km'][il]<=conf['zlim'][1])):
                n=data['n2'][:,il,ip]
                row=dict(region=region,latitude=data['latitude'][il],pressure_hPa=data['pressure_hPa'][ip],
                    reference_height_km=data['reference_height_km'][il,ip],
                    reference_known_days=int(np.isfinite(n[PRE]).sum()),reference_negative_days=int((n[PRE]<0).sum()),
                    comparison_known_days=int(np.isfinite(n[POST]).sum()),comparison_negative_days=int((n[POST]<0).sum()))
                row['new_negative_at_least_2_of_4_days']=bool(np.isfinite(n[PRE]).all() and not (n[PRE]<0).any() and (n[POST]<0).sum()>=2)
                row['new_negative_full_comparison_support']=bool(row['new_negative_at_least_2_of_4_days'] and np.isfinite(n[POST]).all())
                for field in ['U','Uz','Uzz','Uyy','N2','qy','qmer','qvert','Fz','EPFD_W1']:
                    a=data[field][:,il,ip]
                    row[field+'_pre']=a[PRE].mean() if np.isfinite(a[PRE]).all() else np.nan
                    row[field+'_post']=a[POST].mean() if np.isfinite(a[POST]).all() else np.nan
                spatial.append(row)
        for field in ['U','Uz','Uzz','Uyy','N2','qy','qmer','qvert','T_gradient','thermal_shear']:
            a,h=latmean(data[field],data,conf['support'])
            for ip in range(a.shape[1]):
                if conf['zlim'][0]<=h[ip]<=conf['zlim'][1]:
                    profiles.append(dict(region=region,variable=field,band=f'{conf["support"][0]}-{conf["support"][1]}',
                        pressure_hPa=data['pressure_hPa'][ip],reference_height_km=h[ip],
                        reference=a[PRE,ip].mean(),comparison=a[POST,ip].mean(),delta=profile_change(a)[ip]))
    pd.DataFrame(spatial).to_csv(OUT/'spatial_background_changes.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'background_profile_terms.csv',index=False)
    pd.DataFrame(SERIES).to_csv(OUT/'plotted_series.csv',index=False)
    pd.DataFrame(COVERAGE).drop_duplicates().to_csv(OUT/'native_latitude_height_coverage.csv',index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in HASHES.items())
    assert not list(OUT.glob('*.pdf'))
    (OUT/'method_and_checks.json').write_text(json.dumps(dict(figures=outputs,diagnostics=DIAGNOSTICS,
        reference_dates=['2024-12-27','2024-12-30'],comparison_dates=['2025-01-06','2025-01-09'],
        search_windows_are_not_diagnosed_reflectors=True,negative_index_is_total_c0_QG_not_vertical_m2=True,
        no_vertical_wind_or_radiative_heating_imposed=True,source_files_unchanged=True,
        native_grids=True,PDF_export=False,input_sha256=HASHES),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(outputs,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
