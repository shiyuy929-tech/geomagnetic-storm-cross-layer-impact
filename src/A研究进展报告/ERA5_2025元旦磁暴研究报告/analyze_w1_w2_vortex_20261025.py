"""W1/W2 spectra and independent PV geometry; case diagnostics, not causal budgets.

Uses read-only ERA5 cache, daily Dec 28-Jan 11. PNG only.
"""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from PIL import Image
import plot_era5_daily_vortex_boundaries as base
import analyze_vortex_regional_extension as oldgeom
from validate_wave1_vortex_displacement import coefficient, band_weights, wrap, correlation

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'ERA5_W1_W2_vortex_response_20261025'
OUT.mkdir(parents=True,exist_ok=True)
R=6371.0
WINDOWS={'Dec28-30':np.arange(3),'Jan4-6':np.arange(7,10),
         'Jan6-9':np.arange(9,13),'Jan9-11':np.arange(12,15)}

def csvsave(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def fixed_area(q,w,eqlat):
    good=np.isfinite(q)
    ids=np.flatnonzero(good)
    order=ids[np.argsort(q.ravel()[ids])[::-1]]
    cumulative=np.cumsum(w.ravel()[order])
    k=np.argmin(abs(cumulative-2*np.pi*(1-np.sin(np.deg2rad(eqlat)))))
    threshold=q.ravel()[order[k]]
    # Break ties consistently; selected area is within one grid cell of target.
    mask=np.zeros(q.size,bool);mask[order[:k+1]]=True
    return mask.reshape(q.shape),float(threshold)

def geometry(mask,lat,lon,w):
    """Binary-area spherical centroid; LAEA moments centered ON that centroid.

    This rotationally invariant construction separates off-pole position from
    intrinsic elongation (not simply subtracting the centroid on a polar map).
    All lobes are kept for total geometry; connectivity reported separately.
    """
    wt=np.where(mask,w,0);total=wt.sum()
    if total<=0: raise ValueError('Empty core')
    phi,lam=np.deg2rad(lat)[:,None],np.deg2rad(lon)[None,:]
    xyz=np.array([np.cos(phi)*np.cos(lam),np.cos(phi)*np.sin(lam),np.broadcast_to(np.sin(phi),mask.shape)])
    center=(xyz*wt).sum((1,2));center/=np.linalg.norm(center)
    clat=np.arcsin(center[2]);clon=np.arctan2(center[1],center[0])
    east=np.array([-np.sin(clon),np.cos(clon),0])
    north=np.cross(center,east)
    cosc=np.einsum('i,ijk->jk',center,xyz)
    factor=R*np.sqrt(2/np.maximum(1+cosc,1e-8))
    x=factor*np.einsum('i,ijk->jk',east,xyz)
    y=factor*np.einsum('i,ijk->jk',north,xyz)
    x=x-(x*wt).sum()/total;y=y-(y*wt).sum()/total
    cov=np.array([[(wt*x*x).sum(),(wt*x*y).sum()],[(wt*x*y).sum(),(wt*y*y).sum()]])/total
    ev,evec=np.linalg.eigh(cov)
    major=evec[:,1];axis_bearing=float(np.rad2deg(np.arctan2(major[0],major[1]))%180)
    main,frac=oldgeom.main_component(mask,w)
    remaining=mask&~main
    if remaining.any():
        second,_=oldgeom.main_component(remaining,w)
        second_ratio=float(w[second].sum()/w[main].sum())
    else: second_ratio=0.
    sector=(lon>=-130)&(lon<=-60)
    northamerica=wt[:,sector].sum()/total
    southern=np.array([lat[np.flatnonzero(mask[:,j])[0]] if mask[:,j].any() else np.nan for j in range(len(lon))])
    return dict(centroid_lat=float(np.rad2deg(clat)),centroid_lon=float(np.rad2deg(clon)),
                displacement_km=float(R*(np.pi/2-clat)),aspect_ratio=float(np.sqrt(ev[1]/ev[0])),
                major_axis_bearing_deg=axis_bearing,area_million_km2=float(total*R*R/1e6),
                largest_component_fraction=float(frac),second_to_first_area_ratio=second_ratio,
                NA_area_fraction=float(northamerica),NA_mean_outer_lat=float(np.nanmean(southern[sector])),
                NA_min_outer_lat=float(np.nanmin(southern[sector])),
                south_domain_touch=bool(mask[0].any()))

def harmonic(field,lat,lon,band):
    bw=band_weights(lat,band)
    profile=np.sum(field*bw[:,None],axis=-2)
    cs=np.array([coefficient(profile,lon,k) for k in [1,2,3]])
    amps=2*abs(cs);fractions=2*abs(cs)**2/np.var(profile)
    relative=float(wrap(np.rad2deg(np.angle(cs[1])-2*np.angle(cs[0]))))
    t1=float(wrap(180-np.rad2deg(np.angle(cs[0]))))
    t2=float(wrap((180-np.rad2deg(np.angle(cs[1])))/2))
    return dict(W1_amp=float(amps[0]),W2_amp=float(amps[1]),W3_amp=float(amps[2]),
                W1_fraction=float(fractions[0]),W2_fraction=float(fractions[1]),W3_fraction=float(fractions[2]),
                W1_trough=t1,W2_trough_a=t2,W2_trough_b=float(wrap(t2+180)),relative_phase_deg=relative)

def compute(d):
    lat,lon=d['lat'],d['lon'];w=base.cell_weights(lat,len(lon))
    records=[];masks={};waves=[];zgeoms=[]
    for j,theta in enumerate(d['theta']):
        for t,day in enumerate(d['time']):
            q=base.smooth_map(d['pv'][t,j])
            for method in ['daily_PV_edge','fixed_area_60','fixed_area_65','fixed_area_70']:
                if method=='daily_PV_edge':
                    threshold=float(d['pv_edge'][t,j]);mask=(q>=threshold)&np.isfinite(q)
                else: mask,threshold=fixed_area(q,w,int(method[-2:]))
                masks[int(theta),t,method]=mask
                g=geometry(mask,lat,lon,w)
                records.append(dict(date=str(day),theta_K=int(theta),method=method,threshold_PVU=threshold,
                               median_height_km=float(np.nanmedian(d['iso_z'][t,j][mask])/1000),
                               median_pressure_hPa=float(np.nanmedian(d['iso_p'][t,j][mask])),**g))
    for j,pressure in enumerate(d['pressure']):
        if pressure>100: continue
        pre=d['z'][:3,j].mean(0)
        for t,day in enumerate(d['time']):
            for band in [(60,80),(50,70),(45,75)]:
                waves.append(dict(date=str(day),pressure_hPa=int(pressure),band=f'{band[0]}-{band[1]}',**harmonic(d['z'][t,j],lat,lon,band)))
            field=base.smooth_map(d['z'][t,j])
            for refl in [55,60,65]:
                threshold=float(np.interp(refl,lat,pre.mean(-1)))
                mask=field<=threshold
                zgeoms.append(dict(date=str(day),pressure_hPa=int(pressure),reference_lat=refl,threshold_gpm=threshold,**geometry(mask,lat,lon,w)))
    csvsave('daily_PV_geometry.csv',records);csvsave('daily_GPH_W1_W2_W3.csv',waves);csvsave('daily_GPH_envelope_geometry.csv',zgeoms)
    summary=[]
    for theta in d['theta']:
        for method in ['daily_PV_edge','fixed_area_60','fixed_area_65','fixed_area_70']:
            rows=[r for r in records if r['theta_K']==theta and r['method']==method]
            for name,ids in WINDOWS.items():
                ids=np.array([t for t in ids if not rows[t]['south_domain_touch']])
                r=dict(theta_K=int(theta),method=method,period=name,valid_days=len(ids))
                for key in rows[0]:
                    if key not in ['date','theta_K','method']:
                        a=np.array([rows[t][key] for t in ids])
                        r[key]=float(np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(a)))))) if key=='centroid_lon' else float(np.mean(a))
                summary.append(r)
    csvsave('PV_period_summary.csv',summary)
    wsum=[]
    for pp in [1,3,10,50,100]:
        for band in ['60-80','50-70','45-75']:
            rows=[r for r in waves if r['pressure_hPa']==pp and r['band']==band]
            for name,ids in WINDOWS.items():
                r=dict(pressure_hPa=pp,band=band,period=name)
                for key in rows[0]:
                    if key in ['date','pressure_hPa','band']:continue
                    a=np.array([rows[t][key] for t in ids])
                    r[key]=float(np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(a)))))) if ('phase' in key or 'trough' in key) else float(np.mean(a))
                wsum.append(r)
    csvsave('GPH_period_summary.csv',wsum)
    associations=[]
    for pp in [1,3,10,50,100]:
        wr=[r for r in waves if r['pressure_hPa']==pp and r['band']=='60-80']
        zr=[r for r in zgeoms if r['pressure_hPa']==pp and r['reference_lat']==60]
        associations.append(dict(pressure_hPa=pp,r_A1_displacement=correlation([r['W1_amp'] for r in wr],[r['displacement_km'] for r in zr]),
                                 r_A2_aspect=correlation([r['W2_amp'] for r in wr],[r['aspect_ratio'] for r in zr]),
                                 max_second_core_ratio=max(r['second_to_first_area_ratio'] for r in zr)))
    csvsave('descriptive_associations_no_significance_test.csv',associations)
    # Physical sanity checks independent of event interpretation.
    slat=np.arange(30,90.01,.5);slon=np.arange(-180,180,.5)
    sw=2*np.pi/len(slon)*(np.sin(np.deg2rad(np.minimum(90,slat+.25)))-np.sin(np.deg2rad(slat-.25)))
    sw=np.broadcast_to(sw[:,None],(len(slat),len(slon)))
    caps=[]
    for clat,clon in [(90,0),(75,-90),(75,30)]:
        cosine=np.sin(np.deg2rad(slat))[:,None]*np.sin(np.deg2rad(clat))+np.cos(np.deg2rad(slat))[:,None]*np.cos(np.deg2rad(clat))*np.cos(np.deg2rad(slon-clon))
        g=geometry(cosine>=np.cos(np.deg2rad(15)),slat,slon,sw)
        assert abs(g['centroid_lat']-clat)<.1
        assert abs(g['aspect_ratio']-1)<.02
        caps.append(g)
    synth=100+80*np.cos(np.deg2rad(lon-30))+25*np.cos(2*np.deg2rad(lon+40))
    spec=harmonic(np.broadcast_to(synth,(len(lat),len(lon))),lat,lon,(60,80))
    assert np.isclose(spec['W1_amp'],80) and np.isclose(spec['W2_amp'],25)
    assert np.isclose(spec['W1_trough'],-150) and np.isclose(spec['W2_trough_a'],50)
    checks=dict(dates=[str(d['time'][0]),str(d['time'][-1])],baseline='2024-12-28 to 2024-12-30',
                U60N10_daily_mps=d['u'][:,np.flatnonzero(d['pressure']==10)[0],np.flatnonzero(lat==60)[0]].mean(-1).tolist(),
                interpretation='Spectral and geometric correspondence only; no closed nonlinear interaction budget or causal intervention.',
                PV_edge='Cached daily Nash-inspired gradient/jet criterion; fixed-area cores are sensitivity diagnostics, NOT transport edges.',
                amplitude='Fourier of area-weighted band-mean RAW daily Z (zonal mean drops out); NOT climatological anomalies or 7-day running means.',
                ellipse='Binary area moments on a Lambert equal-area projection centered on the spherical area centroid; all lobes retained.',
                PV_domain_touch_cases=[{k:r[k] for k in ['date','theta_K','method']} for r in records if r['south_domain_touch']],
                quality_rule='Domain-touch cases retained and flagged in daily CSV, excluded from period summaries and plotted sensitivity bands.',
                source_sha256=hashlib.sha256(base.CACHE.read_bytes()).hexdigest(),
                validation={'rotated_circular_caps_preserve_unit_aspect_ratio':caps,'known_harmonics_recover_amplitude_and_trough':spec})
    (OUT/'method_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    return records,waves,zgeoms,masks

def savefig(fig,name):
    fig.savefig(OUT/name,dpi=600,facecolor='white')
    plt.close(fig)
    with Image.open(OUT/name) as im:
        im.thumbnail((1450,2000));im.save(OUT/name.replace('.png','_preview.png'))

def time_axis(ax,dates,last):
    ax.set_xlim(dates[0]-.5,dates[-1]+.5)
    ax.set_xticks(dates[[1,4,7,10,13]])
    ax.set_xticklabels(['Dec\n29','Jan\n01','Jan\n04','Jan\n07','Jan\n10'] if last else [])
    ax.axvspan(dates[0]-.5,dates[2]+.5,color='.5',alpha=.10,lw=0)
    ax.axvspan(dates[9]-.5,dates[12]+.5,color='#e4a200',alpha=.10,lw=0)
    ax.tick_params(direction='in',top=True,right=True)
    ax.grid(axis='y',color='.86',ls=':',lw=.6)

def plots(d,records,waves,zgeoms,masks):
    plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.labelsize':12,'axes.titlesize':14,'axes.titleweight':'normal','legend.fontsize':12,'xtick.labelsize':11,'ytick.labelsize':11})
    dates=mdates.date2num(d['time'].astype('datetime64[ms]').astype(object))
    # Same-pressure wave amplitudes and geometry of fixed case-reference Z envelopes.
    fig,axs=plt.subplots(5,2,figsize=(20/2.54,27/2.54),gridspec_kw={'hspace':.29,'wspace':.34})
    fig.subplots_adjust(left=.105,right=.92,bottom=.06,top=.925)
    colors=['#0072B2','#D55E00','#009E73']
    for i,pp in enumerate([1,3,10,50,100]):
        wr=[r for r in waves if r['pressure_hPa']==pp and r['band']=='60-80']
        zr=[r for r in zgeoms if r['pressure_hPa']==pp and r['reference_lat']==60]
        km=np.median(d['z'][:,np.flatnonzero(d['pressure']==pp)[0]][:,(d['lat']>=60)&(d['lat']<=80)])/1000
        a,b=axs[i]
        for k in range(1,4):a.plot(dates,[r[f'W{k}_amp'] for r in wr],color=colors[k-1],lw=1.6,label=f'W{k}')
        a.set_title(f'({chr(97+i*2)}) {pp} hPa (~{km:.0f} km)',loc='left',pad=6)
        a.set_ylabel('Amplitude (m)')
        b.plot(dates,[r['displacement_km']/1000 for r in zr],color='#0072B2',lw=1.8)
        twin=b.twinx();twin.plot(dates,[r['aspect_ratio'] for r in zr],color='#D55E00',ls='--',lw=1.8)
        twin.set_ylabel('Aspect ratio',color='#D55E00');twin.tick_params(direction='in',labelcolor='#D55E00')
        b.set_ylabel('Offset (1000 km)',color='#0072B2');b.tick_params(labelcolor='#0072B2')
        b.set_title(f'({chr(98+i*2)}) Position / shape',loc='left',pad=6)
        for ax in [a,b]:time_axis(ax,dates,i==4)
    handles=[Line2D([],[],color=c,lw=2,label=f'W{k}') for k,c in enumerate(colors,1)]
    fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,.995),frameon=False)
    fig.text(.5,.953,'GPH waves (60–80°N) and Z-envelope geometry',ha='center',fontsize=14)
    savefig(fig,'01_GPH_waves_and_vortex_geometry.png')
    # Independent PV geometry. Daily edge plus fixed-area sensitivity envelopes.
    fig,axs=plt.subplots(5,2,figsize=(20/2.54,27/2.54),gridspec_kw={'hspace':.30,'wspace':.32})
    fig.subplots_adjust(left=.105,right=.97,bottom=.06,top=.92)
    for i,theta in enumerate(d['theta']):
        rows=[r for r in records if r['theta_K']==theta and r['method']=='daily_PV_edge']
        for c,key in enumerate(['centroid_lat','aspect_ratio']):
            ax=axs[i,c]
            ref=[]
            for eql in [60,65,70]:
                rr=[r for r in records if r['theta_K']==theta and r['method']==f'fixed_area_{eql}']
                ref.append([r[key] if not r['south_domain_touch'] else np.nan for r in rr])
            ref=np.array(ref)
            ax.fill_between(dates,np.nanmin(ref,axis=0),np.nanmax(ref,axis=0),color='#0072B2',alpha=.16,lw=0)
            ax.plot(dates,ref[1],color='#0072B2',ls='--',lw=1.4)
            ax.plot(dates,[r[key] for r in rows],color='.15',lw=1.6,marker='o',ms=2.5)
            ax.set_ylabel('Centroid latitude (°N)' if c==0 else 'Centered aspect ratio')
            title=f'{int(theta)} K (~{np.mean([r["median_height_km"] for r in rows]):.0f} km)'
            ax.set_title(f'({chr(97+i*2+c)}) {title}',loc='left',pad=6)
            time_axis(ax,dates,i==4)
    fig.legend(handles=[Line2D([],[],color='.15',label='Daily PV edge'),Line2D([],[],color='#0072B2',ls='--',label='Fixed-area core')],loc='upper center',ncol=2,bbox_to_anchor=(.5,.995),frameon=False)
    fig.text(.5,.95,'Blue band: core areas equivalent to 60–70°N',ha='center',fontsize=12)
    savefig(fig,'02_independent_PV_geometry.png')
    # Matched actual days, not boundaries of averaged fields; main PV-gradient edge.
    fig,axs=plt.subplots(3,3,figsize=(20/2.54,23/2.54),subplot_kw={'projection':base.PROJ})
    fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.89,wspace=.035,hspace=.26)
    for i,theta in enumerate([450,850,1600]):
        j=np.flatnonzero(d['theta']==theta)[0]
        for c,t in enumerate([1,8,12]):
            ax=axs[i,c];ax.set_extent([-180,180,35,90],crs=base.PC)
            angle=np.linspace(0,2*np.pi,200)
            ax.set_boundary(base.MplPath(np.c_[np.sin(angle),np.cos(angle)]*.5+.5),transform=ax.transAxes)
            ax.coastlines(resolution='110m',lw=.55,color='.45')
            ax.gridlines(xlocs=np.arange(-180,181,60),ylocs=[40,60,80],lw=.65,color='.35',ls='--',alpha=.7)
            q=base.smooth_map(d['pv'][t,j]);mask=masks[theta,t,'daily_PV_edge']
            # Display actual gradient-selected region; no forced fill of holes or fragments.
            cyclic,x=base.add_cyclic_point(mask.astype(float),coord=d['lon'])
            ax.contourf(x,d['lat'],cyclic,levels=[.5,1.5],colors=[colors[i]],alpha=.18,transform=base.PC)
            base.contour(ax,q,d['lat'],d['lon'],d['pv_edge'][t,j],colors[i],width=1.6)
            base.contour(ax,base.smooth_map(d['pv'][1,j]),d['lat'],d['lon'],d['pv_edge'][1,j],'.35',style='--',width=1.1)
            r=next(r for r in records if r['theta_K']==theta and r['date']==str(d['time'][t]) and r['method']=='daily_PV_edge')
            ax.plot(r['centroid_lon'],r['centroid_lat'],'o',ms=4,color=colors[i],transform=base.PC)
            ax.plot(0,90,'+',ms=6,color='.2',transform=base.PC)
            for xx in [-130,-60]:ax.plot([xx,xx],[35,90],color='#b57420',ls=':',lw=.9,transform=base.PC)
            ax.set_title(f'({chr(97+3*i+c)}) '+['Dec 29','Jan 5','Jan 9'][c],fontsize=14,pad=5)
            ax.text(.5,-.07,f'{theta} K | {r["median_height_km"]:.0f} km',ha='center',transform=ax.transAxes,fontsize=12)
            if c==0:
                ax.text(.5,.96,'180°',ha='center',transform=ax.transAxes,fontsize=9,color='.25')
                ax.text(.5,.025,'0°',ha='center',transform=ax.transAxes,fontsize=9,color='.25')
                ax.text(.025,.50,'90°W',ha='left',transform=ax.transAxes,fontsize=9,color='.25')
                ax.text(.975,.50,'90°E',ha='right',transform=ax.transAxes,fontsize=9,color='.25')
            ax.text(0,60,'60°N',fontsize=9,color='.25',transform=base.PC)
    fig.suptitle('PV-defined polar vortex: position and deformation',fontsize=16,y=.98)
    fig.legend(handles=[Line2D([],[],color='.2',label='Daily edge'),Line2D([],[],color='.35',ls='--',label='Dec 29'),Line2D([],[],color='.2',marker='o',ls='',label='Centroid')],loc='upper center',bbox_to_anchor=(.5,.946),ncol=3,frameon=False)
    savefig(fig,'03_PV_boundary_comparison.png')

if __name__=='__main__':
    with np.load(base.CACHE) as x:d={k:x[k] for k in x.files}
    rec,wav,zgeo,masks=compute(d)
    plots(d,rec,wav,zgeo,masks)
    print(OUT)
