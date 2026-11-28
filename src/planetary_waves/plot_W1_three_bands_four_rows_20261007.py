"""Three latitude bands, amplitude / signed net proxy / phase / profiles.

Read-only existing native MLS coefficients and existing lower ERA5 diagnostic.
No fitted counterpropagating modes, no source inference from relative phase.
"""
from pathlib import Path
import numpy as np,pandas as pd,json,re,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.patheffects as pe
from matplotlib.colors import SymLogNorm,TwoSlopeNorm,Normalize
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'W1_three_bands_four_rows_Dec27_Jan10_20261007'
SRC=ROOT/'event_evidence_20261002/reflection_origin_20261004/extended_native'
OLD=ROOT/'results/comprehensive_diagnostic/comprehensive_W1_diagnostic_values.csv'
DAYS=pd.date_range('2024-12-27','2025-01-10')
BANDS=[(20,40),(40,60),(60,80)]
PROFILES=['2025-01-02','2025-01-06']
G=9.80665;H=7000.;R=287.05;CP=1004.;A=6371000.;OM=7.2921159e-5

def edges(x):return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]

def average(a,w,axis):
    return np.where(np.isfinite(a).all(axis),np.average(np.nan_to_num(a),weights=w,axis=axis),np.nan)

def interp_contiguous(z,c,target):
    result=np.full(len(target),np.nan+1j*np.nan)
    valid=np.flatnonzero(np.isfinite(c))
    for group in np.split(valid,np.flatnonzero(np.diff(valid)>1)+1):
        if len(group)<2:continue
        keep=(target>=z[group[0]])&(target<=z[group[-1]])
        result[keep]=np.interp(target[keep],z[group],c[group].real)+1j*np.interp(target[keep],z[group],c[group].imag)
    return result

def prepare():
    raw=dict(np.load(SRC/'MLS_W123_sector_fits.npz'))
    dates=pd.DatetimeIndex(np.load(SRC/'dates.npz')['dates']);it=dates.get_indexer(DAYS)
    old=pd.read_csv(OLD);p=raw['pressure'];lat=raw['lat'];zlog=H*np.log(1000/p)
    display=np.arange(20.,90.1,2.5);bands=[];records=[];summary=[]
    for lo,hi in BANDS:
        ids=np.flatnonzero((lat>=lo)&(lat<hi));ll=lat[ids];w=np.cos(np.deg2rad(ll))
        cases=[(0,k)for k in range(13)]+[(1,0),(2,0)]
        zc=np.stack([np.take(raw['cz'][b,k,...,0],ids,axis=1)[it]for b,k in cases])
        tc=np.stack([np.take(raw['ct'][b,k,...,0],ids,axis=1)[it]for b,k in cases])
        t0=np.take(raw['t0'][0,0],ids,axis=1)[it]
        n2=R/H*(np.gradient(t0,zlog,axis=-1,edge_order=2)+R/CP*t0/H)
        factor=.5*G/(2*OM*np.sin(np.deg2rad(ll))*A*np.cos(np.deg2rad(ll)))
        flux=average(np.where(n2[None]>0,factor[None,None,:,None]*np.imag(zc*tc.conj()),np.nan),w,2)
        coeff=average(zc,w,2)
        z=average(np.take(raw['z0'][0,0],ids,axis=1)[it],w,1).mean(0)/1000
        keep=(z>=40)&(z<=85);zn=z[keep];point=coeff[0][:,keep];cc=coeff[:,:,keep]
        anchor=int(np.argmin(abs(zn-50)));relative=np.full(point.shape,np.nan);good_rows=[]
        for i in range(len(DAYS)):
            deviations=np.angle(cc[:,i]*np.conj(point[i])[None,:],deg=True)
            good=np.isfinite(cc[:,i]).all(0)&(abs(point[i])>=50)&(np.max(abs(deviations),axis=0)<=45)
            good_rows.append(good)
            if not good[anchor]:continue
            indices=np.flatnonzero(good)
            # Do not unwrap through an invalid or low-amplitude height gap.
            for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                if anchor not in group:continue
                phase=np.rad2deg(np.unwrap(np.angle(point[i,group])))
                phase-=phase[np.flatnonzero(group==anchor)[0]]
                relative[i,group]=phase
        amplitude=np.full((len(DAYS),len(display)),np.nan)
        lower=old[(old.lat_band==f'{lo}-{hi}N')&(old.height_km<=42.5)].pivot(index='date',columns='height_km',values='w1_amplitude_m').reindex(DAYS.strftime('%Y-%m-%d'))
        for i in range(len(DAYS)):
            for j,h in enumerate(display):
                if h<=42.5 and h in lower.columns:amplitude[i,j]=lower.iloc[i][h]
            high=display>=45
            amplitude[i,high]=abs(interp_contiguous(z,coeff[0,i],display[high]))
        heat=flux[0][:,keep]
        phase_limit=max(90.,np.ceil(np.nanmax(abs(relative))/50)*50)if np.isfinite(relative).any()else 90.
        heat_limit=max(1.,np.ceil(np.nanmax(abs(heat))/10)*10)
        ampstep=100 if lo==20 else 500
        amp_limit=np.ceil(np.nanmax(amplitude)/ampstep)*ampstep
        bands.append(dict(band=f'{lo}°–{hi}°N',z=zn,amplitude=amplitude,display_height=display,
                          heat=heat,phase=relative,amp_limit=amp_limit,heat_limit=heat_limit,phase_limit=phase_limit))
        summary.append(dict(band=f'{lo}-{hi}',latitude_centers=ll.tolist(),phase_anchor_height_km=float(zn[anchor]),
                            amp_range=[0,float(amp_limit)],flux_range=[-float(heat_limit),float(heat_limit)],
                            phase_range=[-float(phase_limit),float(phase_limit)],valid_phase_nodes=int(np.isfinite(relative).sum())))
        for i,day in enumerate(DAYS):
            for j,h in enumerate(zn):records.append(dict(band=f'{lo}-{hi}',date=str(day.date()),height_km=h,net_proxy_Kms=heat[i,j],relative_phase_deg=relative[i,j],phase_sampling_pass=bool(good_rows[i][j])))
        pd.DataFrame(amplitude,index=DAYS,columns=display).to_csv(OUT/f'amplitude_{lo}_{hi}.csv',index_label='date')
        pd.DataFrame(relative,index=DAYS,columns=zn).to_csv(OUT/f'relative_phase_{lo}_{hi}.csv',index_label='date')
        pd.DataFrame(heat,index=DAYS,columns=zn).to_csv(OUT/f'net_proxy_{lo}_{hi}.csv',index_label='date')
    pd.DataFrame(records).to_csv(OUT/'native_flux_phase_values.csv',index=False)
    return bands,summary

def dateaxis(ax,labels,line_color='.2',outlined=False):
    ax.set_xlim(DAYS[0]-pd.Timedelta(hours=12),DAYS[-1]+pd.Timedelta(hours=12))
    ax.set_xticks(pd.date_range('2024-12-28','2025-01-09',freq='4D'))
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %d'));ax.tick_params(labelbottom=labels)
    if labels:plt.setp(ax.get_xticklabels(),rotation=35,ha='right');ax.set_xlabel('UTC date',labelpad=3)
    line=ax.axvline(pd.Timestamp('2025-01-01'),color=line_color,ls='--',lw=.8)
    if outlined:line.set_path_effects([pe.Stroke(linewidth=1.7,foreground='white'),pe.Normal()])

def plot(bands,compact_revision=False,latitude_tick_revision=False,white_date_revision=False,outlined_date_revision=False,export_pdf=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'axes.titlesize':12,
                         'xtick.direction':'in','ytick.direction':'in','axes.linewidth':.8})
    height_cm=22.4 if compact_revision else 23
    fig=plt.figure(figsize=(20/2.54,height_cm/2.54))
    fig.suptitle('W1 amplitude, net flux and phase',fontsize=16,weight='normal',y=.987)
    fig.text(.5,.951 if white_date_revision else .955,'27 Dec 2024–10 Jan 2025',ha='center',va='center',fontsize=12)
    lefts=[.079,.381,.683] if compact_revision else [.10,.397,.695]
    width=.235 if compact_revision else .21
    tops=[.889,.684,.479,.214] if compact_revision else [.87,.665,.46,.19]
    bottoms=[.729,.524,.319,.084] if compact_revision else [.715,.51,.305,.085]
    for col,band in enumerate(bands):
        fig.text(lefts[col]+width/2,.927 if latitude_tick_revision else (.932 if compact_revision else .92),band['band'],ha='center',va='center',fontsize=13)
        for row in range(4):
            ax=fig.add_axes([lefts[col],bottoms[row],width,tops[row]-bottoms[row]])
            letter=chr(97+row*3+col)
            ax.tick_params(top=True,right=True,labelleft=col==0)
            if col==0:ax.set_ylabel('Approx. alt. (km)',labelpad=5)
            if row==0:
                values,z=band['amplitude'],band['display_height'];norm=Normalize(0,band['amp_limit']);cmap='viridis';title='W1 amplitude';unit='m'
                if col==0:ticks=[0,100,200,300,band['amp_limit']] if compact_revision else [0,200,400,band['amp_limit']]
                elif col==1:ticks=np.unique(np.r_[np.arange(0,band['amp_limit']+1,500),band['amp_limit']])
                else:ticks=np.unique(np.r_[np.arange(0,band['amp_limit']+1,1000),band['amp_limit']])
            elif row==1:
                values,z=band['heat'],band.get('heat_z',band['z']);norm=SymLogNorm(1,vmin=-band['heat_limit'],vmax=band['heat_limit']);cmap='RdBu_r';title='Net W1 proxy';unit='K m s⁻¹'
                choices=np.r_[-band['heat_limit'],-100,-10,-1,0,1,10,100,band['heat_limit']]
                ticks=np.unique(choices[(choices>=-band['heat_limit'])&(choices<=band['heat_limit'])])
                if band['heat_limit']==150:ticks=np.array([-150,-10,-1,0,1,10,150])
                if band['heat_limit']==200:ticks=np.array([-200,-10,-1,0,1,10,200])
            elif row==2:
                values,z=band['phase'],band['z'];norm=TwoSlopeNorm(vmin=-band['phase_limit'],vcenter=0,vmax=band['phase_limit']);cmap='PuOr_r';title='Phase / 50 km';unit='°'
                ticks=[-band['phase_limit'],0,band['phase_limit']]
            if row<3:
                im=ax.pcolormesh(edges(md.date2num(DAYS)),edges(z),values.T,cmap=cmap,norm=norm,shading='flat')
                cx=fig.add_axes([lefts[col]+width+.007,bottoms[row],.012,tops[row]-bottoms[row]])
                cb=fig.colorbar(im,cax=cx,ticks=ticks);cb.ax.set_title(unit,fontsize=9,pad=5);cb.ax.tick_params(direction='in',labelsize=9)
                cb.ax.set_yticklabels([f'{v:g}'for v in ticks])
                ax.set_title(f'({letter}) {title}',loc='left',pad=5)
                dateaxis(ax,row==2,line_color='black' if outlined_date_revision else ('white' if white_date_revision else '.2'),outlined=outlined_date_revision)
                if row==0:
                    ax.set(ylim=(20,90),yticks=[20,40,60,80]);ax.axhline(43.75,color='.5',ls=':',lw=.7)
                    if latitude_tick_revision:
                        ax.set_yticks([20,40,50,60,80])
                        ax.tick_params(axis='y',colors='black')
                    if compact_revision:ax.axhline(50,color='white',ls='--',lw=.9)
                else:
                    if (row==1 and 'heat_z'in band)or(row==2 and band.get('extended_phase',False)):
                        upper=90
                        ax.set(ylim=(40,upper),yticks=[40,50,60,70,80,90])
                    else:ax.set(ylim=(40,85),yticks=[40,50,60,70,80])
                    ax.axhline(50,color='.5',ls=':',lw=.7)
            else:
                for day,color,label in [('2025-01-02','#0072B2','2 Jan'),('2025-01-06','#D55E00','6 Jan')]:
                    i=DAYS.get_loc(day);v=band['phase'][i];idx=np.flatnonzero(np.isfinite(v)&(band['z']>=39)&(band['z']<=91));first=True
                    for group in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1):
                        if len(group):ax.plot(v[group],band['z'][group],color=color,lw=1.7,marker='o',ms=3.4,label=label if first else None);first=False
                visible=(band['z']>=40)&(band['z']<=90)
                finite=band['phase'][[DAYS.get_loc(d)for d in PROFILES]][:,visible]
                if np.isfinite(finite).any():
                    low=np.floor(np.nanmin(finite)/50)*50-10;high=np.ceil(np.nanmax(finite)/50)*50+10
                else:low,high=-100,100
                upper=90 if band.get('extended_phase',False)else 85
                ticks=np.arange(np.ceil(low/50)*50,np.floor(high/50)*50+1,50)
                ax.set(xlim=(low,high),ylim=(32,upper+8),yticks=[40,50,60,70,80,90]if band.get('extended_phase',False)else[40,50,60,70,80],xlabel='Phase offset (°)')
                ax.set_xticks(ticks);ax.tick_params(axis='x',rotation=35)
                ax.set_title(f'({letter}) Phase profiles',loc='left',pad=5);ax.axhline(50,color='.5',ls=':',lw=.7)
                ax.legend(loc='lower left',frameon=False,fontsize=10.5 if compact_revision else 12,handlelength=1.4,labelspacing=.25)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bad=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible()and t.get_text():
            bb=t.get_window_extent(renderer)
            if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height:bad.append(t.get_text())
    if bad:raise RuntimeError('Clipped text: '+repr(bad))
    name='04_W1_three_bands_white_Jan1' if white_date_revision else ('03_W1_three_bands_latitude_50km_tick' if latitude_tick_revision else ('02_W1_three_bands_compact_phase150' if compact_revision else '01_W1_amplitude_flux_relative_phase_three_bands'))
    if outlined_date_revision:name='05_W1_three_bands_visible_Jan1'
    if outlined_date_revision and bands[1]['amp_limit']==1800 and bands[2]['amp_limit']==2500:name='06_W1_three_bands_amplitude_1800_2500'
    if outlined_date_revision and bands[0]['amp_limit']==450:name='07_W1_three_bands_amplitude_450'
    fig.savefig(OUT/(name+'.png'),dpi=600)
    if export_pdf:fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)
    with Image.open(OUT/(name+'.png'))as im:assert im.width==4724;pixels=list(im.size)
    if export_pdf:
        media=re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)',(OUT/(name+'.pdf')).read_bytes())
        assert abs(float(media.group(1))/72*2.54-20)<1e-6
    return dict(pixels=pixels,width_cm=20,height_cm=height_cm,dpi=600,text_boundary_check=True,pdf_exported=export_pdf)

def main():
    OUT.mkdir(exist_ok=True);bands,summary=prepare();qa=plot(bands)
    audit=dict(dates=[str(DAYS[0].date()),str(DAYS[-1].date())],profile_dates=PROFILES,
               phase_amplitude_threshold_m=50,phase_case_deviation_max_deg=45,
               unwrap_only_connected_valid_height_segment=True,band_settings=summary,figure=qa,
               source_sha256=hashlib.sha256((SRC/'MLS_W123_sector_fits.npz').read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
