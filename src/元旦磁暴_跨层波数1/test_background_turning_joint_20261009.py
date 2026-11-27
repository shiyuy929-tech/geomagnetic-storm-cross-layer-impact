"""Native-grid observational turning-condition audit, not a wave solver.

Only observed zonal-mean GPH/T enter M0. Observed W1 curvature enters the
separately labelled conditional m2 and QG equation consistency test.
Centred derivative stencils do not extrapolate or fill missing levels.
"""
from pathlib import Path
import json, math, warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(1)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'background_turning_joint_20261009'
G=9.80665; A=6371000.; OMEGA=7.292115e-5; R=287.; CP=1004.; H=7000.
warnings.filterwarnings('ignore', message='Mean of empty slice')


def diff_native(v, x, order, points, axis):
    """Centred local polynomial derivative; no endpoint one-sided estimates."""
    v=np.moveaxis(np.asarray(v),axis,-1)
    out=np.full(v.shape,np.nan,dtype=v.dtype)
    half=points//2
    for j in range(half,len(x)-half):
        ids=np.arange(j-half,j+half+1)
        scale=max(abs(x[ids]-x[j])); xx=(x[ids]-x[j])/scale
        matrix=xx[None,:]**np.arange(points)[:,None]
        rhs=np.zeros(points);rhs[order]=math.factorial(order)/scale**order
        w=np.linalg.solve(matrix,rhs)
        block=v[...,ids];val=block@w
        out[...,j]=np.where(np.isfinite(block).all(-1),val,np.nan)
    return np.moveaxis(out,-1,axis)


def time_mean(v,width):
    if width==1:return v.copy()
    out=np.full_like(v,np.nan);half=width//2
    for j in range(half,len(v)-half):
        b=v[j-half:j+half+1];out[j]=np.where(np.isfinite(b).all(0),np.mean(b,axis=0),np.nan)
    return out


def derive(z0,t0,cz,ct,lat,p,points=5,width=1):
    z0,t0,cz,ct=[time_mean(v,width) for v in [z0,t0,cz,ct]]
    phi=np.deg2rad(lat); cp=np.cos(phi)[None,:,None]
    f=(2*OMEGA*np.sin(phi))[None,:,None];cot=1/np.tan(phi)[None,:,None]
    z=H*np.log(1000/p);dz=lambda v,n:diff_native(v,z,n,points,-1)
    dy=lambda v,n:diff_native(v,phi,n,points,1)
    zp,zpp,zppp=[dy(z0,n) for n in [1,2,3]]
    u=-G*zp/(A*f)
    up=-G/(A*f)*(zpp-cot*zp)
    upp=-G/(A*f)*(zppp-2*cot*zpp+(2*cot*cot+1)*zp)
    uz=-G*dz(zp,1)/(A*f);uzz=-G*dz(zp,2)/(A*f)
    tz,tzz,tzzz=[dz(t0,n) for n in [1,2,3]]
    n2=R/H*(tz+R/CP*t0/H)
    n2z=R/H*(tzz+R/CP*tz/H);n2zz=R/H*(tzzz+R/CP*tzz/H)
    with np.errstate(divide='ignore',invalid='ignore'):
        s=f*f/n2;b=-n2z/n2-1/H
        corr=.5*(-n2zz/n2+(n2z/n2)**2)+.25*b*b
        qmer=-(upp-np.tan(phi)[None,:,None]*up-u/cp**2)/A**2
        qvert=-s*(uzz+b*uz)
        qy=2*OMEGA*cp/A+qmer+qvert
        k=1/(A*cp)
        m0=(qy/u-k*k)/s-corr
        psi=G*cz/f
        lp=(dy(psi,2)-np.tan(phi)[None,:,None]*dy(psi,1))/A**2
        # Hydrostatic log-pressure relation: Phi_z = R T/H. GPH and T
        # share retrieval information; this is NOT independent validation.
        pz=R*ct/(H*f);pzz=R*dz(ct,1)/(H*f)
        lv=s*(pzz+b*pz)
        leff=-np.real(lp/psi)
        mbg=m0-leff/s
        mobs=-np.real((pzz+b*pz)/psi+corr)
        zon=(qy/u-k*k)*psi
        residual=abs(lp+lv+zon)/(abs(lp)+abs(lv)+abs(zon))
        q1=lp+lv-k*k*psi
        qt=diff_native(q1,np.arange(len(q1))*86400.,1,3,0)
        adv=-1j*u*k*q1;pv=-1j*k*psi*qy
        transient=abs(qt+adv+pv)/(abs(qt)+abs(adv)+abs(pv))
        q=G/(2*A*f*cp)*np.imag(cz*np.conj(ct))
        wave_u=-G*dy(cz,1)/(A*f);wave_v=-1j*G*k*cz/f
        merid=-.5*np.real(wave_u*np.conj(wave_v))
    # Background masks have no W1 amplitude dependence.
    bg=(n2>0)&np.isfinite(m0)&(u>5)
    wg=bg&(abs(cz)>=50)&(abs(ct)>=.5)&np.isfinite(residual)
    hydro_error=abs(dz(psi,1)-pz)/(abs(dz(psi,1))+abs(pz))
    return dict(u=u,uz=uz*1000,n2=n2,qy=qy,corr=corr,s=s,k=k,
        m0=np.where(bg,m0*1e6,np.nan),mbg=np.where(wg,mbg*1e6,np.nan),
        mobs=np.where(wg,mobs*1e6,np.nan),residual=np.where(wg,residual,np.nan),
        transient=np.where(wg,transient,np.nan),hydro_error=hydro_error,
        q=q,merid=merid,height=z0/1000,cz=cz,ct=ct,leff=leff,bg=bg,wg=wg)


def main():
    d=dict(np.load(OUT/'MLS_native_with_quality_top_buffer.npz'))
    lat,p,dates=d['lat'],d['pressure'],pd.DatetimeIndex(d['dates'])
    # Unit checks: derivatives of cubic on this irregular grid, no edge filling.
    z=H*np.log(1000/p);poly=(z/1e4)**3
    got=diff_native(poly,z,3,5,0)
    assert np.nanmax(abs(got-6e-12))<1e-20
    runs={};rows=[];native=[];speedrows=[]
    samplings=['combined','LT1','LT2','pair']+[f'delete{s:02}' for s in range(1,13)]
    for points in [5,7]:
        for width in [1,3]:
            for sampling in samplings:
                sector=int(sampling[-2:]) if sampling.startswith('delete') else 0
                if sampling in ['combined','LT1','LT2']:
                    branch=['combined','LT1','LT2'].index(sampling)
                    vals=[d[k][branch,sector] for k in ['z0','t0','cz','ct']]
                else:vals=[np.mean(d[k][1:3,sector],axis=0) for k in ['z0','t0','cz','ct']]
                z0,t0,cz,ct=vals;cz=cz[...,0];ct=ct[...,0]
                v=derive(z0,t0,cz,ct,lat,p,points,width)
                key=f'{sampling}_p{points}_w{width}';runs[key]=v
                region=(lat[None,:,None]>=35)&(lat[None,:,None]<=50)&(v['height']>=75)&(v['height']<=90)
                for name,start,end in [('pre','2024-12-28','2024-12-30'),('early','2025-01-01','2025-01-03'),('return','2025-01-05','2025-01-07'),('late','2025-01-08','2025-01-09')]:
                    mask=region&((dates>=start)&(dates<=end))[:,None,None]
                    valid=mask&v['bg']; wvalid=mask&v['wg']
                    ok=wvalid&(v['residual']<.25)&(v['transient']<.25)
                    r=dict(setting=key,period=name,n_bg=int(valid.sum()),n_wave=int(wvalid.sum()),
                        n_m0_negative=int((valid&(v['m0']<0)).sum()),n_mbg_negative=int((wvalid&(v['mbg']<0)).sum()),
                        n_both_negative=int((wvalid&(v['mbg']<0)&(v['mobs']<0)).sum()),
                        n_both_residual_ok=int(ok.sum()),n_negative_and_residual_ok=int((ok&(v['mbg']<0)&(v['mobs']<0)).sum()))
                    for vn in ['u','uz','n2','m0','mbg','mobs','residual','transient','hydro_error','q','merid']:
                        a=v[vn][mask];a=a[np.isfinite(a)];r[vn+'_median']=float(np.median(a)) if len(a) else np.nan
                    rows.append(r)
                if sampling in ['pair','LT1','LT2','combined']:
                    for it,day in enumerate(dates):
                        for il in np.flatnonzero((lat>=35)&(lat<=50)):
                            for iz in np.flatnonzero((v['height'][it,il]>=68)&(v['height'][it,il]<=98)):
                                row=dict(setting=key,date=str(day.date()),lat=float(lat[il]),p_hPa=float(p[iz]))
                                row.update({vn:float(v[vn][it,il,iz]) for vn in ['height','u','uz','n2','m0','mbg','mobs','residual','transient','q','merid','hydro_error']})
                                native.append(row)
                if sampling=='pair':
                    for period in [0,-20,-10,-5,20,10,5]:
                        c=np.zeros((1,len(lat),1)) if period==0 else (2*np.pi*A*np.cos(np.deg2rad(lat))/(period*86400))[None,:,None]
                        with np.errstate(invalid='ignore',divide='ignore'):
                            m=((v['qy']/(v['u']-c)-v['k']**2)/v['s']-v['corr'])*1e6
                        good=v['bg']&(abs(v['u']-c)>=5)
                        mask=region&((dates>='2025-01-05')&(dates<='2025-01-07'))[:,None,None]&good
                        a=m[mask]
                        speedrows.append(dict(points=points,width=width,period_days=period,n=int(len(a)),n_negative=int((a<0).sum()),median=float(np.median(a)) if len(a) else np.nan))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'candidate_turning_settings.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(native).to_csv(OUT/'native_candidate_fields.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(speedrows).to_csv(OUT/'phase_speed_sensitivity.csv',index=False,encoding='utf-8-sig')
    # Keep only plotted/reference arrays rather than duplicating every run.
    ref=runs['pair_p5_w1']
    saved={k:v for k,v in ref.items() if k not in ['bg','wg','cz','ct']}
    saved.update(dates=dates.values,lat=lat,pressure=p)
    np.savez_compressed(OUT/'native_reference_diagnostics.npz',**saved)
    # Wave coefficients cannot change a background-only M0 diagnostic.
    v2=derive(z0,t0,cz*1.3*np.exp(.7j),ct*.8,lat,p,points,width)
    assert np.allclose(v2['m0'],v['m0'],equal_nan=True)
    # Top-buffer sensitivity: it can extend derivative availability, but it must
    # not be treated as a well-validated independent observation above 90 km.
    pairvals=[np.mean(d[k][1:3,0],axis=0) for k in ['z0','t0','cz','ct']]
    boundary=[]
    for points in [5,7]:
        for width in [1,3]:
            z0b,t0b,czb,ctb=pairvals
            no_top=derive(z0b[...,:-1],t0b[...,:-1],czb[...,:-1,0],ctb[...,:-1,0],lat,p[:-1],points,width)
            yes_top=runs[f'pair_p{points}_w{width}']
            for il in np.flatnonzero((lat>=35)&(lat<=50)):
                for vn in ['m0','mbg']:
                    valid_no=np.isfinite(no_top[vn][:,il]);valid_yes=np.isfinite(yes_top[vn][:,il])
                    common=valid_no&valid_yes[:,:-1]
                    max_no=no_top['height'][:,il][valid_no]
                    max_yes=yes_top['height'][:,il][valid_yes]
                    boundary.append(dict(points=points,width=width,lat=float(lat[il]),diagnostic=vn,
                        highest_without_buffer_km=float(np.max(max_no)) if len(max_no) else None,
                        highest_with_buffer_km=float(np.max(max_yes)) if len(max_yes) else None,
                        common_values_max_difference=float(np.max(abs(no_top[vn][:,il][common]-yes_top[vn][:,il,:-1][common]))) if common.any() else None))
    pd.DataFrame(boundary).to_csv(OUT/'top_boundary_audit.csv',index=False,encoding='utf-8-sig')
    # Compare signs only at the SAME valid cells, do not count missing as agreement.
    signs=[]
    for width in [1,3]:
        keys=[f'{sam}_p{pt}_w{width}' for sam in ['pair','LT1','LT2'] for pt in [5,7]]
        vs=[runs[k] for k in keys]; base=vs[0]
        region=(lat[None,:,None]>=35)&(lat[None,:,None]<=50)&(base['height']>=75)&(base['height']<=90)&((dates>='2025-01-01')&(dates<='2025-01-09'))[:,None,None]
        for vn in ['m0','mbg']:
            aa=np.stack([v[vn] for v in vs]);common=region&np.isfinite(aa).all(0)
            neg=common&(aa<0).all(0)
            strict=neg.copy()
            if vn=='mbg':
                strict&=np.stack([v['mobs']<0 for v in vs]).all(0)
                strict&=np.stack([(v['residual']<.25)&(v['transient']<.25) for v in vs]).all(0)
            cells=[dict(date=str(dates[it].date()),latitude=float(lat[il]),pressure_hPa=float(p[iz]),height_km=float(base['height'][it,il,iz])) for it,il,iz in np.argwhere(neg)]
            signs.append(dict(width=width,diagnostic=vn,n_common=int(common.sum()),n_all_negative=int(neg.sum()),
                n_with_wave_equation_checks=int(strict.sum()) if vn=='mbg' else None,negative_cells=cells))
    summary=dict(method='observational native-grid audit, no propagation equation solve or prediction',
        region='35-50N; 75-90km; 1-9 January 2025',background_only_functional_independence_test='passed',
        local_times_are_sensitivity_not_independent_replication=True,
        amplitude_screen_GPH_m=50,amplitude_screen_T_K=.5,
        equation_residual_screen=.25,residual_threshold_exploratory=True,
        sign_agreement=signs,
        source_readonly=True,new_top_buffer_hPa=float(p[-1]),boundary_audit=boundary,
        vertical_coordinate='z*=H ln(1000/p), H=7km; display altitude is observed mean GPH/1000')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(frame[(frame.setting.isin(['pair_p5_w1','pair_p7_w1','pair_p5_w3','pair_p7_w3']))].to_string(index=False))
    print('Common-cell sign audit:',json.dumps(signs,ensure_ascii=False))
    print('Saved',OUT)


if __name__=='__main__':main()
