"""Paper-inspired, observation-only event sections; no propagation solver.

Retain old coefficients, add lower MLS native levels to a NEW cache. The
spatial harmonic least-squares extraction is not a reflected-wave fit.
"""
from pathlib import Path
from datetime import datetime
import hashlib, json, warnings
import numpy as np
import pandas as pd
import h5py
from threadpoolctl import threadpool_limits
import test_background_turning_joint_20261009 as bg

threadpool_limits(1)
warnings.filterwarnings('ignore', category=RuntimeWarning)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'acp2018_event_figures_20261010'
SRC = ROOT / 'event_evidence_20261002/reflection_origin_20261004/extended_native'
DAYS = pd.date_range('2024-12-25', '2025-01-10')
REF = ('2024-12-25', '2024-12-30')
PERIODS = [('I', '2024-12-27', '2025-01-03'),
           ('II', '2025-01-04', '2025-01-10'),
           ('pre', '2024-12-25', '2024-12-30'),
           ('before4', '2024-12-27', '2024-12-30'),
           ('early3', '2025-01-01', '2025-01-03'),
           ('return3', '2025-01-05', '2025-01-07')]
G, A, OM, H, R, CP = 9.80665, 6371000., 7.292115e-5, 7000., 287., 1004.


def read(path, product):
    with h5py.File(path) as h:
        sw = h[f'HDFEOS/SWATHS/{product}']
        return {k: sw['Data Fields'][k][:] for k in
                ['L2gpValue', 'L2gpPrecision', 'Quality', 'Convergence', 'Status']} | {
                k: sw['Geolocation Fields'][k][:] for k in
                ['Pressure', 'Latitude', 'Longitude', 'LocalSolarTime']}


def build_cache():
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'MLS_W123_fullheight_native.npz'
    if cache.exists():
        return dict(np.load(cache))
    source = SRC / 'MLS_W123_sector_fits.npz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    old = dict(np.load(source))
    dates = pd.DatetimeIndex(np.load(SRC / 'dates.npz')['dates'])
    ti = dates.get_indexer(DAYS)
    assert (ti >= 0).all()
    maps = {}
    for product in ['GPH', 'Temperature']:
        folder = ROOT / 'data' / f'MLS_{product}_v06_20241225_20250110'
        maps[product] = {pd.Timestamp(datetime.strptime(p.stem.rsplit('_', 1)[-1], '%Yd%j').date()): p
                         for p in folder.glob('*.he5')}
        assert all(day in maps[product] for day in DAYS)
    first = read(maps['GPH'][DAYS[0]], 'GPH')
    levels = np.flatnonzero((first['Pressure'] <= 261.1) & (first['Pressure'] >= .00046))
    pressure = first['Pressure'][levels].astype(float)
    lat = old['lat']
    shape = (3, 13, len(DAYS), len(lat), len(levels))
    new = {k: np.full(shape + (3,) if k in ['cz', 'ct'] else shape,
                     np.nan + 1j*np.nan if k in ['cz', 'ct'] else np.nan,
                     dtype=complex if k in ['cz', 'ct'] else float)
           for k in ['cz', 'ct', 'z0', 't0', 'counts']}
    added, copied = [], []
    for ip, p in enumerate(pressure):
        j = int(np.argmin(abs(old['pressure'] - p)))
        if np.isclose(old['pressure'][j], p, rtol=1e-5):
            for k in new:
                new[k][:, :, :, :, ip] = np.take(old[k], ti, axis=2)[:, :, :, :, j]
            copied.append(ip)
        else:
            added.append(ip)
    manifest = []
    for it, day in enumerate(DAYS):
        gp, tp = maps['GPH'][day], maps['Temperature'][day]
        g, t = read(gp, 'GPH'), read(tp, 'Temperature')
        assert np.array_equal(g['Pressure'], t['Pressure'])
        assert np.array_equal(g['Latitude'], t['Latitude'])
        la, lon, lst = [g[k] for k in ['Latitude', 'Longitude', 'LocalSolarTime']]
        sectors = (lon % 360 // 30).astype(int)
        lam = np.deg2rad(lon)
        common_quality = np.isfinite(lon) & np.isfinite(lst)
        for v in [g, t]:
            common_quality &= (v['Status'] % 2 == 0) & (v['Convergence'] < 1.03)
        harmonics = np.column_stack([f(k*lam) for k in [1, 2, 3] for f in [np.cos, np.sin]])
        for il, center in enumerate(lat):
            xlat = (la - center) / 5
            common = np.column_stack([np.ones(len(la)), xlat, xlat*xlat])
            for ib in range(3):
                branch = np.ones(len(la), bool) if ib == 0 else (lst < 8 if ib == 1 else lst >= 8)
                X = np.column_stack([common, np.where(lst < 8, 1., -1.), harmonics]) if ib == 0 else np.column_stack([common, harmonics])
                for iz in added:
                    j = levels[iz]
                    Y = np.column_stack([g['L2gpValue'][:, j], t['L2gpValue'][:, j]])
                    keep = common_quality & branch & (abs(la - center) <= 5)
                    threshold = .9 if pressure[iz] >= 100 else .2
                    for v in [g, t]:
                        keep &= (v['Quality'] > threshold) & (v['L2gpPrecision'][:, j] > 0)
                    keep &= np.isfinite(Y).all(1) & (Y[:, 0] > 0) & (Y[:, 0] < 130000)
                    keep &= (Y[:, 1] > 100) & (Y[:, 1] < 350)
                    if keep.sum() < 50 or np.unique(sectors[keep]).size < 9:
                        continue
                    xx, yy, ss = X[keep], Y[keep], sectors[keep]
                    w = np.cos(np.deg2rad(la[keep]))
                    nx = np.array([(xx[ss == s].T*w[ss == s]) @ xx[ss == s] for s in range(12)])
                    ny = np.array([(xx[ss == s].T*w[ss == s]) @ yy[ss == s] for s in range(12)])
                    normal = np.concatenate([nx.sum(0)[None], nx.sum(0)[None] - nx])
                    cross = np.concatenate([ny.sum(0)[None], ny.sum(0)[None] - ny])
                    nn = np.bincount(ss, minlength=12)
                    n = np.r_[nn.sum(), nn.sum() - nn]
                    valid = (np.linalg.cond(normal) < 900) & (n >= 40)
                    valid &= np.r_[np.count_nonzero(nn) >= 9, np.count_nonzero(nn) - (nn > 0) >= 8]
                    beta = np.linalg.solve(normal[valid], cross[valid])
                    new['cz'][ib, valid, it, il, iz] = beta[:, -6::2, 0] + 1j*beta[:, -5::2, 0]
                    new['ct'][ib, valid, it, il, iz] = beta[:, -6::2, 1] + 1j*beta[:, -5::2, 1]
                    new['z0'][ib, valid, it, il, iz] = beta[:, 0, 0]
                    new['t0'][ib, valid, it, il, iz] = beta[:, 0, 1]
                    new['counts'][ib, valid, it, il, iz] = n[valid]
        for product, path in [('GPH', gp), ('Temperature', tp)]:
            manifest.append(dict(date=str(day.date()), product=product, path=str(path),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('New lower/top levels', day.date(), flush=True)
    new.update(pressure=pressure, lat=lat, dates=DAYS.values)
    for ip in copied:
        j = int(np.argmin(abs(old['pressure'] - pressure[ip])))
        assert np.allclose(new['cz'][..., ip, :], np.take(old['cz'], ti, axis=2)[..., j, :], equal_nan=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    np.savez_compressed(cache, **new)
    (OUT / 'data_manifest.json').write_text(json.dumps(dict(source=str(source), source_sha256=digest,
        source_unchanged=True, copied_levels=len(copied), added_pressure_hPa=pressure[added].tolist(),
        raw_files=manifest, missing_not_filled=True, high_layers_not_independent_resolution=True),
        ensure_ascii=False, indent=2), encoding='utf8')
    return new


def extract(d, branch=-1, sector=0):
    return [np.mean(d[k][1:3, sector], axis=0) if branch < 0 else d[k][branch, sector]
            for k in ['cz', 'ct', 'z0', 't0']]


def background(z0, t0, lat, pressure, points=5):
    """Matthias & Ern Eq.2-3 at c=0; diagnostic, not vertical m^2."""
    phi = np.deg2rad(lat)
    zstar = H*np.log(1000/pressure)
    f = (2*OM*np.sin(phi))[:, None]
    cot = 1/np.tan(phi)[:, None]
    cosine = np.cos(phi)[:, None]
    dy = lambda v, n: bg.diff_native(v, phi, n, points, 0)
    dz = lambda v, n: bg.diff_native(v, zstar, n, points, -1)
    zp, zpp, zppp = [dy(z0, n) for n in [1, 2, 3]]
    U = -G*zp/(A*f)
    up = -G*(zpp - cot*zp)/(A*f)
    upp = -G*(zppp - 2*cot*zpp + (2*cot*cot + 1)*zp)/(A*f)
    uz, uzz = -G*dz(zp, 1)/(A*f), -G*dz(zp, 2)/(A*f)
    tz, tzz = dz(t0, 1), dz(t0, 2)
    N2 = R/H*(tz + R/CP*t0/H)
    N2z = R/H*(tzz + R/CP*tz/H)
    qmer = -(upp - np.tan(phi)[:, None]*up - U/cosine**2)/A**2
    qvert = -f*f/N2*(uzz - (1/H + N2z/N2)*uz)
    qy = 2*OM*cosine/A + qmer + qvert
    n2 = qy/U - 1/(A*cosine)**2 - f*f/(4*N2*H*H)
    valid = (N2 > 0) & (abs(U) >= 5) & np.isfinite(n2)
    # A wind map requires only the first latitude derivative, not third order.
    Uplot = -G*bg.diff_native(z0, phi, 1, 3, 0)/(A*f)
    return dict(U=Uplot, n2=np.where(valid, n2, np.nan), qy=qy, N2=N2, Uz=uz*1000)


def flux(z, t, z0, t0, lat, pressure, wave=1, points=3):
    """Eq.4-5, p-proportional density; no geometric display stretch in divergence."""
    phi = np.deg2rad(lat)
    f = (2*OM*np.sin(phi))[:, None]
    cos = np.cos(phi)[:, None]
    zz = H*np.log(1000/pressure)
    u = -G*bg.diff_native(z, phi, 1, points, 0)/(A*f)
    v = -1j*wave*G*z/(A*f*cos)
    strat = bg.diff_native(t0, zz, 1, 3, -1) + R/CP*t0/H
    strat = np.where(strat > 0, strat, np.nan)
    P = -.5*np.real(u*v.conj())
    Q = .5*np.real(v*t.conj())
    W = f*Q/strat
    rho = pressure[None, :]/1000  # constant reference density cancels in EPFD
    Fy, Fz = rho*A*cos*P, rho*A*cos*W
    mer = bg.diff_native(Fy*cos, phi, 1, 3, 0)/(A*cos)
    vert = bg.diff_native(Fz, zz, 1, 3, -1)
    div = (mer + vert)/(rho*A*cos)*86400
    return dict(u=u, v=v, Q=Q, P=P, W=W, Fy=Fy, Fz=Fz, div=div)


def avg(a, use):
    block = a[use]
    # Complete period required in each native cell, no renormalised partial phases.
    return np.where(np.isfinite(block).all(0), np.mean(block, axis=0), np.nan)


def bandmean(v, lat, lo, hi):
    lm = (lat >= lo) & (lat < hi)
    w = np.cos(np.deg2rad(lat[lm]))[:, None]
    a = v[lm]
    ok = np.isfinite(a)
    return np.divide(np.sum(np.where(ok, a, 0)*w, 0), np.sum(ok*w, 0),
                     out=np.full(a.shape[1:], np.nan, dtype=a.dtype), where=np.sum(ok*w, 0) > 0)


def regional(v, lat, pressure, lo, hi, plo, phi):
    lm = (lat >= lo) & (lat <= hi)
    pm = (pressure >= plo - 1e-6) & (pressure <= phi + 1e-6)
    a = v[np.ix_(lm, pm)]
    w = np.cos(np.deg2rad(lat[lm]))[:, None] * np.ones(a.shape)
    ok = np.isfinite(a)
    return float(np.sum(np.where(ok, a, 0)*w)/np.sum(ok*w)) if ok.any() else np.nan


def unwrap_parts(c, minimum=3.):
    """No bridge across invalid amplitude levels; anchor each segment independently."""
    valid = np.isfinite(c) & (abs(c) >= minimum)
    phase = np.full(len(c), np.nan)
    ids = np.flatnonzero(valid)
    if len(ids):
        groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
        for group in groups:
            phase[group] = np.rad2deg(np.unwrap(np.angle(c[group])))
    return phase


def calculate(d):
    lat, p, dates = d['lat'], d['pressure'], pd.DatetimeIndex(d['dates'])
    ref = (dates >= REF[0]) & (dates <= REF[1])
    samples = [('pair', -1, 0), ('LT1', 1, 0), ('LT2', 2, 0)]
    samples += [(f'delete{i:02}', -1, i) for i in range(1, 13)]
    plots, rows, phase_rows, cells, dailyrows = {}, [], [], [], []
    regions = [('mid_input', 52.5, 62.5, .0464, .1469),
               ('upper_bridge', 37.5, 47.5, .00464, .01001),
               ('low_return', 22.5, 37.5, .00464, .01001)]
    for name, branch, sector in samples:
        Z, T, Z0, T0 = extract(d, branch, sector)
        refZ0, refT0 = avg(Z0, ref), avg(T0, ref)
        refbg = background(refZ0, refT0, lat, p)
        refamp = [abs(avg(T[..., k], ref)) for k in [0, 1]]
        refzamp = [abs(avg(Z[..., k], ref)) for k in [0, 1]]
        dailyflux = [flux(Z[i, ..., 0], T[i, ..., 0], Z0[i], T0[i], lat, p) for i in range(len(dates))]
        if name in ['pair', 'LT1', 'LT2']:
            for it, day in enumerate(dates):
                b = background(Z0[it], T0[it], lat, p)
                for reg, lo, hi, plo, phi in regions:
                    rr = dict(sample=name, date=str(day.date()), region=reg)
                    rr.update({k: regional(dailyflux[it][k], lat, p, lo, hi, plo, phi) for k in ['Q', 'P', 'W']})
                    dailyrows.append(rr)
                for lo, hi in [(30, 40), (50, 60)]:
                    bm = bandmean(b['U'], lat, lo, hi)
                    inds = [int(np.argmin(abs(np.log(p/x)))) for x in [.1, .01, .00464159]]
                    dailyrows.append(dict(sample=name, date=str(day.date()), region=f'U_{lo}_{hi}',
                        U62=bm[inds[0]], U77=bm[inds[1]], U82=bm[inds[2]],
                        shear82_62=bm[inds[2]]-bm[inds[0]], shear82_77=bm[inds[2]]-bm[inds[1]]))
        for tag, start, end in PERIODS:
            use = (dates >= start) & (dates <= end)
            z0, t0 = avg(Z0, use), avg(T0, use)
            z, t = avg(Z[..., 0], use), avg(T[..., 0], use)
            b = background(z0, t0, lat, p)
            q = flux(z, t, z0, t0, lat, p)
            q5 = flux(z, t, z0, t0, lat, p, points=5)
            daily = {k: avg(np.stack([v[k] for v in dailyflux]), use)
                     for k in ['Q', 'P', 'W', 'Fy', 'Fz', 'div']}
            coherence = abs(z)**2 / avg(abs(Z[..., 0])**2, use)
            tcoherence = abs(t)**2 / avg(abs(T[..., 0])**2, use)
            if name == 'pair':
                for k, v in (b | q | {'height': z0/1000, 'Z': z, 'T': t,
                       'deltaU': b['U']-refbg['U'], 'coherence': coherence,
                       'Tcoherence': tcoherence}).items():
                    plots[tag+'_'+k] = v
                for k in [0, 1]:
                    at = abs(avg(T[..., k], use))
                    plots[f'{tag}_Tamp{k+1}'] = at
                    plots[f'{tag}_dTamp{k+1}'] = at-refamp[k]
                    plots[f'{tag}_dailyTamp{k+1}'] = avg(abs(T[..., k]), use)
                    az = abs(avg(Z[..., k], use))
                    plots[f'{tag}_Zamp{k+1}'] = az
                    plots[f'{tag}_dZamp{k+1}'] = az-refzamp[k]
                    plots[f'{tag}_dailyZamp{k+1}'] = avg(abs(Z[..., k]), use)
                for key in daily:
                    plots[tag+'_daily_'+key] = daily[key]
                for il in range(len(lat)):
                    for ip in range(len(p)):
                        cells.append(dict(period=tag, latitude=lat[il], pressure_hPa=p[ip],
                          height=z0[il,ip]/1000, U=b['U'][il,ip], n2=b['n2'][il,ip],
                          Q=q['Q'][il,ip], P=q['P'][il,ip], div=q['div'][il,ip],
                          daily_Q=daily['Q'][il,ip], Zcoherence=coherence[il,ip],
                          Tamp1=plots[tag+'_Tamp1'][il,ip], Tamp2=plots[tag+'_Tamp2'][il,ip]))
            for reg, lo, hi, plo, phi in regions:
                rr = dict(sample=name, period=tag, region=reg)
                for key in ['Q', 'P', 'W', 'div']:
                    rr[key] = regional(q[key], lat, p, lo, hi, plo, phi)
                rr.update({f'daily_{k}': regional(daily[k], lat, p, lo, hi, plo, phi) for k in ['Q','P','W']})
                rr['P5'] = regional(q5['P'], lat, p, lo, hi, plo, phi)
                rr['Zcoherence'] = regional(coherence, lat, p, lo, hi, plo, phi)
                rr['Tcoherence'] = regional(tcoherence, lat, p, lo, hi, plo, phi)
                rows.append(rr)
            for lo, hi in [(30,40), (40,50), (50,60), (60,70)]:
                uc = -bandmean(q['u'], lat, lo, hi)  # minimum, paper Fig.2d
                phase = unwrap_parts(uc)
                for ip in range(len(p)):
                    phase_rows.append(dict(sample=name, period=tag, band=f'{lo}-{hi}',
                        pressure_hPa=p[ip], height=bandmean(z0/1000,lat,lo,hi)[ip],
                        phase_deg=phase[ip], amplitude_ms=abs(uc[ip])))
    np.savez_compressed(OUT/'period_diagnostics.npz', lat=lat, pressure=p, dates=dates.values, **plots)
    pd.DataFrame(rows).to_csv(OUT/'regional_period_checks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(phase_rows).to_csv(OUT/'absolute_wind_phase_checks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cells).to_csv(OUT/'native_period_fields.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(dailyrows).to_csv(OUT/'pulse_and_shear_chronology.csv', index=False, encoding='utf-8-sig')
    (OUT/'method.json').write_text(json.dumps(dict(periods=PERIODS, reference=REF,
        reference_is_event_baseline_not_multiyear_climatology=True,
        first_period_partly_overlaps_reference=True, maps_are_period_mean_complex_W1=True,
        figure1='GPH W1 and W2 |mean coefficient|; contours relative to 25-30 Dec; historical PNG filename retained',
        figure23='same W1 in both periods, unlike original Fig3 W2',
        diagnostic='Matthias & Ern 2018 Eq2-5, c=0, H=7 km, density proportional to p',
        n2_is_total_refractive_index_not_vertical_m2=True, no_wave_solver=True,
        ep_primary='quadratic flux of period-mean wave; daily quadratic mean exported separately',
        figure4='daily quadratic W1 EP flux averaged over pre 25-30 Dec and post 4-10 Jan; no stationary-mode claim',
        arrow_display='vertical component multiplied by 1000; same normalization in both panels',
        no_gap_fill=True, native_stencil_points=5, phase_minimum_wind_amplitude_ms=3,
        uncertainty='LT branches, delete-longitude-sector and 3/5 derivative settings; not confidence limits',
        stationary_projection_not_unique_mode=True),ensure_ascii=False,indent=2), encoding='utf8')
    frame=pd.DataFrame(rows)
    print(frame[frame['sample']=='pair'].round(3).to_string(index=False))
    print('SAVED',OUT,flush=True)


if __name__ == '__main__':
    calculate(build_cache())
