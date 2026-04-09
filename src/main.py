from netcdf2dets import get_detections, load
from detections import Detection, cluster_det
from track import Track, track1

from matplotlib import pyplot as plt, colors as mcolors
import numpy as np
import xarray as xr
from datetime import datetime, timezone

import sys
from typing import Dict, List

def plot(ch: Dict[str, xr.Dataset], tracks: List[Track] = []) -> None:
    '''Generate plots of PC data and prominence'''
    for g, x in ch.items():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

        # Backscatter and prominence
        b = x['backscatter']
        log_norm = mcolors.LogNorm(vmin=b.min().item(), vmax=b.max().item())
        b.T.plot.imshow(ax=ax1, norm=log_norm)
        ax1.set_title("Backscatter")
        ax1.set_aspect("auto")
        x['prominence'].T.plot.imshow(ax=ax2, cmap='viridis')
        ax2.set_title("Prominence")
        ax2.set_aspect("auto")

        # Date handling in Python is a stinking mess
        def t_pings(trd: List[List[Detection]]) -> List[datetime]: return [datetime.fromtimestamp(d[0].time / 1e9, tz=timezone.utc) for d in trd]
        def t_ranges(trd: List[List[Detection]]) -> List[float]: return [d[0].location().z for d in trd]

        # Plot tracks
        if tracks:
            for t in tracks:
                if len(t.detections) > 1:
                    tdata = xr.DataArray(t_ranges(t.detections),
                                         coords={'time': t_pings(t.detections)},
                                         dims=['time'])
                    tdata.plot(ax=ax1, color='red', marker='+')
                    # print(t_pings(t.detections))
                    # ax1.plot(t_ranges(t.detections), t_pings(t.detections))

        # Finish up
        fig.suptitle(x.wbtlabel)
        plt.tight_layout()
        plt.show()

def regrid(ch: Dict[str, xr.Dataset], brighten: float = 1.0) -> xr.DataArray:
    for g in ch.keys():
        f = ch[g].frequency
        eps = 1e-10
        if f == 38000:
            f38k = np.log10(ch[g]['backscatter'] + eps)
        elif f == 120000:
            f120k = np.log10(ch[g]['backscatter'] + eps)
        elif f == 200000:
            f200k = np.log10(ch[g]['backscatter'] + eps)
        else:
            pass

    def normalize_01(da): return brighten * (da - da.min()) / (da.max() - da.min() + eps)

    f38k = normalize_01(f38k)
    f120k = normalize_01(f120k.interp(range=f38k.range, method='linear'))
    f200k = normalize_01(f200k.interp(range=f38k.range, method='linear'))

    return xr.concat([f38k, f120k, f200k], dim='color')


# Parameter
max_age = 3.1  # seconds

def track(ch: Dict[str, xr.Dataset], pings: range, minprom: float = 2.0, minrng: float = 6.0, maxrng: float = 8.0) -> List[Track]:
    '''Run tracking, iterating over pings'''
    tracks = []
    old_tracks = []
    for p in pings:
        # print('Processing ping:', p, '#tracks:', len(tracks), end='')
        dets = get_detections(ch, p, minprom=minprom, minrng=minrng, maxrng=maxrng)

        # Prune aged tracks
        for g, d in dets.items():
            dtime = d[0].time / 1e9
            break
        # print('dtime:', dtime, end='\r')
        for i, t in enumerate(tracks):
            ttime = t.last()[0].time / 1e9
            if dtime - ttime > max_age:
                old_tracks.append(tracks.pop(i))
        acc = cluster_det(dets)
        tracks = track1(tracks, acc)
    # print()
    return tracks + old_tracks

def showtracks(ts: List[Track]) -> None:
    for t in ts:
        print('Track:')
        for d in t.detections:
            for e in d:
                print(e)
            print()
        print(f'Velocity: {t.velocity}')
        print()


if __name__ == "__main__":
    print('Reading netCDF file: ' + sys.argv[1])
    ds = load(sys.argv[1])

    # rds = regrid(ds)
    # rds.T.plot.imshow(rgb='color', add_colorbar=False)
    # plt.show()
    ds1: Dict[str, List[Detection]] = get_detections(ds, 100, minrng=6.0, maxrng=8.0, minprom=2.0)
    ds2: Dict[str, List[Detection]] = get_detections(ds, 101, minrng=6.0, maxrng=8.0, minprom=2.0)
    for d in [ds1, ds2]:
        for s, z in d.items():
            print(s)
            for z0 in z: print(z0)
        print()

    acc = cluster_det(ds1)
    for a in acc:
        print()
        for aa in a:
            print(aa)

    print('\n*** Tracking ***\n')
    ts = []
    ts = track(ds, range(100, 200), minprom=2, minrng=6.0, maxrng=6.5)
    def sortidx(t): return t.detections[0][0].range  # workaround for mypy
    showtracks(sorted(ts, key=sortidx))

    # exit(0)
    # ts = track(ds, range(100, 150), minprom=3, minrng=0, maxrng=99999)
    # plot(ds, ts)

    # Todo - tests:
    # 1. number of tracks
    # 2. number of singleton detections
    # 3. track lengths?
    # 4. number of detections
