from netcdf2dets import readnetcdf, calc_prom_arrays, detections
from track import track1, link_det, Track, Detection

from matplotlib import pyplot as plt, colors as mcolors
import numpy as np
import xarray as xr
from datetime import datetime, timezone

import sys
from typing import Dict, List

def load(infile: str) -> Dict[str, xr.Dataset]:
    '''Read the NetCDF file and calculate prominence'''
    ch = readnetcdf(infile)
    calc_prom_arrays(ch)
    return ch

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

def track(ch: Dict[str, xr.Dataset], pings: range, minprom: float = 1.0, minrng: float = 6.0, maxrng: float = 8.0) -> List[Track]:
    # todo: prune old tracks and maybe singletons?
    
    '''Run tracking, iterating over pings'''
    tracks = []
    for p in pings:
        print(p, len(tracks), end='\r')
        dets = detections(ch, p, minprom=minprom, minrng=minrng, maxrng=maxrng)
        acc = None
        for g, ds in dets.items():
            ds = [[d] for d in ds]
            if not acc:
                acc = ds
            else:
                acc = link_det(acc, ds)
        tracks = track1(tracks, acc)

    return tracks

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
    ds1: Dict[str, List[Detection]] = detections(ds, 100, minrng=6.0, maxrng=8.0, minprom=2.0)
    ds2: Dict[str, List[Detection]] = detections(ds, 101, minrng=6.0, maxrng=8.0, minprom=2.0)
    for d in [ds1, ds2]:
        for s, z in d.items():
            print(s)
            for z0 in z: print(z0)
        print()

    acc = None
    for g, dd in ds1.items():
        dd = [[d] for d in dd]
        if not acc:
            acc = dd
        else:
            acc = link_det(acc, dd)
    for a in acc:
        print()
        for aa in a:
            print(aa)

    print('\n*** Tracking ***\n')
    ts = []
    ts = track(ds, range(100, 102), minprom=2, minrng=6.0, maxrng=8.0)
    def sortidx(t): return t.detections[0][0].range  # workaround for mypy
    showtracks(sorted(ts, key=sortidx))

    # exit(0)
    # ts = track(ds, range(100, 150), minprom=3, minrng=0, maxrng=99999)
    # plot(ds, ts)
