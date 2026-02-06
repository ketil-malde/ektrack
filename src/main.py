from netcdf2dets import readnetcdf, calc_prom_arrays, detections
from track import track1, link_det

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import xarray_regrid

import sys

def load(infile):
    '''Read the NetCDF file and calculate prominence'''
    ch = readnetcdf(infile)
    calc_prom_arrays(ch)
    return ch

def plot(ch, track=[]):
    '''Generate plots of PC data and prominence'''
    for g, x in ch.items():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

        # Backscatter and prominence
        b = x['backscatter']
        log_norm = mcolors.LogNorm(vmin=b.min().item(), vmax=b.max().item())
        b.T.plot.imshow(ax=ax1, norm=log_norm)
        ax1.set_title("Backscatter")
        x['prominence'].T.plot.imshow(ax=ax2, cmap='viridis')
        ax2.set_title("Prominence")

        # TODO: Plot tracks

        # Finish up 
        fig.suptitle(x.wbtlabel)
        plt.tight_layout()
        plt.show()

def regrid(ch, brighten=1):
    for g in ch.keys():
        f = ch[g].frequency
        print(f)
        eps = 1e-10
        if f == 38000:
            print('38k')
            f38k = np.log10(ch[g]['backscatter'] + eps)
        elif f == 120000:
            print('120k')
            f120k = np.log10(ch[g]['backscatter'] + eps)
        elif f == 200000:
            print('200k')
            f200k = np.log10(ch[g]['backscatter'] + eps)
        else:
            pass

    def normalize_01(da): return brighten * (da - da.min()) / (da.max() - da.min() + eps)

    f38k = normalize_01(f38k)
    f120k = normalize_01(f120k.interp(range=f38k.range, method='linear'))
    f200k = normalize_01(f200k.interp(range=f38k.range, method='linear'))

    ds = xr.concat([f38k, f120k, f200k], dim='color')
    ds.T.plot.imshow(rgb='color', add_colorbar=False)
    plt.show()

def track(ch, pings, minprom=1.0, minrng=6.0, maxrng=8.0):
    '''Run tracking, iterating over pings'''
    tracks = []
    for p in pings:
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

def showtracks(ts):
    for t in ts:
        print('Track:')
        for d in t.detections:
            for e in d:
                print(e)
            print()
        print()


if __name__ == "__main__":
    print('Reading netCDF file: ' + sys.argv[1])
    ds = load(sys.argv[1])
    regrid(ds)
    # ts = track(ds, range(90, 105), minprom=2.5)
    # showtracks(ts)
