from netcdf2dets import readnetcdf, calc_prom_arrays, detections
from track import track1, link_det
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import sys

def load(infile):
    '''Read the NetCDF file and calculate prominence'''
    ch = readnetcdf(infile)
    calc_prom_arrays(ch)
    return ch

def plot(ch, track):
    '''Generate plots of PC data and prominence'''
    for g, x in ch.items():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
        b = x['backscatter']
        log_norm = mcolors.LogNorm(vmin=b.min().item(), vmax=b.max().item())
        b.T.plot.imshow(ax=ax1, norm=log_norm)
        ax1.set_title("Backscatter")
        x['prominence'].T.plot.imshow(ax=ax2, cmap='Greys')
        ax2.set_title("Prominence")
        fig.suptitle(x.wbtlabel)
        plt.tight_layout()
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
    ts = track(ds, range(90, 105), minprom=2.5)
    showtracks(ts)
