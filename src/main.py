from netcdf2dets import readnetcdf, calc_prom_arrays, detections
from track import track1, link_det
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

PLOT = False

import sys

def main(infile, pings):
    # 1. read NetCDF file
    print('Reading netCDF file: ' + infile)
    ch = readnetcdf(infile)

    # 2. calculate prominences - updates its parameter (ch)
    calc_prom_arrays(ch)

    if PLOT:
      for g, x in ch.items():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        b = x['backscatter']
        log_norm = mcolors.LogNorm(vmin=b.min().item(), vmax=b.max().item())
        b.T.plot.imshow(ax=ax1, norm=log_norm)
        ax1.set_title("Backscatter")
        print(x['prominence'].max(), x['prominence'].min())  # prominence can be super-negative, why?
        x['prominence'].T.plot.imshow(ax=ax2, cmap='Greys')
        ax2.set_title("Prominence")
        fig.suptitle(x.wbtlabel)
        plt.tight_layout()
        plt.show()

    # 3. run tracking, iterating over pings
    tracks = []
    for p in pings:
        dets = detections(ch, p, minprom=1.0, minrng=6.0, maxrng=8.0)
        acc = None
        for g, ds in dets.items():
            ds = [[d] for d in ds]
            if not acc:
                acc = ds
            else:
                acc = link_det(acc, ds)
        tracks = track1(tracks, acc)

    return tracks


if __name__ == "__main__":
    ts = main(sys.argv[1], range(100, 200))
    for t in ts:
        print('Track:')
        for d in t.detections:
            for e in d:
                print(e)
            print()
        print()
