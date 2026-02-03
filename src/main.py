from netcdf2dets import readnetcdf, calc_prom_arrays, detections
from track import track1, link_det

import sys

def main(infile, pings):
    print("Hello from ektrack2!")
    # 1. read NetCDF file
    ch = readnetcdf(infile)

    # 2. calculate prominences
    calc_prom_arrays(ch)

    # 3. run tracking
    tracks = []
    for p in pings:
        dets = detections(ch, p, minprom=2.0, minrng=6.0, maxrng=8.0)
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
