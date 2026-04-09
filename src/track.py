from __future__ import annotations
from typing import List, Optional  # Union, Iterable, Callable, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from math import exp
import numpy as np

from detections import Location, Detection, avgloc
from detections import link_det, mkdet

debug = False

# ############### Tracking across time ########################

@dataclass
class Track:
    # also track adjustments?
    detections: List[List[Detection]]

    # for predictions:
    velocity: Optional[Location]
    # certainty: float

    def __init__(self, det: List[Detection]):
        self.detections = [det]
        self.velocity = None
        # self.certainty = 0

    def last(self):
        return self.detections[-1]

    def append(self, dets):
        # first order velocity
        self.velocity = _velocity(self.last(), dets)
        self.detections.append(dets)

    def summarize(self):
        '''Generate a finished track output'''
        pass

def _pairs(d1, d2):
    '''Link detections by frequency'''
    pairs = {}
    d1f = {d.freq: d for d in d1}
    d2f = {d.freq: d for d in d2}
    for fk in set(list(d1f.keys()) + list(d2f.keys())):
        if fk in d1f.keys() and fk in d2f.keys():
            pairs[fk] = (d1f[fk], d2f[fk])
    return pairs

def average(lst): return sum(lst) / len(lst)

def _velocity(d1, d2):
    ps = _pairs(d1, d2)
    if ps:
        locdiffs = [p2.location() - p1.location() for (p1, p2) in ps.values()]
        tdiffs = [(p2.time - p1.time) / 1e9 for (p1, p2) in ps.values()]
        return avgloc(locdiffs).scale(1 / average(tdiffs))  # whops: todo: divide individually
    else:
        loc1 = avgloc([d.location() for d in d1])
        loc2 = avgloc([d.location() for d in d2])
        t1 = average([d.time for d in d1]) / 1e9
        t2 = average([d.time for d in d2]) / 1e9
        return (loc2 - loc1).scale(1 / (t2 - t1))

# calculate a gaussian pdf, scaled to 2\pi (or something)
def sigmoid(x2, sd): return exp(-x2 / (sd * sd)) / sd


# Parameters
fspectrum_sd = 1
matched_f_z_sd = 0.05
matched_f_xy_sd = 0.15
avgloc_z_sd = 0.1
avgloc_xy_sd = 0.2
score_sd = 0.5

def fspec_sim_squared(d1: List[Detection], d2: List[Detection]) -> float:
    '''Square dotproduct between detection score by frequency'''
    ssq = sum([x.score * y.score for (x, y) in _pairs(d1, d2).values()])
    return sigmoid(1 / (0.0001 + ssq), fspectrum_sd)

def score_similarity(d1: List[Detection], d2: List[Detection]) -> float:
    '''Direct score similarity over matched frequencies (higher when scores are close and high).'''
    ps = _pairs(d1, d2)
    if not ps:
        return 0.0
    diffsq = sum([(a.score - b.score) * (a.score - b.score) for (a, b) in ps.values()])
    strengths = sum([(a.score + b.score) for (a, b) in ps.values()])
    return sigmoid(diffsq, score_sd) * (1.0 + strengths)

def location_difference(trdet: List[Detection], det: List[Detection], velocity: Optional[Location] = None) -> Optional[float]:
    '''Calculate similiarty score between a track and a new detection'''
    ps = _pairs(trdet, det)
    if not ps: return None

    def tdelta(d1, d2): return (d2.time - d1.time) / 1e9
    plocs = [b.location() - a.location() - (velocity.scale(tdelta(a, b)) if velocity else Location(0, 0, 0)) for (a, b) in ps.values()]
    zsquares = sum([loc.z * loc.z for loc in plocs])
    xysquares = sum([loc.x * loc.x + loc.y * loc.y for loc in plocs])
    return sigmoid(zsquares, matched_f_z_sd) * sigmoid(xysquares, matched_f_xy_sd)

def avg_loc_diff(trdet: List[Detection], det: List[Detection], velocity: Optional[Location] = None) -> float:
    '''Calculate similarity of avg location, return z and xy separately'''
    tloc = avgloc([x.location() for x in trdet])
    dloc = avgloc([y.location() for y in det])
    tdelta = average([d.time for d in det]) / 1e9 - average([d.time for d in trdet]) / 1e9
    predicted_tloc = tloc + (velocity.scale(tdelta) if velocity else Location(0, 0, 0))
    diff = dloc - predicted_tloc
    return sigmoid(diff.z * diff.z, avgloc_z_sd) * sigmoid(diff.x * diff.x + diff.y * diff.y, avgloc_xy_sd)

def track_similarity(tr: Track, det: List[Detection]) -> float:
    d0 = tr.last()
    ret = (1 + fspec_sim_squared(d0, det))       # up to 100% bonus for freq score match
    ret = ret * (1 + score_similarity(d0, det))  # bonus for direct signal-strength consistency
    locscore = location_difference(tr.last(), det, tr.velocity)
    if locscore is not None:
        ret = ret * locscore
    else:
        ret = ret * avg_loc_diff(tr.last(), det, tr.velocity)
    return ret

def track1(tracks: List[Track], detections: List[List[Detection]], threshold: float = 1e-6) -> List[Track]:
    # link tracks to detections (high confidence)
    ntracks, ndets = len(tracks), len(detections)
    mx = np.empty((ntracks, ndets))
    for t in range(ntracks):
        for d in range(ndets):
            mx[t, d] = track_similarity(tracks[t], detections[d])
    tind, dind = linear_sum_assignment(mx, maximize=True)

    # todo: add velocity, time! and uncertainty
    # todo: adjust for average movement (include MRU)
    # todo: predict locations (include older tracks) and recalculate

    updatedtracks = []
    tmatch = [tracks[i] for i in tind]
    dmatch = [detections[i] for i in dind]
    for i in range(len(tind)):
        if mx[tind[i], dind[i]] > threshold:
            t = tmatch[i]
            if debug:
                print('Appended by score:', mx[tind[i], dind[i]])
                for e in t.last(): print(e)
                print('--')
                for e in dmatch[i]: print(e)
                print()
            t.append(dmatch[i])
            updatedtracks.append(t)
        else:
            if debug:
                print('Not above threshold:', mx[tind[i], dind[i]])
                for e in dmatch[i]: print(e)
                print('--')
                for e in tmatch[i].last(): print(e)
                print()
            updatedtracks.append(tmatch[i])
            updatedtracks.append(Track(dmatch[i]))

    # add unmatched detections
    trest = [tracks[i] for i in range(len(tracks)) if i not in tind]
    # print('Trest:', trest)
    drest = [detections[i] for i in range(len(detections)) if i not in dind]
    # print('Drest:', drest)

    return updatedtracks + trest + [Track(d) for d in drest]


# ### Testing #########################################################################
import csv
from itertools import groupby

def _genmdet(dets):  # iterator(Detection) -> list(list(Detection))
    '''Group across frequencies from one ping.'''
    # input is an iterator of detections from one ping
    acc = None
    for f, ds in groupby(dets, key=lambda x: x.freq):
        d1 = [[d] for d in ds]  # wrap in lists
        if not acc:
            acc = d1
        else:
            acc = link_det(acc, d1)
    return acc

def _readcsvfile():
    # detfile = 'DetectedSingleTargets(in).csv'
    # detfile = 'minitest.csv'
    detfile = 'detections_short.csv'
    with open(detfile, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        next(r, None)  # skip header

        # Match detections between transducers
        ps = []
        for k, ds in groupby(map(mkdet, r), key=lambda x: x.pingno):
            print('Ping:', k)
            p0 = _genmdet(ds)  # NB: destructive on ds
            for d in p0:
                for x in d:
                    print(x)
                print()
            ps.append(p0)
    return ps


if __name__ == '__main__':
    ps = _readcsvfile()
    # build tracks from ps
    tracks: list[Track] = []
    for p in ps:
        tracks = track1(tracks, p)

    def sortidx(t): return t.detections[0][0].range  # workaround for mypy
    sorted_tracks: List[Track] = sorted(tracks, key=sortidx)

    print('--------------------------------------------------')
    for t in sorted_tracks:
        print('Track:')
        for d in t.detections:
            for e in d:
                print(e)
            print()
        print()
