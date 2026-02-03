from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from math import exp, sin, sqrt
from datetime import datetime
import numpy as np
import sys

from typing import List, Tuple, Dict, Any

# 5 cm discrepancy reduces score with 1/e
range_sigma = 0.05
# angle of 2 degrees give same penalty as above
angle_sigma = 2
rank_scale = 10  # ranks count 0.5, 0.45, 0.42, 0.38...

debug = False
PI = 3.1415926


@dataclass
class Detection:
    pingno: int
    time: int
    freq: str
    range: float  # meters
    theta: float  # degrees
    phi: float
    rank: int

    def __str__(self):
        x0, y0, z0 = self.location()
        return f'D {self.pingno} {self.time} {self.freq // 1000:3d}kHz Score: {self.rank:5.2f}\tR:{self.range:6.2f}  \u0398:{self.theta:4.1f}  \u0278:{self.phi:4.1f}\tXYZ: [{x0:5.2f}, {y0:5.2f}, {z0:6.2f}]'

    def location(self, mru=None):
        # todo: use Yngve's formula: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-coordinate-transformation/blob/main/transformCoordinates.py
        '''Convert to 3D coordinates'''
        x = self.range * sin(PI * self.theta / 180)
        y = self.range * sin(PI * self.phi / 180)
        z = sqrt(self.range * self.range - x * x - y * y)
        return x, y, z

    def from_location(self, x, y, z):
        '''Create from 3D coordinates'''
        pass


def detection_similarity(det1, det2, uncertainty=1):
    '''Distance between detections in pings with given time difference'''
    d_theta = det1.theta - det2.theta
    d_phi = det1.phi - det2.phi
    d_range = det1.range - det2.range
    d_rank = rank_scale / (rank_scale * 2 + det1.rank) + rank_scale / (rank_scale * 2 + det2.rank)
    # one if perfect match (all zero), towards zero if deviation is too large
    # uncertainty less than one means the distribution is wider but lower

    # todo: downweigh the angles a lot!
    return sqrt(uncertainty) * d_rank * exp(- (d_range**2 / range_sigma**2 + d_theta**2 / angle_sigma**2 + d_phi**2 / angle_sigma**2) * uncertainty)


def detection_max_similarity(detlist1, detlist2, uncertainty=1):
    res = 0
    for d1 in detlist1:
        for d2 in detlist2:
            sim = detection_similarity(d1, d2, uncertainty)
            if sim > res: res = sim
    return res


def mkdet(fields):
    '''Parse a line of CSV input into a detection.'''
    def parsetime(s):
        x = datetime.strptime(s, "%M:%S.%f")
        return int((x.minute * 60 + x.second) * 1e6 + x.microsecond)
    return Detection(pingno=int(fields[0]), time=parsetime(fields[1]), freq=int(fields[3]), range=float(fields[4]),
                     theta=float(fields[5]), phi=float(fields[6]), rank=0)


# ########### Tracking across frequencies ######################

def link_det(dets1, dets2, threshold=0.0005):
    '''Find optimal pairing'''
    # calc matrix and return list of multi-dets
    ndets1, ndets2 = len(dets1), len(dets2)
    mx = np.zeros((ndets1, ndets2))
    # todo: optimize by searching in a window?
    for t in range(ndets1):
        for d in range(ndets2):
            mx[t, d] = detection_max_similarity(dets1[t], dets2[d])
    ind1, ind2 = linear_sum_assignment(mx, maximize=True)

    res = []
    d1s = [dets1[i] for i in ind1]
    d2s = [dets2[i] for i in ind2]
    d1rest = [dets1[i] for i in range(len(dets1)) if i not in ind1]
    d2rest = [dets2[i] for i in range(len(dets2)) if i not in ind2]
    for i in range(len(ind1)):
        if mx[ind1[i], ind2[i]] > threshold:
            res.append(d1s[i] + d2s[i])
        else:
            # if debug and mx[ind1[i], ind2[i]] > threshold/10:
            #     print('DEBUG:\n', d1s[i], '\n', d2s[i], '\n', mx[ind1[i], ind2[i]], file=sys.stderr)
            res.append(d1s[i])
            res.append(d2s[i])

    res = res + d1rest + d2rest
    return sorted(res, key=lambda x: x[0].range)


# ############### Tracking across time ########################

# Sigh.  Time to make location a class?
def avgtuple(tups):
    if tups: return (sum([a[0] for a in tups]) / len(tups),
                     sum([a[1] for a in tups]) / len(tups),
                     sum([a[2] for a in tups]) / len(tups))
    else: return (0, 0, 0)

def difftuple(tup1, tup2):
    return (tup1[0] - tup2[0], tup1[1] - tup2[1], tup1[2] - tup2[2])

def addtuple(tup1, tup2):
    return (tup1[0] + tup2[0], tup1[1] + tup2[1], tup1[2] + tup2[2])

def abstuple(tup1):
    return sqrt(tup1[0]**2 + tup1[1]**2 + tup1[2]**2)


@dataclass
class Track:
    # also track adjustments?
    detections: List[Detection]

    # for predictions
    velocity: Tuple[float, float, float]
    certainty: float

    def __init__(self, det):
        self.detections = [det]
        self.velocity = (0, 0, 0)
        self.certainty = 0

    def _delta_loc(self):
        '''Estimate movement from penultimate to last detection'''
        # todo need to interpolate time as well
        d1 = {d.freq: d for d in self.detections[-1]}
        d2 = {d.freq: d for d in self.detections[-2]}
        # accumulate location diffs
        acc = []
        for f in set(list(d1.keys()) + list(d2.keys())):
            if f in d1.keys() and f in d2.keys(): acc.append(difftuple(d1[f].location(), d2[f].location()))
        # and calculate average
        return avgtuple(acc)

    def predict(self, time=None, avgvel=(0, 0, 0), avgrot=(0, 0)):
        '''Update velocity and certainty and predict next detection from a track'''
        assert isinstance(avgvel, tuple)
        assert isinstance(avgrot, tuple)
        if len(self.detections) == 1:
            # Predict avg of other tracks? with high certainty
            self.certainty = 0
            self.velocity = avgvel
        if len(self.detections) >= 2:
            # Linear extrapolation, high certainty
            delta = self._delta_loc()
            if len(self.detections) == 2:
                self.certainty = 0
                self.velocity = delta
            else:
                # Linear extrapolation of last two with estimated certainty from third
                self.certainty = 0.5 * self.certainty + 0.5 * abstuple(difftuple(delta,self.velocity))
                self.velocity = delta
        avgloc = avgtuple([d.location() for d in self.detections[-1]])
        return addtuple(avgloc, self.velocity)

    def summarize(self):
        '''Generate a finished track output'''
        pass


@dataclass
class Tracks:
    track: List[Track]
    mru: List[Any]  # roll, pitch, heave?
    offsets: Dict[Any, Any]     # frequency -> location


def similarity_locations(l1, l2):
    dist2 = (l1[0] - l2[0])**2 + (l1[1] - l2[1])**2 + (l1[2] - l2[2])**2
    return exp(-dist2 / 0.01)


# What type is detections here? List of detections grouped by frequency?
def track1(tracks: List[Track], detections: List[List[Detection]], threshold: float = 1e-9) -> List[Track]:
    # link tracks to detections (high confidence)
    ntracks, ndets = len(tracks), len(detections)
    mx = np.empty((ntracks, ndets))
    for t in range(ntracks):
        t_pred = tracks[t].predict()
        for d in range(ndets):
            mx[t, d] = similarity_locations(t_pred, avgtuple([x.location() for x in detections[d]]))
    tind, dind = linear_sum_assignment(mx, maximize=True)

    # todo: adjust for average movement (include MRU)
    # todo: predict locations (include older tracks) and recalculate

    updatedtracks = []
    tmatch = [tracks[i] for i in tind]
    dmatch = [detections[i] for i in dind]
    for i in range(len(tind)):
        if mx[tind[i], dind[i]] > threshold:
            t = tmatch[i]
            t.detections.append(dmatch[i])
            updatedtracks.append(t)
        else:
            # print('Not below threshold:', mx[tind[i], dind[i]], dmatch[i], tmatch[i])
            updatedtracks.append(tmatch[i])
            updatedtracks.append(Track(dmatch[i]))

    # add unmatched detections
    trest = [tracks[i] for i in range(len(tracks)) if i not in tind]
    # print('Trest:', trest)
    drest = [detections[i] for i in range(len(detections)) if i not in dind]
    # print('Drest:', drest)

    return updatedtracks + trest + [Track(d) for d in drest]


# ############################################################################
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
            # for d in ds: print(d) # what the FLYING FUCK?!
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
    sorted_tracks = sorted(tracks, key=sortidx)

    print('--------------------------------------------------')
    for t in sorted_tracks:
        print('Track:')
        for d in t.detections:
            for e in d:
                print(e)
            print()
        print()
