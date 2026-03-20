from __future__ import annotations
from typing import List, Any, Optional, Union  # Iterable, Callable, Tuple, Dict

from datetime import datetime
from dataclasses import dataclass
from math import sin, sqrt, exp
import numpy as np
from scipy.optimize import linear_sum_assignment

# debug = False
PI = 3.1415926

@dataclass
class Location:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __sub__(self, other: Location) -> Location: return Location(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: Location) -> Location:
        return Location(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other: Union[int, Location]) -> Location:
        if other == 0:
            return self
        else:
            return self.__add__(other)  # type: ignore # Other will be Location in actual use

    def __str__(self) -> str: return f'[{self.x:5.2f}, {self.y:5.2f}, {self.z:6.2f}]'
    def magnitude2(self) -> float: return self.x**2 + self.y**2 + self.z**2
    def scale(self, scalar: float) -> Location: return Location(self.x * scalar, self.y * scalar, self.z * scalar)

def avgloc(locs: List[Location]) -> Location:
    if locs:
        s = sum(locs)
        return s.scale(1.0 / len(locs))  # type: ignore
    else:
        return Location(0, 0, 0)


@dataclass
class Detection:
    pingno: int   # Current ping number
    time: int     # Timestamp in nanoseconds (since epoch)
    freq: int     # Frequency in Hz
    range: float  # Range in meters
    theta: float  # degrees
    phi: float
    score: float

    def __str__(self):
        ts, tus = divmod(self.time // 1000, 1000000)
        return f'D {self.pingno} {ts}.{tus:06} {self.freq // 1000:3d}kHz Score: {self.score:5.2f}\tR:{self.range:6.2f}  \u0398:{self.theta:4.1f}  \u0278:{self.phi:4.1f}\tXYZ: {self.location()}'

    def location(self, mru: Optional[Any] = None) -> Location:
        # todo: use Yngve's formula: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-coordinate-transformation/blob/main/transformCoordinates.py
        '''Convert to 3D coordinates'''
        x = self.range * sin(PI * self.theta / 180)
        y = self.range * sin(PI * self.phi / 180)
        z = sqrt(max(self.range * self.range - x * x - y * y, 0))
        return Location(x, y, z)

    def from_location(self, x: float, y: float, z: float) -> None:
        '''Create from 3D coordinates'''
        pass


# Detection clustering parameters ---
range_sigma = 0.02  # 2 cm discrepancy reduces score with 1/e
angle_sigma = 2  # angle of 2 degrees give same penalty as above

def detection_similarity(det1: Detection, det2: Detection, uncertainty: float = 1.0) -> float:
    '''Distance between detections in pings with given time difference'''
    d_theta = det1.theta - det2.theta
    d_phi = det1.phi - det2.phi
    d_range = det1.range - det2.range
    d_score = det1.score + det2.score
    # one if perfect match (all zero), towards zero if deviation is too large
    # uncertainty less than one means the distribution is wider but lower

    # todo: downweigh the angles a lot!
    return sqrt(uncertainty) * d_score * exp(- (d_range**2 / range_sigma**2 + d_theta**2 / angle_sigma**2 + d_phi**2 / angle_sigma**2) * uncertainty)

def detection_max_similarity(detlist1: List[Detection], detlist2: List[Detection], uncertainty: float = 1.0) -> float:
    '''Max similiarity over all pairs of detections'''
    # shouldn't this be for fixed frequencies?  Or weighted by frequency equality/similarity?
    res = 0
    for d1 in detlist1:
        for d2 in detlist2:
            sim = detection_similarity(d1, d2, uncertainty)
            if sim > res: res = float(sim)  # ensure float for mypy
    return res

def mkdet(fields):
    '''Parse a line of CSV input into a detection.'''
    def parsetime(s):
        x = datetime.strptime(s, "%M:%S.%f")
        return int((x.minute * 60 + x.second) * 1e6 + x.microsecond)
    return Detection(pingno=int(fields[0]), time=parsetime(fields[1]), freq=int(fields[3]), range=float(fields[4]),
                     theta=float(fields[5]), phi=float(fields[6]), score=0)

def link_det(dets1: List[List[Detection]], dets2: List[List[Detection]], threshold: float = 0.0005) -> List[List[Detection]]:
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

# Maybe: class Detections = Dict [freq] -> Detection?


if __name__ == '__main__':
    # Run tests/benchmark across parameters
    
