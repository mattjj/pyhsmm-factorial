from __future__ import division

from matplotlib import cbook
def union_changepoints(allchangepoints):
    startpoints = sorted(set(cbook.flatten(allchangepoints)))
    return [(startpoint,nextstartpoint) for startpoint,nextstartpoint in zip(startpoints[:-1],startpoints[1:])]

