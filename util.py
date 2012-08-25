from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cbook

def union_changepoints(allchangepoints):
    startpoints = sorted(set(cbook.flatten(allchangepoints)))
    return [(startpoint,nextstartpoint) for startpoint,nextstartpoint in zip(startpoints[:-1],startpoints[1:])]

def plot_with_changepoints(data,changepoints):
    plt.figure()
    plt.plot(data)
    rect = plt.axis() # xmin xmax ymin ymax
    plt.vlines([c[1] for c in changepoints[:-1]],rect[2],rect[3],color='r',linestyles='dashed')
    plt.axis(rect)

def indicators_to_changepoints(indicators):
    '''
    indicators is 0 everywhere except the index at which a new segment starts

    example:
        indicators = [False, False,  True, False, False, False, False,  True, False]
        +. [(0, 2), (2, 7), (7, 9)]
    '''
    pos, = np.where(np.diff(np.array(indicators,dtype=int)) == 1) # ends of segments
    pos = np.concatenate(((0,),pos + 1,(len(indicators),))) # starts of segments
    return [(a,b) for a,b in zip(pos[:-1],pos[1:])]

