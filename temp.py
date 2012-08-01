from __future__ import division
import numpy as np
np.seterr(divide='ignore')

import models, pyhsmm

from matplotlib import cbook
def union_changepoints(allchangepoints):
    startpoints = sorted(set(cbook.flatten(allchangepoints)))
    return [(startpoint,nextstartpoint) for startpoint,nextstartpoint in zip(startpoints[:-1],startpoints[1:])]

T = 500


blah = models.factorial([models.factorial_component_hsmm(alpha=6.,gamma=6.,obs_distns=[pyhsmm.observations.scalar_gaussian_nonconj_gelparams(mu_0=0.,tausq_0=5.**2,sigmasq_0=0.5,nu_0=1.) for hi in range(4)],dur_distns=[pyhsmm.durations.poisson() for hi in range(4)]) for grr in range(2)])

sumobs, allobs, allstates = blah.generate(T)

# get changepoints
allchangepoints = []
for truemodel in blah.component_models:
    # copied from my code which gets changepoints for a single state sequence
    temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
    changepoints = zip(temp[:-1],temp[1:])
    changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
    allchangepoints.append(changepoints)
    print len(changepoints)
changepoints = union_changepoints(allchangepoints)
# or i could just estimate them from the data...

newblah = models.factorial([models.factorial_component_hsmm_possiblechangepoints(alpha=6.,gamma=6.,obs_distns=[pyhsmm.observations.scalar_gaussian_nonconj_gelparams(mu_0=0.,tausq_0=10.**2,sigmasq_0=0.5,nu_0=1.) for hi in range(4)],dur_distns=[pyhsmm.durations.poisson() for hi in range(4)]) for grr in range(2)])

newblah.add_data(data=sumobs,changepoints=changepoints)

# newblah.resample(min_extra_noise=1.,max_extra_noise=100.,niter=50)

