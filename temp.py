from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import models, pyhsmm, util
from pyhsmm.util.text import progprint_xrange

T = 1000
Ntrue = 4
Nmax = 10

obshypparamss = [
        dict(mu_0=0.,tausq_0=5.,sigmasq_0=1.,nu_0=100.),
        dict(mu_0=20.,tausq_0=5.,sigmasq_0=2.,nu_0=100.),
        ]

durhypparamss = [
        dict(k=10,theta=10.),
        dict(k=30,theta=10.),
        ]

blah = models.factorial([models.factorial_component_hsmm(alpha=6.,gamma=6.,obs_distns=[pyhsmm.observations.scalar_gaussian_nonconj_gelparams(**obshypparams) for hi in range(Ntrue)],dur_distns=[pyhsmm.durations.poisson(**durhypparams) for hi in range(Ntrue)]) for obshypparams,durhypparams in zip(obshypparamss,durhypparamss)])

sumobs, allobs, allstates = blah.generate(T)

# get changepoints
allchangepoints = []
for truemodel in blah.component_models:
    # copied from my code which gets changepoints for a single state sequence
    temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
    changepoints = zip(temp[:-1],temp[1:])
    changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
    allchangepoints.append(changepoints)
changepoints = util.union_changepoints(allchangepoints)
# TODO or i could just estimate them from the data...

newblah = models.factorial([models.factorial_component_hsmm_possiblechangepoints(alpha=6.,gamma=6.,obs_distns=[pyhsmm.observations.scalar_gaussian_nonconj_gelparams(**obshypparams) for hi in range(Nmax)],dur_distns=[pyhsmm.durations.poisson(**durhypparams) for hi in range(Nmax)]) for obshypparams,durhypparams in zip(obshypparamss,durhypparamss)])

newblah.add_data(data=sumobs,changepoints=changepoints)

for itr in progprint_xrange(5):
    newblah.resample(min_extra_noise=1.,max_extra_noise=100.,niter=50)

plt.figure(); plt.plot(blah.states_list[0].museqs); plt.title('truth')
plt.figure(); plt.plot(newblah.states_list[0].museqs); plt.title('sampled')
plt.show()

