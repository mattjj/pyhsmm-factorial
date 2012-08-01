from __future__ import division
import numpy as np
np.seterr(divide='ignore')
import models, pyhsmm

blah = models.factorial([models.factorial_component_hsmm(alpha=6.,gamma=6.,obs_distns=[pyhsmm.observations.scalar_gaussian_nonconj_gelparams(mu_0=0.,tausq_0=5.**2,sigmasq_0=0.5,nu_0=1.) for hi in range(4)],dur_distns=[pyhsmm.durations.poisson() for hi in range(4)]) for grr in range(2)])

sumobs, allobs, allstates = blah.generate(500)

# get changepoints

newblah = models.factorial([models.factorial_component_hsmm(alpha=6.,gamma=6.,obs_distns=[pyhsmm.observations.scalar_gaussian_nonconj_gelparams(mu_0=0.,tausq_0=10.**2,sigmasq_0=0.5,nu_0=1.) for hi in range(4)],dur_distns=[pyhsmm.durations.poisson() for hi in range(4)]) for grr in range(2)])

newblah.add_data(sumobs)

newblah.resample(min_extra_noise=1.,max_extra_noise=100.,niter=50)

