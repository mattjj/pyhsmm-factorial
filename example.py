from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import models, pyhsmm
from pyhsmm.util.text import progprint_xrange
import util as futil

T = 500
Ntrue = 2
Nmax = 10

obshypparamss = [
        dict(mu_0=0.,tausq_0=5.,sigmasq_0=0.01,nu_0=100.),
        dict(mu_0=20.,tausq_0=5.,sigmasq_0=0.01,nu_0=100.),
        ]

durhypparamss = [
        dict(alpha_0=20*20,beta_0=20.),
        dict(alpha_0=20*60,beta_0=20.),
        ]

truemodel = models.Factorial([models.FactorialComponentHSMM(
    init_state_concentration=2.,
    alpha=2.,gamma=4.,
    obs_distns=[pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(**obshypparams)
        for hi in range(Ntrue)],
    dur_distns=[pyhsmm.basic.distributions.PoissonDuration(**durhypparams)
        for hi in range(Ntrue)])
    for obshypparams,durhypparams in zip(obshypparamss,durhypparamss)])

sumobs, allobs, allstates = truemodel.generate(T)

plt.figure(); plt.plot(sumobs); plt.title('summed data')
plt.figure(); plt.plot(truemodel.states_list[0].museqs); plt.title('true decomposition')

### estimate changepoints
changepoints = futil.indicators_to_changepoints(np.concatenate(((0,),np.abs(np.diff(sumobs)) > 1)))
futil.plot_with_changepoints(sumobs,changepoints)

### construct posterior model
posteriormodel = models.Factorial([models.FactorialComponentHSMMPossibleChangepoints(
    init_state_concentration=2.,
    alpha=2.,gamma=4.,
    obs_distns=[pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(**obshypparams)
        for hi in range(Nmax)],
    dur_distns=[pyhsmm.basic.distributions.PoissonDuration(**durhypparams)
        for hi in range(Nmax)])
    for obshypparams,durhypparams in zip(obshypparamss,durhypparamss)])

posteriormodel.add_data(data=sumobs,changepoints=changepoints)

nsubiter=50
for itr in progprint_xrange(4):
    posteriormodel.resample_model(min_extra_noise=0.1,max_extra_noise=100.**2,niter=nsubiter)
    plt.figure(); plt.plot(posteriormodel.states_list[0].museqs);
    plt.title('sampled after %d iterations' % ((itr+1)*nsubiter))

plt.show()

