from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange

import models
import util as futil

T = 400
Nmax = 10

# observation distributions used to generate data
true_obsdistns_chain1 = [
        pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(
            None,None,None,None, # no hyperparameters since we won't resample
            mu=0,sigmasq=0.01),
        pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(
            None,None,None,None,
            mu=10,sigmasq=0.01),
        ]

true_obsdistns_chain2 = [
        pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(
            None,None,None,None,
            mu=20,sigmasq=0.01),
        pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(
            None,None,None,None,
            mu=30,sigmasq=0.01),
        ]

# observation hyperparameters used during inference
obshypparamss = [
        dict(mu_0=5.,tausq_0=10.**2,sigmasq_0=0.01,nu_0=100.),
        dict(mu_0=25.,tausq_0=10.**2,sigmasq_0=0.01,nu_0=100.),
        ]

# duration hyperparameters used both for data generation and inference
durhypparamss = [
        dict(alpha_0=20*20,beta_0=20.),
        dict(alpha_0=20*75,beta_0=20.),
        ]

truemodel = models.Factorial([models.FactorialComponentHSMM(
        init_state_concentration=2.,
        alpha=2.,gamma=4.,
        obs_distns=od,
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(**durhypparams) for hi in range(len(od))])
    for od,durhypparams in zip([true_obsdistns_chain1,true_obsdistns_chain2],durhypparamss)])

sumobs, allobs, allstates = truemodel.generate(T)

plt.figure(); plt.plot(sumobs); plt.title('summed data')
plt.figure(); plt.plot(truemodel.states_list[0].museqs); plt.title('true decomposition')

### estimate changepoints (threshold should probably be a function of the empirical variance, or something)
changepoints = futil.indicators_to_changepoints(np.concatenate(((0,),np.abs(np.diff(sumobs)) > 1)))
futil.plot_with_changepoints(sumobs,changepoints)

### construct posterior model
posteriormodel = models.Factorial([models.FactorialComponentHSMMPossibleChangepoints(
        init_state_concentration=2.,
        alpha=1.,gamma=4.,
        obs_distns=[pyhsmm.basic.distributions.ScalarGaussianNonconjNIX(**obshypparams) for idx in range(Nmax)],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(**durhypparams) for idx in range(Nmax)],
        trunc=200)
    for obshypparams, durhypparams in zip(obshypparamss,durhypparamss)])

posteriormodel.add_data(data=sumobs,changepoints=changepoints)

nsubiter=25
for itr in progprint_xrange(10):
    posteriormodel.resample_model(min_extra_noise=0.1,max_extra_noise=100.**2,niter=nsubiter)

plt.figure(); plt.plot(posteriormodel.states_list[0].museqs);
plt.title('sampled after %d iterations' % ((itr+1)))

plt.show()

