from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt

import pyhsmm
import models, util
import pyhsmm.plugins.factorial.hierarchical as hierarchical
from pyhsmm.util.text import progprint_xrange

### generate some data
T = 500
N = 2

obs_hypparams = dict(mu_0=0.,tausq_0=10**2,sigmasq_0=0.05**2,nu_0=100.)
dur_hypparams = dict(alpha_0=20*60,beta_0=20.)

true_obs_distns = [pyhsmm.distributions.ScalarGaussianNonconjNIX(**obs_hypparams) for state in range(N)]
true_dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(N)]

truemodel = pyhsmm.models.HSMM(alpha=2.,gamma=6.,
                               obs_distns=true_obs_distns,
                               dur_distns=true_dur_distns)

# perturb and sample
datas = []
for itr in range(3):
    for o in truemodel.obs_distns:
        o.mu += 0.5*np.random.randn()

    datas.append(truemodel.generate(T)[0])

for data in datas:
    plt.plot(data)

plt.show()

assert raw_input('proceed? ').lower() == 'y'

### train new hsmms
Nmax = 10

obs_parents = [hierarchical.HierarchicalGaussian(
    0,100**2, # means can be all over the place
    2**2,1.1,  # unceratain about within-class variance, but not high
    10,0.01**2,  # obs noise is usually quite low
    ) for state in range(Nmax)]

dur_parents = [hierarchical.HierarchicalNegativeBinomial(1,1,0.1,1,0.1,1,0.1)
        for state in range(Nmax)]

h = models.HierarchicalHSMM(pyhsmm.models.HSMMPossibleChangepoints,2,6,obs_parents,dur_parents)

for data in datas:
    changepoints = util.indicators_to_changepoints(np.concatenate(((0,),np.abs(np.diff(data)) > 1.)))
    h.new_instance().add_data(data,changepoints)
    for itr in progprint_xrange(20):
        h._instances[-1].resample_model()

### resample the whole enchilada
# for itr in progprint_xrange(10):
#     h.resample_model()


# TODO

# set iter to be small after initialization
# instead of resampling parent every time, just do that on the first go-round
