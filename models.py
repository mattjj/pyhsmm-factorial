from __future__ import division
import numpy as np

import pyhsmm

###################################
#  overall problem wrapper class  #
###################################


class factorial(object):
    def __init__(self,component_models):
        self.component_models = component_models # should be a list of factorial_component models

        self.states_list = [] # a list of factorial_allstates

    def add_data(self,data,**kwargs):
        # pass in state dimensions so that museqs and varseqs can be maintained
        # kwargs is for changepoints
        self.states_list.append(pyhsmm.plugins.factorial.factorial_allstates(data=data,component_models=self.component_models,**kwargs))

    def resample(self,**kwargs):
        # tell each states object to resample each of its component state chains
        # (marginalizing out the component emissions)
        # this call will also delete any instantiated component emissions (in
        # principle)
        # kwargs is for any temperature schedule stuff
        for s in self.states_list:
            s.resample(**kwargs)

        # then resample component emissions so that the other models can be
        # resampled
        for s in self.states_list:
            s.instantiate_component_emissions()

        # resample component models (this call will not cause any states objects
        # referenced by self.states_list to resample, but the parameter
        # resampling involved in resampling these models will need the component
        # emissions)
        for c in self.component_models:
            c.resample()

    def generate(self,T,keep=True): # TODO TODO
        # this will be good for synthetic data testing
        raise NotImplementedError

    def plot(self,color=None): # TODO
        # this is ALWAYS useful
        raise NotImplementedError


######################################
#  classes for the component models  #
######################################

# NOTE: component_models must have scalar gaussian observation
# distributions! this code, which references the same cached means and vars as
# the states, requires it!
class factorial_component_hsmm(pyhsmm.models.hsmm):
    def __init__(self,**kwargs): # no explicit parameter naming because DRY
        assert 'obs_distns' in kwargs
        obs_distns = kwargs['obs_distns']
        self.means, self.vars = np.zeros(len(obs_distns)), np.zeros(len(obs_distns))
        for idx, distn in enumerate(obs_distns):
            assert isinstance(distn,pyhsmm.distributions.observations.scalar_gaussian),\
                    'Factorial model components must have scalar Gaussian observation distributions!'
            distn.mubin = self.means[idx,...]
            distn.sigmasqbin = self.vars[idx,...]
        super(factorial_component_hsmm,self).__init__()

    def add_factorial_sumdata(self,data,**kwargs):
        assert data.ndim == 1 or data.ndim == 2
        data = np.reshape(data,(-1,1))
        self.states_list.append(
                pyhsmm.plugins.factorial.states.factorial_component_hsmm_states(
                    data=data,
                    means=self.means,
                    vars=self.vars,
                    T=data.shape[0],
                    state_dim=len(self.obs_distns),
                    obs_distns=self.obs_distns,
                    dur_distns=self.dur_distns,
                    transition_distn=self.trans_distn,
                    initial_distn=self.init_state_distn,
                    trunc=self.trunc,
                    **kwargs))

class factorial_component_hsmm_possiblechangepoints(factorial_component_hsmm):
    def add_factorial_sumdata(self,data,changepoints,**kwargs):
        assert data.ndim == 1 or data.ndim == 2
        data = np.reshape(data,(-1,1))
        self.states_list.append(
                pyhsmm.plugins.factorial.states.factorial_component_hsmm_states(
                    data=data,
                    changepoints=changepoints,
                    means=self.means,
                    vars=self.vars,
                    T=data.shape[0],
                    state_dim=len(self.obs_distns),
                    obs_distns=self.obs_distns,
                    dur_distns=self.dur_distns,
                    transition_distn=self.trans_distn,
                    initial_distn=self.init_state_distn,
                    trunc=self.trunc,
                    **kwargs))

# TODO hmm versions below here

# class factorial_component_hmm(pyhsmm.models.hmm):
#     means = None
#     vars = None
#     def add_factorial_sumdata(self,data,**kwargs):
#         self.states_list.append(pyhsmm.plugins.factorial.states.factorial_component_hmm_states(data,**kwargs))

# class factorial_component_hmm_possiblechangepoints(pyhsmm.models.hmm):
#     means = None
#     vars = None
#     def add_factorial_sumdata(self,data,changepoints,**kwargs):
#         self.states_list.append(pyhsmm.plugins.factorial.states.factorial_component_hmm_states_possiblechangepoints(data,changepoints,**kwargs))
