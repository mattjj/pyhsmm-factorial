from __future__ import division

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

    def generate(self,T,keep=True): # TODO
        # this will be good for synthetic data testing
        raise NotImplementedError

    def plot(self,color=None): # TODO
        # this is ALWAYS useful
        raise NotImplementedError


######################################
#  classes for the component models  #
######################################

class factorial_component_hsmm(pyhsmm.models.hsmm):
    def add_factorial_sumdata(self,data,**kwargs):
        self.states_list.append(pyhsmm.plugins.factorial.states.factorial_component_hsmm_states(data,**kwargs))

class factorial_component_hsmm_possiblechangepoints(pyhsmm.models.hsmm):
    def add_factorial_sumdata(self,data,changepoints,**kwargs):
        self.states_list.append(pyhsmm.plugins.factorial.states.factorial_component_hsmm_states_possiblechangepoints(data,changepoints,**kwargs))


class factorial_component_hmm(pyhsmm.models.hmm):
    def add_factorial_sumdata(self,data,**kwargs):
        self.states_list.append(pyhsmm.plugins.factorial.states.factorial_component_hmm_states(data,**kwargs))

class factorial_component_hmm_possiblechangepoints(pyhsmm.models.hmm):
    def add_factorial_sumdata(self,data,changepoints,**kwargs):
        self.states_list.append(pyhsmm.plugins.factorial.states.factorial_component_hmm_states_possiblechangepoints(data,changepoints,**kwargs))
