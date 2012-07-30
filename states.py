from __future__ import division
import numpy as np

import pyhsmm

######################################################
#  used by pyhsmm.plugins.factorial.factorial class  #
######################################################

class factorial_allstates(object):
    def __init__(self,data,component_models):
        self.data = data # sum data
        self.component_models = component_models

        self.states_list = []
        for c in component_models:
            c.add_factorial_summdata(data)
            self.states_list.append(c.states_list[-1])

        # track museqs and varseqs so they don't have to be rebuilt too much
        # NOTE: component_models must have scalar gaussian observation
        # distributions! this part is one of those that requires it!
        self.museqs = np.zeros((len(self.component_models),data.shape[0]))
        self.varseqs = np.zeros((len(self.component_models),data.shape[0]))
        for idx, (c,s) in enumerate(zip(component_models,self.states_list)):
            self.museqs[idx] = c.means[s.stateseq]
            self.varseqs[idx] = c.vars[s.stateseq]

    def resample(self,**kwargs): # TODO kwargs is for temperature stuff
        raise NotImplementedError

    def instantiate_component_emissions(self):
        raise NotImplementedError


####################################################################
#  used by pyhsmm.plugins.factorial.factorial_component_* classes  #
####################################################################

class factorial_component_hsmm_states(pyhsmm.internals.states.hsmm_states_python):
    def resample(self):
        # left as a no-op so that the states aren't resampled when an hsmm model
        # containing a reference to these states are resampled
        pass

    def resample_factorial(self):
        raise NotImplementedError

class factorial_component_hmm_states(pyhsmm.internals.states.hmm_states_python):
    def resample(self):
        # left as a no-op so that the states aren't resampled when an hsmm model
        # containing a reference to these states are resampled
        pass

    def resample_factorial(self):
        raise NotImplementedError
