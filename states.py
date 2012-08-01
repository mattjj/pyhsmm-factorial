from __future__ import division
import numpy as np
na = np.newaxis
import scipy.weave

import pdb

# TODO just changed museqs and varseqs shape, hope they work out now!

import pyhsmm

######################################################
#  used by pyhsmm.plugins.factorial.factorial class  #
######################################################

class factorial_allstates(object):
    def __init__(self,component_models,data=None,T=None,keep=False,**kwargs):
        # kwargs is for changepoints, passed to
        # component_model.add_factorial_sumdata
        # keep is used only when calling models.factorial.generate()

        self.component_models = component_models

        self.states_list = []
        if data is not None:
            assert data.ndim == 1 or data.ndim == 2
            self.data = np.reshape(data,(-1,1))
            T = data.shape[0]
            for c in component_models:
                c.add_factorial_sumdata(data=data,**kwargs)
                self.states_list.append(c.states_list[-1])
                self.states_list[-1].allstates_obj = self # give a reference to self
                # the added states object will get its resample() method called, but
                # since that object doesn't do anything at the moment,
                # resample_factorial needs to be called here to intialize
                # s.stateseq
                self.states_list[-1].generate_states()
        else:
            # generating from the prior
            allobs = np.zeros((T,len(component_models)))
            allstates = np.zeros((T,len(component_models)),dtype=np.int32)
            assert T is not None, 'need to pass in either T (when generating) or data'
            for idx,c in enumerate(component_models):
                allobs[:,idx],allstates[:,idx] = c.generate(T=T,keep=keep,**kwargs)
                self.states_list.append(c.states_list[-1])
                self.states_list[-1].allstates_obj = self # give a reference to self
            self.sumobs = allobs.sum(1)
            self.data = np.reshape(self.sumobs,(-1,1))
            self.allstates = allstates
            self.allobs = allobs

        # track museqs and varseqs so they don't have to be rebuilt too much
        # NOTE: component_models must have scalar gaussian observation
        # distributions! this part is one of those that requires it!
        self.museqs = np.zeros((T,len(self.component_models)))
        self.varseqs = np.zeros((T,len(self.component_models)))
        for idx, (c,s) in enumerate(zip(component_models,self.states_list)):
            self.museqs[:,idx] = c.means[s.stateseq]
            self.varseqs[:,idx] = c.vars[s.stateseq]

        # build eigen codestr
        self.codestr = base_codestr % {'T':T,'K':len(component_models)}

        # just to avoid extra malloc calls... used in
        # self._get_other_mean_var_seqs
        self.summers = np.ones((len(self.component_models),len(self.component_models))) \
                - np.eye(len(self.component_models))

    def resample(self,**kwargs): # kwargs is for temp stuff
        # tell each chain to resample its statesequence, then update the
        # corresponding rows of museqs and varseqs
        # also, delete instantiated emissions
        for idx, (c,s) in enumerate(zip(self.component_models,self.states_list)):
            if 'data' in s.__dict__:
                del s.data
            s.resample_factorial(**kwargs)
            self.museqs[:,idx] = c.means[s.stateseq]
            self.varseqs[:,idx] = c.vars[s.stateseq]

    def instantiate_component_emissions(self,temp_noise=0.):
        # get the emissions
        emissions = self._sample_component_emissions(temp_noise).T.copy() # emissions is now ncomponents x T

        # add the emissions to each comopnent states list
        for e, s in zip(emissions,self.states_list):
            s.data = e

    # this method is called by the members of self.states_list; it's them asking
    # for a sum of part of self.museqs and self.varseqs
    def _get_other_mean_var_seqs(self,statesobj):
        statesobjindex = self.states_list.index(statesobj)
        return self.museqs.dot(self.summers[statesobjindex]), \
                self.varseqs.dot(self.summers[statesobjindex])

    def _sample_component_emissions_python(self,temp_noise=0.):
        # this algorithm is 'safe' but it computes lots of unnecessary cholesky
        # factorizations. the eigen code uses a smart custom cholesky downdate
        K,T = len(self.component_models), self.data.shape[0]
        contributions = np.zeros((T,K))

        meanseq = self.museqs
        varseq = self.varseqs

        tots = varseq.sum(1)[:,na] + temp_noise
        post_meanseq = meanseq + varseq * ((self.data - meanseq.sum(1)[:,na]) / tots)

        for t in range(T):
            contributions[t] = np.dot(np.linalg.cholesky(np.diag(varseq[t]) -
                1./tots[t] * np.outer(varseq[t],varseq[t])),np.random.randn(K)) + post_meanseq[t]

        return contributions

    def _sample_component_emissions_eigen(self,temp_noise=0.):
        # NOTE: this version does a smart cholesky downdate
        K,T = len(self.component_models), self.data.shape[0]
        contributions = np.zeros((T,K))
        G = np.random.randn(T,K)

        meanseq = self.museqs
        varseq = self.varseqs

        tots = varseq.sum(1)[:,na] + temp_noise
        post_meanseq = meanseq + varseq * ((self.data - meanseq.sum(1)[:,na]) / tots)

        noise_variance = temp_noise

        scipy.weave.inline(self.codestr,['varseq','meanseq','post_meanseq','G','contributions','noise_variance'],
                headers=['<Eigen/Core>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3'])

        return contributions

    _sample_component_emissions = _sample_component_emissions_eigen


####################################################################
#  used by pyhsmm.plugins.factorial.factorial_component_* classes  #
####################################################################

# the only difference between these and standard hsmm or hmm states classes is
# that they have special resample_factorial and get_aBl methods for working with
# the case where component emissions are marginalized out. they also have a
# no-op resample method, since that method might be called by the resample
# method in an hsmm or hmm model class and assumes instantiated data
# essentially, we only want these state sequences to be resampled when a
# factorial_allstates objects tells them to be resampled

# NOTE: component_models must have scalar gaussian observation
# distributions! this code, which references the same cached means and vars as
# the models, requires it!

class factorial_component_hsmm_states(pyhsmm.internals.states.hsmm_states_python):
    def __init__(self,means,vars,**kwargs):
        self.means = means
        self.vars = vars
        super(factorial_component_hsmm_states,self).__init__(**kwargs)

    def resample(self):
        pass

    def resample_factorial(self,temp_noise=0.):
        self.temp_noise = temp_noise
        self.data = object() # a little shady, this is a placeholder to trick parent resample()
        super(factorial_component_hsmm_states,self).resample()
        del self.data
        del self.temp_noise

    # NOTE: component_models must have scalar gaussian observation
    # distributions! this code requires it!
    def get_aBl(self,fakedata):
        mymeans = self.means # 1D, length state_dim
        myvars = self.vars # 1D, length state_dim

        sumothermeansseq, sumothervarsseq = self.allstates_obj._get_other_mean_var_seqs(self)
        sumothermeansseq.shape = (-1,1) # 2D, T x 1
        sumothervarsseq.shape = (-1,1) # 2D, T x 1

        sigmasq = myvars + sumothervarsseq + self.temp_noise

        return -0.5*(self.allstates_obj.data - sumothermeansseq - mymeans)**2/sigmasq \
                - np.log(np.sqrt(2*np.pi*sigmasq))

class factorial_component_hsmm_states_possiblechangepoints(
        factorial_component_hsmm_states,
        pyhsmm.internals.states.hsmm_states_possiblechangepoints
        ):
    # NOTE: this multiple-inheritance forms the diamond patern:
    #                   hsmm_states_python
    #                      /            \
    #                     /              \
    # factorial_component_hsmm_states    hsmm_states_possiblechangepoints
    #                      \             /
    #                       \           /
    #                        this class
    #
    # you can check by importing and checking thisclassname.__mro__, which will
    # list the two middle levels before the top level
    # it will make sure factorial_component_hsmm_states's get_aBl is called
    # and hsmm_states_possiblechangepoints's messages_backwards is called
    # still need to explicitly ask for hsmm_states_posschange's init method

    def __init__(self,means,vars,**kwargs):
        assert 'changepoints' in kwargs, 'must pass in a changepoints argument!'
        self.means = means
        self.vars = vars
        pyhsmm.internals.states.hsmm_states_possiblechangepoints.__init__(self,**kwargs)

    def get_aBL(self,data):
        aBBl = np.zeros((len(self.blocks),self.state_dim))
        aBl = super(factorial_component_hsmm_states_possiblechangepoints,self).get_aBl(data)
        for blockt, (start,end) in enumerate(self.blocks):
            aBBl[blockt] = aBl[start:end].sum(0)
        self.aBBl = aBBl
        return None


# TODO hmm versions below here

# class factorial_component_hmm_states(pyhsmm.internals.states.hmm_states_python):
#     def resample(self):
#         pass

#     def resample_factorial(self,temp_noise=0.):
#         self.temp_noise = temp_noise
#         super(factorial_component_hsmm_states,self).resample()
#         del self.temp_noise

#     def get_abl(self,data):
#         raise notimplementederror

# class factorial_component_hmm_states_possiblechangepoints(
#         factorial_component_hmm_states,
#         pyhsmm.internals.states.hmm_states_possiblechangepoints
#         ):
#     def resample(self):
#         pass

#     def resample_factorial(self,temp_noise=0.):
#         self.temp_noise = temp_noise
#         super(factorial_component_hsmm_states,self).resample()
#         del self.temp_noise

#     def get_aBl(self,data):
#         raise NotImplementedError


########################
#  global eigen stuff  #
########################

base_codestr = '''
    using namespace Eigen;

    // inputs

    double enoise_variance = noise_variance;
    Map<MatrixXd> evarseq(varseq,%(K)d,%(T)d);
    Map<MatrixXd> emeanseq(meanseq,%(K)d,%(T)d);
    // for now, calculate these in numpy, too
    Map<MatrixXd> epost_meanseq(post_meanseq,%(K)d,%(T)d);
    Map<MatrixXd> eG(G,%(K)d,%(T)d);

    // outputs

    Map<MatrixXd> econtributions(contributions,%(K)d,%(T)d);

    // local vars

    MatrixXd updated_chol(%(K)d,%(K)d);
    MatrixXd X(%(K)d,%(K)d);
    VectorXd sumsq(%(K)d);
    VectorXd ev(%(K)d), el(%(K)d);

    for (int t=0; t < %(T)d; t++) {

        sumsq.setZero();
        X.setZero();

        el = evarseq.col(t).cwiseSqrt();
        ev = evarseq.col(t) / (sqrt(evarseq.col(t).sum() + enoise_variance)); // TODO make noise_variance work passed in

        // compute update into X
        for (int j = 0; j < %(K)d; j++) {
            for (int i = j; i < %(K)d; i++) {
                if (i == j) {
                    X(i,i) = el(i) - sqrt(el(i)*el(i) - sumsq(i) - ev(i)*ev(i));
                } else {
                    X(i,j) = (-1.*ev(i)*ev(j) - X.row(i).head(j).dot(X.row(j).head(j))) / (X(j,j) - el(j));
                    sumsq(i) += X(i,j) * X(i,j);
                }
            }
        }

        // write into updated_chol
        updated_chol = el.asDiagonal();
        updated_chol -= X;

        // generate a sample
        econtributions.col(t) = updated_chol * eG.col(t) + epost_meanseq.col(t);
    }
'''

