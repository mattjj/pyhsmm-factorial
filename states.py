from __future__ import division
import numpy as np
na = np.newaxis
import scipy.weave

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

        # build eigen codestr
        self.codestr = base_codestr % {'T':data.shape[0],'K':len(component_models)}

    def resample(self,**kwargs): # kwargs is for temp stuff
        # tell each chain to resample its statesequence, then update the
        # corresponding rows of museqs and varseqs
        # also, delete instantiated emissions
        for idx, (c,s) in enumerate(zip(self.component_models,self.states_list)):
            if 'data' in s.__dict__:
                del s.data
            s.resample_factorial(**kwargs)
            self.museqs[idx] = c.means[s.stateseq]
            self.varseqs[idx] = c.vars[s.stateseq]

    def instantiate_component_emissions(self,temp_noise=0.):
        # get the emissions
        emissions = self._sample_component_emissions(temp_noise)

        # add the emissions to each comopnent states list
        for e, s in zip(emissions,self.states_list):
            s.data = e

    def _sample_component_emissions_python(self,temp_noise=0.):
        K,T = len(self.component_models), self.data.shape[0]
        contributions = np.zeros((T,K))

        meanseq = self.meanseqs
        varseq = self.varseqs

        tots = varseq.sum(1)[:,na] + temp_noise
        post_meanseq = meanseq + varseq * ((self.data - meanseq.sum(1)[:,na]) / tots)

        for t in range(T):
            contributions[t] = np.dot(np.linalg.cholesky(np.diag(varseq[t]) - 1./tots[t] * np.outer(varseq[t],varseq[t])),np.random.randn(K)) + post_meanseq[t]

        return contributions

    def _sample_component_emissions(self,temp_noise=0.):
        K,T = len(self.component_models), self.data.shape[0]
        contributions = np.zeros((T,K))
        G = np.random.randn(T,K)

        meanseq = self.meanseqs
        varseq = self.varseqs

        tots = varseq.sum(1)[:,na] + temp_noise
        post_meanseq = meanseq + varseq * ((self.data - meanseq.sum(1)[:,na]) / tots)

        scipy.weave.inline(self.codestr,['varseq','meanseq','post_meanseq','G','contributions','temp_noise'],headers=['<Eigen/Core>'],include_dirs=['/usr/local/include/eigen3'],extra_compile_args=['-O3'])

        return contributions


####################################################################
#  used by pyhsmm.plugins.factorial.factorial_component_* classes  #
####################################################################

class factorial_component_hsmm_states(pyhsmm.internals.states.hsmm_states_python):
    def resample(self):
        # left as a no-op so that the states aren't resampled when an hsmm model
        # containing a reference to these states are resampled
        pass

    def resample_factorial(self,temp_noise=0.):
        raise NotImplementedError

class factorial_component_hsmm_states_possiblechangepoints(
        pyhsmm.internals.states.hsmm_states_possiblechangepoints):
    def resample(self):
        pass

    def resample_factorial(self,temp_noise=0.):
        raise NotImplementedError



class factorial_component_hmm_states(pyhsmm.internals.states.hmm_states_python):
    def resample(self):
        pass

    def resample_factorial(self):
        raise NotImplementedError

class factorial_component_hmm_states_possiblechangepoints(
        pyhsmm.internals.states.hmm_states_possiblechangepoints):
    def resample(self):
        pass

    def resample_factorial(self,temp_noise=0.):
        raise NotImplementedError


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

