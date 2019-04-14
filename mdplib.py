from mdptoolbox.mdp import _computeDimensions, MDP,\
                           _MSG_STOP_MAX_ITER,\
                           _MSG_STOP_EPSILON_OPTIMAL_POLICY,\
                           _MSG_STOP_EPSILON_OPTIMAL_VALUE,\
                           _MSG_STOP_UNCHANGING_POLICY
import mdptoolbox.util as _util
import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

class PolicyIteration(MDP):

    """
    Modified from pymdptoolbox
    """

    def __init__(self, transitions, reward, discount, policy0=None,
                 max_iter=1000, eval_type=0):
        # Initialise a policy iteration MDP.
        #
        # Set up the MDP, but don't need to worry about epsilon values
        MDP.__init__(self, transitions, reward, discount, None, max_iter)
        # Check if the user has supplied an initial policy. If not make one.
        if policy0 is None:
            # Initialise the policy to the one which maximises the expected
            # immediate reward
            null = _np.zeros(self.S)
            self.policy, null = self._bellmanOperator(null)
            del null
        else:
            # Use the policy that the user supplied
            # Make sure it is a numpy array
            policy0 = _np.array(policy0)
            # Make sure the policy is the right size and shape
            assert policy0.shape in ((self.S, ), (self.S, 1), (1, self.S)), \
                "'policy0' must a vector with length S."
            # reshape the policy to be a vector
            policy0 = policy0.reshape(self.S)
            # The policy can only contain integers between 0 and S-1
            msg = "'policy0' must be a vector of integers between 0 and S-1."
            assert not _np.mod(policy0, 1).any(), msg
            assert (policy0 >= 0).all(), msg
            assert (policy0 < self.S).all(), msg
            self.policy = policy0
        # set the initial values to zero
        self.V = _np.zeros(self.S)
        # Do some setup depending on the evaluation type
        if eval_type in (0, "matrix"):
            self.eval_type = "matrix"
        elif eval_type in (1, "iterative"):
            self.eval_type = "iterative"
        else:
            raise ValueError("'eval_type' should be '0' for matrix evaluation "
                             "or '1' for iterative evaluation. The strings "
                             "'matrix' and 'iterative' can also be used.")

    def _computePpolicyPRpolicy(self):
        Ppolicy = _np.empty((self.S, self.S))
        Rpolicy = _np.zeros(self.S)
        for aa in range(self.A): # avoid looping over S
            # the rows that use action a.
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                #PR = self._computePR() # an apparently uneeded line, and
                # perhaps harmful in this implementation c.f.
                # mdp_computePpolicyPRpolicy.m
                Rpolicy[ind] = self.R[aa][ind]
        # self.R cannot be sparse with the code in its current condition, but
        # it should be possible in the future. Also, if R is so big that its
        # a good idea to use a sparse matrix for it, then converting PRpolicy
        # from a dense to sparse matrix doesn't seem very memory efficient
        if type(self.R) is _sp.csr_matrix:
            Rpolicy = _sp.csr_matrix(Rpolicy)
        #self.Ppolicy = Ppolicy
        #self.Rpolicy = Rpolicy
        return (Ppolicy, Rpolicy)

    def _evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
        try:
            assert V0.shape in ((self.S, ), (self.S, 1), (1, self.S)), \
                "'V0' must be a vector of length S."
            policy_V = _np.array(V0).reshape(self.S)
        except AttributeError:
            if V0 == 0:
                policy_V = _np.zeros(self.S)
            else:
                policy_V = _np.array(V0).reshape(self.S)

        policy_P, policy_R = self._computePpolicyPRpolicy()
        print(policy_R)

        if self.verbose:
            print('    Iteration\t\t    V variation')

        itr = 0
        done = False
        while not done:
            itr += 1

            Vprev = policy_V
            policy_V = policy_R + self.discount * policy_P.dot(Vprev)

            variation = _np.absolute(policy_V - Vprev).max()
            if self.verbose:
                print(('      %s\t\t      %s') % (itr, variation))

            # ensure |Vn - Vpolicy| < epsilon
            if variation < ((1 - self.discount) / self.discount) * epsilon:
                done = True
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_VALUE)
            elif itr == max_iter:
                done = True
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)

        self.V = policy_V
        return policy_R

    def _evalPolicyMatrix(self):
        Ppolicy, Rpolicy = self._computePpolicyPRpolicy()
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        self.V = _np.linalg.solve(
            (_sp.eye(self.S, self.S) - self.discount * Ppolicy), Rpolicy)
        return Rpolicy

    def run(self):
        # Run the policy iteration algorithm.
        # If verbose the print a header
        if self.verbose:
            print('  Iteration\t\tNumber of different actions')
        # Set up the while stopping condition and the current time
        done = False
        self.time = _time.time()
        
        # Setup rewards accumulator
        totalR = 0
        
        # loop until a stopping condition is reached
        self.n_different = []
        while not done:
            self.iter += 1
            # these _evalPolicy* functions will update the classes value
            # attribute
            if self.eval_type == "matrix":
                Rpolicy = self._evalPolicyMatrix()
            elif self.eval_type == "iterative":
                Rpolicy = self._evalPolicyIterative()
            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, null = self._bellmanOperator()
            del null
            # calculate in how many places does the old policy disagree with
            # the new policy
            n_different = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                print(('    %s\t\t  %s') % (self.iter, n_different))
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop
            if n_different == 0:
                done = True
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
            elif self.iter == self.max_iter:
                done = True
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
            else:
                self.policy = policy_next
            
            self.n_different.append(n_different)
            totalR += _np.sum(Rpolicy)

        # update the time to return th computation time
        self.time = _time.time() - self.time
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
        self.totalR = totalR

class QLearning(MDP):

    """
    Modified from pymdptoolbox. Add learning rate.
    """

    def __init__(self, transitions, reward, discount, n_iter=10000, interval=None,
                 learning_rate = None, explore_interval=100,
                 skip_check=False):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        if not skip_check:
            # We don't want to send this to MDP because _computePR should not
            #  be run on it, so check that it defines an MDP
            _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward
        self.totalR = 0.0

        self.discount = discount
        self.learning_rate = learning_rate

        # Initialisations
        self.Q = _np.zeros((self.S, self.A))
        self.mean_discrepancy = []

        self.policies = []
        self.iterations = []
        self.elapsedtimes = []
        self.interval = interval
        self.explore_interval = explore_interval

    def run(self):
        # Run the Q-learning algoritm.
        discrepancy = []

        self.time = _time.time()
        self.starttime = _time.time()

        # initial state choice
        s = _np.random.randint(0, self.S)
        d = self.learning_rate

        for n in range(1, self.max_iter + 1):

            # Reinitialisation of trajectories every explore_interval transitions
            if (n % self.explore_interval) == 0:
                s = _np.random.randint(0, self.S)

            # Action choice : greedy with increasing probability
            # probability 1-(1/log(n+2)) can be changed
            pn = _np.random.random()
            if pn < (1 - (1 / (n)**(1/6))):
            #if pn < (1 - (1 / _math.log(n + 2))):
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()
            else:
                a = _np.random.randint(0, self.A)

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]
            
            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Updating the value of Q
            d = (1 / _math.sqrt(n + 2)) 
            if self.learning_rate:
                d = self.learning_rate
            
            self.totalR += r
            
            futureQ = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = d*futureQ
            self.Q[s, a] = self.Q[s, a] + dQ

            # current state is updated
            s = s_new

            # Computing and saving maximal values of the Q variation
            discrepancy.append(_np.absolute(dQ))

            # Computing means all over maximal Q variations values
            #if len(discrepancy) == 100:
            #    self.mean_discrepancy.append(_np.mean(discrepancy))
            #    discrepancy = []

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

            if self.interval:
                if n % self.interval == 0:
                    self.elapsedtimes.append(_time.time()-self.starttime) 
                    self.policies.append(self.policy) 
                    self.iterations.append(n)           
                    self.mean_discrepancy.append(_np.mean(discrepancy))
                    discrepancy = []
        self.time = _time.time() - self.time

        # convert V and policy to tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())
