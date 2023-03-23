from SignalDetection import SignalDetection
import numpy as np
import scipy.stats

class Metropolis:

    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.currentState = initialState
        self.stepSize = 1.0
        self.samples = [initialState]
        self.acceptanceRate = None

    def __accept(self, proposal):
        """
        A private method that checks whether to accept or reject the proposed value proposal
        based on the acceptance probability calculated from the current state and the proposed
        state. It returns True if the proposal is accepted and False otherwise.
        """
        logRatio = self.logTarget(proposal) - self.logTarget(self.currentState)
        if np.log(np.random.uniform()) < logRatio:
            self.currentState = proposal
            self.samples.append(proposal)
            return True
        else:
            self.samples.append(self.currentState)
            return False


    def adapt(self, blockLengths):
        """
        Performs the adaptation phase of the Metropolis algorithm. It tries to adjust the step
        size sigma  to achieve a target acceptance rate of approximately 0.4. It does so by running
        a few blocks of iterations (the number of blocks and their length defined by blockLengths)
        and adjusting the sigma value using a formula based on the acceptance rate.
        """
        acceptanceRate = 0
        for blockLength in blockLengths:
            accepts = 0
            for i in range(blockLength):
                proposal = np.random.normal(loc = self.currentState, scale = self.stepSize)
                accepts += self.__accept(proposal)
            acceptanceRate = accepts / blockLength
            if acceptanceRate < 0.1:
                self.stepSize /= 2
            elif acceptanceRate > 0.6:
                self.stepSize *= 2
        self.acceptanceRate = acceptanceRate
        return self


    def sample(self, nSamples):
        """
        Generates n samples from the target distribution using the Metropolis algorithm. It
        starts from the current state and proposes a new state using a normal distribution with the
        current state as the mean and sigma_K as the standard deviation. If the proposed state is
        accepted, it becomes the new state.
        """
        for i in range(nSamples):
            proposal = np.random.normal(loc = self.currentState, scale = self.stepSize)
            self.__accept(proposal)
        return self


    def summary(self):
        """
        Returns a dictionary or structure containing the mean and 95% credible interval of the
        generated samples.
        """
        return {'mean': np.mean(self.samples), 'c025': np.percentile(self.samples, 2.5),
                'c975': np.percentile(self.samples, 97.5)}

