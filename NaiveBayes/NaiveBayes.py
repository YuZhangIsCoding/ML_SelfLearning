import scipy
import numpy as np
from collections import Counter
from scipy.stats import norm

class NaiveBayes(object):
    '''X binary cases, y could be multiclasses
    '''
    def __init__(self):
        '''Initialze some variables'''
        self.priors = []
        self.condi = []
        self.counts = None
        self.label_names = None
    def get_classes(self, labels):
        '''Count each class, and store the name of each class'''
        self.counts = Counter(labels)
        self.label_names = list(self.counts.keys())
    def get_prior(self, labels):
        '''Calculate the prior probability of each class'''
        for name in self.label_names:
            self.priors.append(self.counts[name]/len(labels))
    def get_conditional(self, training, labels):
        '''Get conditional probability of features given label'''
        for label_name in self.label_names:
            pos = np.where(labels == label_name)[0]
            # here we include a prior probability.
            # for any features we did not in the training set
            # we assume it has the probability of 1/(# of classes)
            self.condi.append((np.sum(training[pos, :], axis = 0)+1)/(len(pos)+len(self.label_names)))
    def train(self, training, labels):
        '''Train dataset'''
        labels = labels.ravel()
        self.get_classes(labels)
        self.get_prior(labels)
        self.get_conditional(training, labels)
    def predict(self, testset):
        '''Predict given input dataset
        Return:
            list of classes
        '''
        if scipy.sparse.issparse(testset):
            testset = testset.todense()
        results = []
        for test in testset:
            best = None
            for i in range(len(self.label_names)):
                joint_prob = np.multiply(test, self.condi[i])+np.multiply(1-test, 1-self.condi[i])
                log_jprob = np.sum(np.log(joint_prob))+np.log(self.priors[i])
                if best is None or best < log_jprob:
                    best = log_jprob
                    index = i
            results.append(self.label_names[index])
        return results
    @staticmethod
    def accuracy(predicts, labels):
        '''
        Return:
            float, accuracy of prediction
        '''
        return sum(predicts == labels.ravel())/len(labels)

class GaussianNaiveBayes(NaiveBayes):
    def get_conditional(self, training, labels):
        '''Estimate the mean and variance for Gaussian distribution'''
        for label_name in self.label_names:
            pos = np.where(labels == label_name)[0]
            mu = np.mean(training[pos, :], axis = 0)
            # biased estimates of variance
            # use ddof = 1 if want to use unbiased variance
            std = np.std(training[pos, :], axis = 0)
            pos = np.where(std == 0)[0]
            # There are cases when the training set for some labels have no variance
            # sometimes due to outliers in the dataset.
            if len(pos):
                print('Attention: feature located at', pos, 'with label %s has variance of 0' %str(label_name))
            self.condi.append((mu, std))
    def predict(self, testset):
        '''Predict given input dataset'''
        results = []
        for test in testset:
            best = None
            for i in range(len(self.label_names)):
                log_jprob = np.log(self.priors[i])
                for j in range(len(test)):
                    condi_prob = norm.pdf(test[j], self.condi[i][0][j], self.condi[i][1][j])
                    log_jprob += np.log(condi_prob)
                if best is None or best < log_jprob:
                    best = log_jprob
                    index = i
            results.append(self.label_names[index])
        return results
