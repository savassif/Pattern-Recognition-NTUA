from pomegranate import *
import numpy as np 

class HMM_GMM():
    def __init__(self,n_states=2,n_mixtures=2,max_iter=5):
        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.max_iter = max_iter
        self.model = None

    def fit(self,digit_instances):
        
        # X contains data from a single digit (can be a numpy array)
        X=np.array(digit_instances[0])
        for i,something in enumerate(digit_instances):
            if i == 0 :
                continue
            else :
                X = np.concatenate((X,something),axis=0)
        X=X.astype('float64')
        n_states = self.n_states # the number of HMM states
        n_mixtures = self.n_mixtures # the number of Gaussians
        if n_mixtures == 1: 
            gmm = False
        else:
            gmm = True # whether to use GMM or plain Gaussian
        dists = [] # list of probability distributions for the HMM states
        for i in range(n_states):
            if gmm:
                a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components = n_mixtures, X=X)
            else:
                a = MultivariateGaussianDistribution.from_samples(X)
            dists.append(a)

    # Possible approach to initialization of trans_mat, since we have no prior knowledge

    #   Create a n_states x n_states array with arbitrary assigned values between 0 , 1 

        trans_mat = np.array(np.random.rand(n_states,n_states))

    #   Check that the values of cell [i,j] with j<i or j>i+2 are 0.
    #   Also check that the sum of transition probabilities for each state equals to 1.

        for i in range(trans_mat.shape[0]):
            sum_of_probs = 0
            for j in range (trans_mat.shape[1]):
                if j<i or j > i+1 :
                    trans_mat[i][j] = 0
                sum_of_probs += trans_mat[i][j]
                if i == trans_mat.shape[0]-1 :
                    trans_mat[i][i] =1.0
                else :
                    if sum_of_probs < 1 :
                        if j == trans_mat.shape[1] -1 :
                            trans_mat[i][i+1] += 1-sum_of_probs
                    else :
    #                  Ignore current probability 
                        sum_of_probs -= trans_mat[i][j]
                        if not j == trans_mat.shape[1]-1 :
    #                      If you haven't reached the end of transistion matrix then select randomly a value between
    #                      0 and 1-sum_of_probs
                            trans_mat[i][j] = np.random.uniform(0,1-sum_of_probs)
                        else :
    #                      If you've reached end of transistion matrix then just set transition probability to what's left
    #                      AKA 1-sum_of_probs
                            trans_mat[i][j] = 1- sum_of_probs
                        sum_of_probs+= trans_mat[i][j]

        trans_mat = trans_mat.tolist() # your transition matrix
        starts = [0]*n_states # your starting probability matrix
        starts[0] = 1.0

        ends = [0]*n_states # your ending probability matrix
        ends[-1] = 1.0 # Based on pomegrante documentation

        data = list(map(lambda x:x.tolist(),digit_instances)) # your data: must be a Python list that contains: 2D lists with the sequences (so its dimension would be num_sequences x seq_length x feature_dimension)
                # But be careful, it is not a numpy array, it is a Python list (so each sequence can have different length)

        # Define the GMM-HMM
        self.model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])
    #     model.bake()
        # Fit the model
        self.model.fit(data, max_iterations=self.max_iter)
        return self.model 

    # Method used for calculation of the log probability.
    def viterbi(self,sample):
         
        return self.model.viterbi(sample)
