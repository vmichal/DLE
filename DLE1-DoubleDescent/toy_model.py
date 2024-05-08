# %% 
import scipy.stats
from typing import Tuple

import math
import numpy as np

import pickle

import os
import sys

""" matplotlib drawing to a pdf setup """
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#!%matplotlib inline


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """ For a more elaborate solution take a look at the EasyDict package https://pypi.org/project/easydict/ """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # these are needed for deepcopy / pickle
    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)


def save_pdf(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)


figsize = (6.0, 6.0 * 3 / 4)


def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.DEFAULT_PROTOCOL)


def load_object(filename):
    res = pickle.load(open(filename, "rb"))
    return res


""" Simulation Model, similar to the one in the book 'The Elements of Statistical Learning' """


class G2Model:
    def __init__(self):
        self.K = 3  # mixture components
        self.priors = [0.5, 0.5]
        self.cls = [dotdict(), dotdict()]
        self.cls[0].mus = np.array([[-1, -1], [-1, 0], [0, 0]])
        self.cls[1].mus = np.array([[0, 1], [0, -1], [1, 0]])
        self.Sigma = np.eye(2) * 1 / 20
        self.name = 'GTmodel'

    def samples_from_class(self, c, sample_size):
        """
        :return: x -- [sample_size x d] -- samples from class c
        """
        # draw components
        kk = np.random.randint(0, self.K, size=sample_size)
        x = np.empty((sample_size, 2))
        for k in range(self.K):
            mask = kk == k
            # draw from Gaussian of component k
            x[mask, :] = np.random.multivariate_normal(self.cls[c].mus[k, :], self.Sigma, size=mask.sum())
        return x

    def generate_sample(self, sample_size):
        """
        function to draw labeled samples from the model
        :param sample_size: how many in total
        :return: (x,y) -- features, class, x: [sample_size x d],  y : [sample_size]
        """
        assert (sample_size % 2 == 0), 'use even sample size to obtain equal number of pints for each class'
        y = (np.arange(sample_size) >= sample_size // 2) * 1  # class labels
        x = np.zeros((sample_size, 2))
        for c in [0, 1]:
            # draw from Gaussian Mixture of class c
            x[y == c, :] = self.samples_from_class(c, sample_size // 2)
        y = 2 * y - 1  # remap to -1, 1
        return x, y

    def score_class(self, c, x: np.array) -> np.array:
        """
            Compute log probability for data x and class c (sometimes also called score for the multinomial model)
            x: [N x d]
            return score : [N]
        """
        N = x.shape[0]
        S = np.empty((N, self.K))
        # compute log density of each mixture component
        for k in range(self.K):
            S[:, k] = scipy.stats.multivariate_normal(self.cls[c].mus[k, :], self.Sigma).logpdf(x)
        # compute log density of the mixture
        score = scipy.special.logsumexp(S, axis=1) + math.log(1.0 / self.K) + math.log(self.priors[c])
        return score

    def score(self, x: np.array) -> np.array:
        """ Return log odds (logits) of predictive probability p(y|x) of the network
	"""
        scores = [self.score_class(c, x) for c in range(2)]
        score = scores[1] - scores[0]
        return score

    def classify(self, x: np.array) -> np.array:
        """
        Make class prediction for a given input
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] class -1 or 1 per input point
        """
        return np.sign(self.score(x))

    def test_error(self, predictor, test_data):
        """
        evaluate test error of a predictor
        :param predictor: object with predictor.classify(x:np.array) -> np.array
        :param test_data: tuple (x,y) of the test points
        :return: error rate
        """
        x, y = test_data
        y1 = predictor.classify(x)
        err_rate = (y1 != y).sum() / x.shape[0]
        return err_rate
    
    def loss(self, predictor, test_data):
        """
        evaluate loss of a predictor
        :param predictor: object with predictor.classify(x:np.array) -> np.array
        :param test_data: tuple (x,y) of the test points
        :return: total loss
        """
        x, y = test_data
        s = predictor.score(x)
        loss = (np.square(s - y)).sum() / x.shape[0]
        return loss


    def plot_boundary(self, train_data, predictor=None):
        """
        Visualizes the GT model, training points and the decisison boundary of a given predictor
        :param train_data: tuple (x,y)
        predictor: object with
            predictor.score(x:np.array) -> np.array
            predictor.name -- str to appear in the figure
        """
        x, y = train_data
        #
        plt.figure(2, figsize=figsize)
        plt.rc('lines', linewidth=1)
        # plot points
        mask0 = y == -1
        mask1 = y == 1
        plt.plot(x[mask0, 0], x[mask0, 1], 'bo', ms=3)
        plt.plot(x[mask1, 0], x[mask1, 1], 'rd', ms=3)
        # plot classifier boundary
        ngrid = [200, 200]
        xx = [np.linspace(x[:, i].min() - 0.5, x[:, i].max() + 0.5, ngrid[i]) for i in range(2)]
        # xx = [np.linspace(-3, 4, ngrid[i]) for i in range(2)]
        # xx = [np.linspace(-2, 4, ngrid[0]), np.linspace(-3, 3, ngrid[0])]
        Xi, Yi = np.meshgrid(xx[0], xx[1], indexing='ij')  # 200 x 200 matrices
        X = np.stack([Xi.flatten(), Yi.flatten()], axis=1)  # 200*200 x 2
        # Plot the GT scores contour
        score = self.score(X).reshape(ngrid)
        m1 = np.linspace(0, score.max(), 4)
        m2 = np.linspace(score.min(), 0, 4)
        # plt.contour(Xi, Yi, score, np.sort(np.concatenate((m1[1:], m2[0:-1]))), linewidths=0.5) # intermediate contour lines of the score
        CS = plt.contour(Xi, Yi, score, [0], colors='r', linestyles='dashed')
        # CS.collections[0].set_label('Bayes optimal')
        #l = dict()
        h,_ = CS.legend_elements()
        H = [h[0]]
        L = ["Bayes optimal"]
        #l[h[0]] = 'GT boundary'
        # CS.collections[0].set_label('GT boundary')
        # Plot Predictor's decision boundary
        if predictor is not None:
            score = predictor.score(X).reshape(ngrid)
            CS = plt.contour(Xi, Yi, score, [0], colors='k', linewidths=1)
            h,_ = CS.legend_elements()
            H += [h[0]]
            L += ["Predictor"]
            # CS.collections[0].set_label('Predictor boundary')
            #h,_ = CS.legend_elements()
            #l[h[0]] = 'GT boundary'
            y1 = predictor.classify(x)
            err = y1 != y
            h = plt.plot(x[err, 0], x[err, 1], 'ko', ms=6, fillstyle='none', label='errors', markeredgewidth=0.5)
            #l[h[0]] = 'Errors'
            H += [h[0]]
            L += ["errors"]
        plt.xlabel("x0")
        plt.ylabel("x1")
        # plt.text(0.3, 1.0, name, ha='center', va='top', transform=plt.gca().transAxes)
        # plt.legend(loc=0)
        #plt.legend(l.keys(), l.values(), loc=0)
        plt.legend(H, L, loc=0)

# %%

class Lifting:
    def __init__(self, input_size, hidden_size):
        self.W1 = (np.random.rand(hidden_size, input_size) * 2 - 1)
        self.W1 /= np.linalg.norm(self.W1, axis=1).reshape(hidden_size, 1)
        self.b1 = (np.random.rand(hidden_size) * 2 - 1) * 2

    def __call__(self, x):
        """
        input: x [N x 2] data points
        output: [N x hidden_size]
        """
        return np.tanh((x @ self.W1.T + self.b1[np.newaxis, :])*5)


class MyNet:
    """ Template example for the network """

    def __init__(self, input_size, hidden_size):
        # name is needed for printing
        self.name = f'test-net-{hidden_size}'
        self.D = hidden_size
        self.lifting = Lifting(input_size, self.D)
        self.w = np.zeros(self.D)
        self.b = 0

    def score(self, x: np.array) -> np.array:
        """
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: s: np.array [N] predicted scores of class 1 for all points
        """
        #s = self.lifting(x).sum(axis=-1)

        phi = self.lifting(x)
        
        s = phi @ self.w + self.b

        return s

    def classify(self, x: np.array) -> np.array:
        """
        Make class prediction for the given gata
        *
        :param x: np.array [N x d], N number of points, d dimensionality of the input features
        :return: y: np.array [N] class 0 or 1 per input point
        """
        return np.sign(self.score(x))

    def train(self, train_data, lambda_regularization=0):
        """
        Train the model on the provided data
        *
        :param train_data: tuple (x,y) of trianing data arrays: x[N x 2], y[N]
        """
        x, y = train_data
        N = x.shape[0] # Number of points
        phi = np.hstack([self.lifting(x), np.ones((N,1))]) # Data matrix
        assert phi.shape == (N, self.D+1)
        # Learn params by linear regression
        theta = np.linalg.inv(phi.T @ phi + lambda_regularization * np.eye(self.D+1)) @ phi.T @ y
        self.w = theta[0:self.D]
        self.b = theta[self.D]

# %%

def observe_boundary(D, N):
    np.random.seed(seed=1)
    
    plt.clf()
    G = G2Model()
    train_data = G.generate_sample(N)
    test_data = G.generate_sample(50000)
    G.plot_boundary(train_data)
    # task 
    net = MyNet(2, D)
    net.train(train_data, 1e-6)
    G.plot_boundary(train_data, net)
    err = G.test_error(net, test_data)
    print(f'{D = }, {N = }: Test error {err*100:.3f}%')
    #plt.title(f'{D = } {N = }')
    plt.draw()
    save_pdf(f'D_{D}_N_{N}.png')


def run_fix_D_vary_N(D):
    repetitions = 250
    G = G2Model()
    tests_N = np.arange(2, 101, 2)
    test_num = len(tests_N)
    test_errors = np.zeros((test_num, ))
    test_losses = np.zeros((test_num, ))
    test_data = G.generate_sample(50000)
    for test_id in range(test_num):
        N = tests_N[test_id]
        print(f'starting w/ {N = }')
        for _ in range(repetitions):
            train_data = G.generate_sample(N) 
            net = MyNet(2, D)
            net.train(train_data, 1e-6)
            test_errors[test_id] += G.test_error(net, test_data) * 100
            test_losses[test_id] += G.loss(net, test_data)
    test_errors /= repetitions
    test_losses /= repetitions
    plt.figure(5, figsize=figsize)
    plt.plot(tests_N, test_errors)
    plt.grid(True)
    plt.xlabel("Training set size N")
    plt.ylabel("Test error [%]")
    plt.draw()
    save_pdf(f'fix_D_{D}_vary_N_test_error.png')

    plt.figure(7, figsize=figsize)
    plt.plot(tests_N, test_losses)
    plt.grid(True)
    plt.xlabel("Training set size N")
    plt.ylabel("Test loss [-]")
    plt.yscale('log')
    plt.draw()
    save_pdf(f'fix_D_{D}_vary_N_test_loss.png')

def run_fix_N_vary_D(N):
    repetitions = 250
    G = G2Model()
    tests_D = [1] + [10*x for x in range(1,21)]
    test_num = len(tests_D)
    test_errors = np.zeros((test_num, ))
    train_errors = np.zeros((test_num, ))
    test_losses = np.zeros((test_num, ))
    train_losses = np.zeros((test_num, ))
    test_data = G.generate_sample(50000)
    for test_id in range(test_num):
        D = tests_D[test_id]
        print(f'starting w/ {D = }')
        for _ in range(repetitions):
            train_data = G.generate_sample(40) 
            net = MyNet(2, D)
            net.train(train_data, 1e-6)
            train_errors[test_id] += G.test_error(net, train_data) * 100
            test_errors[test_id] += G.test_error(net, test_data) * 100
            train_losses[test_id] += G.loss(net, train_data)
            test_losses[test_id] += G.loss(net, test_data)
            
    test_errors /= repetitions
    train_errors /= repetitions
    test_losses /= repetitions
    train_losses /= repetitions
    plt.figure(6, figsize=figsize)
    plt.plot(tests_D, train_errors, label='Train error')
    plt.plot(tests_D, test_errors, label='Test error')
    plt.grid(True)
    plt.xlabel("Hidden layer size D")
    plt.ylabel("Error [%]")
    plt.legend()
    plt.yscale('log')
    plt.draw()
    save_pdf(f'fix_N_{N}_vary_D_error.png')

    plt.figure(8, figsize=figsize)
    plt.plot(tests_D, train_losses, label='Train loss')
    plt.plot(tests_D, test_losses, label='Test loss')
    plt.grid(True)
    plt.xlabel("Hidden layer size D")
    plt.ylabel("Loss []")
    plt.yscale('log')
    plt.legend()
    plt.draw()
    save_pdf(f'fix_N_{N}_vary_D_loss.png')
    plt.show()


if __name__ == "__main__":

    import matplotlib, sys
    
    #np.random.seed(seed=1)

    # Task 1 observe decision boundary 
    #D = int(sys.argv[1])
    #N = int(sys.argv[2])
    for D in [10, 20, 30, 40, 50, 75, 100, 150, 250, 500, 1000]:
        observe_boundary(D, 40)

    np.random.seed(seed=105365)
    # Fix D = 40 and vary training set size
    run_fix_D_vary_N(40)

    # Fix N = 40 and vary hidden layer size
    run_fix_N_vary_D(40)
