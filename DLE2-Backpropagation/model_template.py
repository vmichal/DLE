import torch
import torch.nn as nn
import copy
import numpy as np

class FFModel():
    def __init__(self, hdim, device, dtype):
        """ hdim -- hidden layer size """
        # hidden layer
        self.w1 = torch.empty([2, hdim], dtype=dtype, device=device).uniform_(-1.0, 1.0) # [2 hdim]
        self.w1.requires_grad_()
        self.b1 = torch.empty([1, hdim], dtype=dtype, device=device).uniform_(-1.0, 1.0) # [1 hdim]
        self.b1.requires_grad_()
        self.w = torch.empty([1, hdim], dtype=dtype, device=device).uniform_(-1.0, 1.0) # [1 hdim]
        self.w.requires_grad_()
        self.b = torch.empty([1, 1], dtype=dtype, device=device).uniform_(-1.0, 1.0) # [1 1]
        self.b.requires_grad_()
        self.parameters = [self.w1, self.b1, self.w, self.b]
    
    def score(self, x):
        """ Compute scores for inputs x 
        x : [N x d] 
        output:
        s : [N] - scores
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.w1)
        Phi = torch.tanh(x @ self.w1 + self.b1)
        s = Phi @ self.w.T + self.b
        N = x.shape[0]
        s = s.reshape((N, ))
        return s

    def classify(self, x):
        scores = self.score(x)
        return scores.sign()

    def mean_loss(self, x, y):
        """               
        Compute the mean_loss of the training data = average negative log likelihood
        *
        :param train_data: tuple(x,y)
        x [N x d]
        y [N], encoded +-1 classes
        :return: mean negative log likelihood
        """
        N = x.shape[0]
        s = self.score(x)
        arg = s * y
        return -1.0 / N * torch.nn.functional.logsigmoid(arg).sum()

    def mean_accuracy(self, x, targets):
        y = self.classify(x)
        acc = (y == targets).float().mean()
        return acc
    
    def errors(self, x, targets):
        return self.classify(x) != targets
    
    def empirical_test_error(self, x, targets):
        return self.errors(x, targets).float().mean()

    def zero_grad(self):
        # set .grad to None (or zeroes) for all parameters
        for p in self.parameters:
            p.grad = None
  
    def check_gradient(self, x, targ, pname, epsilon):

        w = getattr(self, pname)
        u = torch.empty(w.shape, dtype=w.dtype, device=w.device).uniform_(-1.0, 1.0) # [2 hdim]
        u = torch.nn.functional.normalize(u, 2)

        setattr(self, pname, w + epsilon * u)
        L1 = self.mean_loss(x, targ)
        setattr(self, pname, w - epsilon * u)
        L2 = self.mean_loss(x, targ)
        setattr(self, pname, w)
        
        numeric_derivative = (L1 - L2) / (2 * epsilon)

        l = self.mean_loss(x, targ)
        self.zero_grad()
        l.backward()

        analytic_derivative = (w.grad * u).sum()

        error = numeric_derivative - analytic_derivative
        
        # TODO compute gradients
        print(f"# Grad error in {pname}: {torch.abs(error):.4}")
