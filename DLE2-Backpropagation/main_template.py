#%%
import torch
import torch.nn as nn
import numpy as np
from toy_model import *
from model_template import FFModel
import math, sys, scipy, code

import matplotlib
import matplotlib.pyplot as plt

#!%load_ext autoreload
#!autoreload 2
#!%matplotlib inline


class Predictor():
    """ Wrapper class to convert tensors to numpy and interface with G2Model, e.g. to call
        G2Model.plot_boundary()    
    """

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def score(self, x):
        if isinstance(x,np.ndarray):
            x = torch.tensor(x).to(self.model.w1)
        return self.model.score(x).detach().numpy()

    def classify(self, x_i):
        scores = self.score(x_i)
        return np.sign(scores)

def save_pdf(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)

#%%
# make experiment reproducible by fixing the seed for all random generators
torch.manual_seed(1)
# general praparations
smpl_size = 200
vsmpl_size = 1000
tsmpl_size = 1000
lrate = 1.0e-1
dtype = torch.float64 # torch.float64


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# get training/validation data
# transform them to PyTorch tensors
gmx = G2Model()

def generate_data(sample_size):
    x, t = gmx.generate_sample(sample_size)
    x = torch.tensor(x, dtype=dtype).to(device)
    t = torch.tensor(t, dtype=torch.int).to(device)
    return x,t

x, t = generate_data(smpl_size)
xv, tv = generate_data(vsmpl_size)

    #%% Gradient check
model = FFModel(500, device, dtype)
#TODO VOMI
#print('# Gradient checks', flush=True)
#for epsilon in [1e-2, 1e-3, 1e-4, 1e-5]:
#    for parameter in ["w1", "b1", "w", "b"]:
#        model.check_gradient(xv, tv, parameter, epsilon)

#for hdim in [5, 10, 100, 500]: # TODO VOMI
for hdim in [500]:
    name = "model_{}".format(hdim) 

    # model
    model = FFModel(hdim, device, dtype)

    # Plot predictor at initialization
    pred = Predictor(name, model)
    plt.clf()
    gmx.plot_boundary((x.cpu().numpy(), t.cpu().numpy()), pred)
    plt.draw()
    save_pdf(f'figures/hidden-{hdim}-no-train.pdf')



    #%% Training
    niterations = 1000
    log_period = 100
    print('# Starting', flush=True)
    xv, tv = generate_data(26492)
    for count in range(niterations):
        # compute loss
        l = model.mean_loss(x, t)
        # compute gradient
        model.zero_grad()
        l.backward()
        # make a gradinet descent step
        for p in model.parameters:
            p.data -= lrate * p.grad.data
        # evaluate and print
        if (count % log_period == log_period-1) or (count == niterations-1):
            with torch.no_grad():
                test_error = model.empirical_test_error(xv, tv)
                train_error = model.empirical_test_error(x, t)
                gen_gap = test_error - train_error
            print(f'epoch: {count}  loss: {l.item():.4f} test error: {100 * test_error:.4f} %, gen gap {100 * gen_gap:.4f} %', flush=True)
        m = -1 / 2 / 1e-4 * np.log(0.01/2)
    print(f'Required {m = }')

    #%% 
    # Plot predictor
    pred = Predictor(name, model)
    plt.clf()
    gmx.plot_boundary((x.cpu().numpy(), t.cpu().numpy()), pred)
    plt.draw()
    save_pdf(f'figures/hidden-{hdim}.pdf')

#%% 
# Test
# True test risk distribution
print("Working on task 3")
niterations = 10000 # TODO go to 1e4
emp_test_error = [0] * niterations
m = 1000
for i in range(niterations):
    xv, tv = generate_data(m)
    emp_test_error[i] = model.empirical_test_error(xv, tv) * 100.0

k = 2
bin_width = 100 / m * k
bin_edges = np.arange(0,8, bin_width)




# Single test set T
xv, tv = generate_data(m)
alpha = 0.9
res = scipy.stats.bootstrap((model.errors(xv, tv).numpy(),), np.mean, confidence_level=alpha, method='BCa')
res.bootstrap_distribution *= 100.0

plt.clf()
plt.hist(emp_test_error, bins=bin_edges, label='Test error distribution', alpha=0.5, density=True)
plt.hist(res.bootstrap_distribution, bins=bin_edges, label='Bootstrap distribution', alpha=0.5, density=True)
R_T = np.mean(emp_test_error)
plt.axvline(R_T, label='True (expected) error rate')

plt.axvline(np.mean(res.bootstrap_distribution), label='Empirical error rate R_T')

plt.plot([100 * res.confidence_interval.low, res.confidence_interval.high * 100], [0.2, 0.2], label='90 % Confidence interval BCa')

alpha = 0.9
delta_l = 1
epsilon = np.sqrt(-delta_l**2 / 2 / m * np.log((1-alpha)/2))
print(f'{epsilon = }')

plt.plot([R_T - epsilon*100, R_T + epsilon*100], [0.1, 0.1], label='90 % Concentration interval')


plt.xlabel('Empirical test error [%]')
plt.xlabel('Density')
plt.legend()
plt.draw()
#plt.show()
save_pdf(f'figures/histogram.pdf')

#code.interact(local =locals())

#%% Confidence intervals and plots
# Chebyshev
# Hoeffding
# Plot




