import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dstr

from tools import Dataset_to_XY, DataXY
from argparse import ArgumentParser

import pickle



class Encoder(nn.Module):
    def __init__(self, zdim, layer_widths):
        super(Encoder, self).__init__()
        # construct the network
        self.zdim = zdim
        self.net = nn.Sequential()

        in_width = 28*28
        for out_width in layer_widths:
            # Loop is no-op for empty layer_widths
            self.net.append(nn.Linear(in_width, out_width))
            self.net.append(nn.ReLU(True))
            in_width = out_width
        self.net.append(nn.Linear(in_width, 2* self.zdim))
 
    def forward(self, x):
        scores = self.net(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        sigma = torch.exp(sigma)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, zdim, layer_widths):
        super(Decoder, self).__init__()
        # construct the network
        self.zdim = zdim
        self.net = nn.Sequential()
        
        in_width = self.zdim
        for out_width in layer_widths:
            # Loop is no-op for empty layer_widths
            self.net.append(nn.Linear(in_width, out_width))
            self.net.append(nn.ReLU(True))
            in_width = out_width
        self.net.append(nn.Linear(in_width, 784))

        # if you learn the sigma of the decoder 
        self.logsigma = torch.nn.Parameter(torch.ones(1))
 
    def forward(self, x):
        mu = self.net(x) 
        return mu

class VAE(nn.Module):
    def __init__(self, zdim, stepsize, layer_widths):
        if layer_widths:
            assert layer_widths[-1] > zdim
        self.layer_widths = layer_widths
        super(VAE, self).__init__()
        self.decoder = Decoder(zdim, reversed(layer_widths))
        self.encoder = Encoder(zdim, layer_widths)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=stepsize)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
    def learn_step(self, x):
        self.optimizer.zero_grad()
        # apply encoder q(z|x)
        z_mu, z_sigma = self.encoder(x)
        qz = dstr.Normal(z_mu, z_sigma)
        # sample with re-parametrization
        z = qz.rsample()
        # apply decoder p(x|z)
        x_mu = self.decoder(z)
        px = dstr.Normal(x_mu, torch.exp(self.decoder.logsigma))
        # prior p(z)
        pz = dstr.Normal(torch.zeros_like(z_mu), torch.ones_like(z_mu))
        # learn
        logx = px.log_prob(x)
        logx = logx.mean(0).sum()
        # KL-Div term
        kl_div = dstr.kl_divergence(qz, pz).mean(0).sum()
        nelbo = kl_div - logx
        nelbo.backward()
        self.optimizer.step()
 
        return nelbo.detach(), kl_div.detach()

# Shared training function used for all cases. When training the net for classification,
# train_for_classification is True and the result of net(data) is passed to loss function.
# When trained for embeddings, result of net.features is passed to the loss function.
# Accuracy/loss metrics are still recorded in order to plot them and conclude that
# the accuracy of the model trained for embedding is abysmal because it is not what the model is trying to achieve
def train(vae, train_loader, num_epochs):

    list_nelbo = []
    list_kl_div = []

    for e in range(num_epochs):
        vae.train()
        elbo_sum = 0
        kl_div_sum = 0

        for data, _ in train_loader:
            nelbo, kl_div = vae.learn_step(data)
            
            elbo_sum += nelbo
            kl_div_sum += kl_div
        
        list_nelbo.append(elbo_sum / len(train_loader))
        list_kl_div.append(kl_div_sum / len(train_loader))
        print(f'Epoch {e}: ELBO {list_nelbo[-1]:.3e}, KL div {list_kl_div[-1]:.3e}')

    return list_nelbo, list_kl_div
        