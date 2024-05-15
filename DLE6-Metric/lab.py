import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tools import Dataset_to_XY, DataXY
from argparse import ArgumentParser

import pickle

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)

# Global datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# training and validation sets
train_set = Dataset_to_XY(torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform))
train_set, val_set = train_set.split(valid_size=0.1)
# test set
test_set = Dataset_to_XY(torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform))
test_set = test_set.fraction(fraction=0.1)
#
# dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
#

model_names = ['./models/' + name for name in ['net_class.pl', 'net_triplet.pl', "net_smoothAP.pl"]]


class ConvNet(nn.Sequential):
    def __init__(self, num_classes: int = 10) -> None:
        layers = []
        layers += [nn.Conv2d(1, 32, kernel_size=3)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 32, kernel_size=3)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d((2, 2))]
        layers += [nn.Flatten()]
        layers += [nn.Linear(64 * 2 * 2, num_classes)]
        super().__init__(*layers)
        self.layers = layers

    def features(self, x):
        f = nn.Sequential(*self.layers[:-1]).forward(x)
        f = nn.functional.normalize(f, p=2, dim=1)
        return f

def new_net():
    return ConvNet().to(dev)


def load_net(filename):
    net = ConvNet()
    net.to(dev)
    net.load_state_dict(torch.load(filename,map_location=dev))
    return net

def get_features(net, dataset):
    features = net.features(dataset.X)
    labels = dataset.Y
    return features, labels

def distances(f1: torch.Tensor, f2: torch.Tensor):
    """All pairwise distances between feature vectors in f1 and feature vectors in f2:
    f1: [N, d] array of N normalized feature vectors of dimension d
    f2: [M, d] array of M normalized feature vectors of dimension d
    return D [N, M] -- pairwise Euclidean distance matrix
    """
    assert (f1.dim() == 2)
    assert (f2.dim() == 2)
    assert (f1.size(1) == f2.size(1))
    # VoMi: implement
    return torch.cdist(f1, f2)


def evaluate_AP(dist: np.array, labels: np.array, query_label: int):
    """Average Precision
    dits: [N] array of distances to all documents from the query
    labels: [N] labels of all documents
    query_label: label of the query document
    return: AP -- average precision, Prec -- Precision, Rec -- Recall
    """
    N = labels.size(0)
    ii = torch.argsort(dist)
    dist = dist[ii]
    labels = labels[ii]
    rel = (labels == query_label).double()
    T = rel.sum()
    # VoMi: implement the computation
    Prec = torch.cumsum(rel, dim=0) / np.arange(1, N + 1)
    Rec = torch.cumsum(rel, dim=0) / T
    AP = torch.inner(Prec, rel) / T
    return AP, Prec, Rec


def evaluate_mAP(net, dataset: DataXY):
    """
    Compute Mean Average Precision
    net: a network with features() method 
    dataset: dataset of input images and labels
    Returns: mAP -- mean average precision, mRec -- mean recall, mPrec -- mean precision
    """
    torch.manual_seed(1)
    # use first 100 samples from this loader as queries, use all samples as possible
    # items to retrive, exclude the query from the retrieved items
    # VoMi: implement
    # will need to call evaluate_AP
    mAP = []
    mPrec = []
    mRec = []

    features, labels = get_features(net, dataset)
    dists = distances(features, features)
    query_idxs = np.random.choice(len(dataset), size=100, replace=False)
    for i, query_index in enumerate(query_idxs):
        all_but_query = np.arange(len(dataset)) != query_index
        AP, Prec, Rec = evaluate_AP(dists[query_index, all_but_query], labels[all_but_query], labels[query_index])
        mAP.append(AP)
        mPrec.append(Prec)
        mRec.append(Rec)

    mAP = torch.stack(mAP).mean()
    mPrec = torch.stack(mPrec, dim=-1).mean(dim=-1)
    mRec = torch.stack(mRec, dim=-1).mean(dim=-1)

    return mAP, mPrec, mRec

def evaluate_acc(net, loss_f, loader):
    net.eval()
    with torch.no_grad():
        acc = 0
        loss = 0
        n_data = 0
        for data, target in loader:
            data, target = data, target
            y = net(data)
            l = loss_f(y, target)  # noqa: E741
            loss += l.sum()
            acc += (torch.argmax(y, dim=1) == target).float().sum().item()
            n_data += data.size(0)
        acc /= n_data
        loss /= n_data
    return (loss, acc)

def train(net, train_loader, val_loader, loss_f, epochs, name, filename):
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []
    best_val_acc = 0
    for epoch in range(epochs):
        print(f"{epoch = }")
        train_acc = 0
        train_loss = 0
        n_train_data = 0
        net.train()
        for i, (data, target) in enumerate(train_loader):
            y = net(data)
            l = loss_f(y, target)  # noqa: E741
            train_loss += l.sum().item()
            train_acc += (torch.argmax(y, dim=1) == target).float().sum().item()
            n_train_data += data.size(0)
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()

        train_loss /= n_train_data
        train_acc /= n_train_data

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        val_loss, val_acc = evaluate_acc(net, loss_f, val_loader)
        
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch} mean loss: {train_loss}')
        print(f"Train accuracy {train_acc}, Val accuracy: {val_acc}")
        d = {
            'train_acc' : train_accuracies,
            'train_loss' : train_losses,
            'val_acc' : val_accuracies,
            'val_loss' : val_losses
        }
        with open(filename, 'wb') as file:
                pickle.dump(d, file)

        if val_acc > best_val_acc:
            print(f'New best val acc {val_acc}, saving model')
            best_val_acc = val_acc
            if name is not None:
                torch.save(net.state_dict(), name)
        

def train_class(net, train_loader, val_loader, epochs, name: str, filename):
    return train(net, train_loader, val_loader, nn.CrossEntropyLoss(reduction='none'), epochs, name, filename)

def triplet_loss(features: torch.Tensor, labels: torch.Tensor, alpha=0.5):
    """
    triplet loss
    features [N, d] tensor of features for N data points
    labels [N] true labels of the data points

    Implement: max(0, d(a,p) - d(a,n) + alpha )) for a=0:10 and all valid p,n in the batch
    """
    # VoMi: implement
    anchors = torch.arange(10) # anchors
    dists = distances(features, features)
    L = torch.zeros_like(anchors, dtype=torch.float64)
    for a in anchors:
        p = labels == labels[a]
        n = labels != labels[a]

        dp = dists[a, p]
        dn = dists[a, n]

        dp = dp.reshape(1, -1) # Allow broadcasting
        dn = dn.reshape(-1, 1) # Allow broadcasting

        expression = dp - dn + alpha
        expression = torch.nn.functional.relu(expression)
        l = expression.sum()

        L[a] = l

    return L


def train_triplets(net, train_loader, epochs, name: str, filename):
    """
    training with triplet loss
    """
    # VoMi: implement
    return train(net, train_loader, val_loader, triplet_loss, epochs, name, filename)

def smooth_AP_loss(features: torch.Tensor, labels: torch.Tensor, tau=0.5):
    """
    smoothAP loss
    features [N, d] tensor of features for N data points
    labels [N] true labels of the data points

    Implement smoothAP loss as defied in the lab for a=0:10 and all valid p,n in the batch
    """
    # VoMi: implement
    anchors = torch.arange(10) # anchors
    dists = distances(features, features)
    L = torch.zeros_like(anchors, dtype=torch.float64)
    sigmoid = lambda x: torch.sigmoid(x / tau)
    for a in anchors:
        p = labels == labels[a]
        n = labels != labels[a]
        #print(f'{p = }')
        #print(f'{n = }')

        dp = dists[a, p]
        dn = dists[a, n]
        #print(f'{dp = }')
        #print(f'{dn = }')
        
        # Calculate k
        dx = dists[a, :].reshape(1, -1) # Allow broadcasting
        dp = dp.reshape(-1, 1)
        #print(f'{dp = }')
        #print(f'{dx = }')
        tmp = dp - dx
        #print(f'{tmp = }')
        kp = sigmoid(tmp).sum(dim=1)
        #print(f'{kp = }')

        # Calculate numerator
        dn = dn.reshape(1, -1) # Allow broadcasting
        #print(f'{dp = }')
        #print(f'{dn = }')
        tmp = dp - dn
        #print(f'{tmp = }')
        num = sigmoid(tmp).sum(dim=1)
        #print(f'{num = }')

        #print(f'{(num / kp) = }')
        l = (num / kp).sum()

        L[a] = l

    return L


def train_smooth_AP(net, train_loader, epochs, name: str, filename):
    """
    training with smoothAP loss
    """
    # VoMi: implement
    return train(net, train_loader, val_loader, smooth_AP_loss, epochs, name, filename)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--train", type=int, default=-1, help="run training: 0 -- classification loss, 1 -- triplet loss")
    ap.add_argument("--eval", type=int, default=-1, help="run evaluation: 0 -- classification loss, 1 -- triplet loss")
    ap.add_argument("-e", "--epochs", type=int, default=100, help="training epochs")
    args = ap.parse_args()

    if args.train == 0:
        net = new_net()
        print('Training class')
        train_class(net, train_loader, val_loader, epochs=args.epochs, name=model_names[0], filename='pickle/train-class.pickle')
        

    if args.train == 1:
        net = new_net()
        print('Training triplets')
        train_triplets(net, train_loader, epochs=args.epochs, name=model_names[1], filename='pickle/train-triplets.pickle')

    if args.train == 2:
        net = new_net()
        print('Training smooth AP')
        train_smooth_AP(net, train_loader, epochs=args.epochs, name=model_names[2], filename='pickle/train-smoothAP.pickle')
        

    if args.eval > -1:
        print(f'Eval {model_names[args.eval]}')
        net = load_net(model_names[args.eval])

        mAP, mPrec, mRec = evaluate_mAP(net, test_set)
        print(f"Test mAP: {mAP:3.2f}")