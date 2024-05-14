import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tools import Dataset_to_XY, DataXY
from argparse import ArgumentParser

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


def distances(f1: torch.Tensor, f2: torch.Tensor):
    """All pairwise distances between feature vectors in f1 and feature vectors in f2:
    f1: [N, d] array of N normalized feature vectors of dimension d
    f2: [M, d] array of M normalized feature vectors of dimension d
    return D [N, M] -- pairwise Euclidean distance matrix
    """
    assert (f1.dim() == 2)
    assert (f2.dim() == 2)
    assert (f1.size(1) == f2.size(1))
    # TODO: implement


def evaluate_AP(dist: np.array, labels: np.array, query_label: int):
    """Average Precision
    dits: [N] array of distances to all documents from the query
    labels: [N] labels of all documents
    query_label: label of the query document
    return: AP -- average precision, Prec -- Precision, Rec -- Recall
    """
    ii = np.argsort(dist)
    dist = dist[ii]
    labels = labels[ii]
    rel = np.equal(labels, query_label).astype(int)
    # TODO: implement the computation
    AP = 0
    Prec = 0
    Rec = 0
    return AP, Prec, Rec


def evaluate_mAP(net, dataset: DataXY):
    """
    Compute Mean Average Precision
    net: a network with features() method 
    dataset: dataset of input images and labels
    Returns: mAP -- mean average precision, mRec -- mean recall, mPrec -- mean precision
    """
    torch.manual_seed(1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    # use first 100 samples from this loader as queries, use all samples as possible items to retrive, exclude the query from the retrieved items
    # TODO: implement
    # will need to call evaluate_AP
    mAP = 0
    mPrec = 0
    mRec = 0
    return mAP, mPrec, mRec


def evaluate_acc(net, loss_f, loader):
    net.eval()
    with torch.no_grad():
        acc = 0
        loss = 0
        n_data = 0
        for i, (data, target) in enumerate(loader):
            data, target = data, target
            y = net(data)
            l = loss_f(y, target)  # noqa: E741
            loss += l.sum()
            acc += (torch.argmax(y, dim=1) == target).float().sum().item()
            n_data += data.size(0)
        acc /= n_data
        loss /= n_data
    return (loss, acc)


def train_class(net, train_loader, val_loader, epochs=20, name: str = None):
    loss_f = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
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
        #
        val_loss, val_acc = evaluate_acc(net, loss_f, val_loader)
        print(f'Epoch: {epoch} mean loss: {train_loss}')
        print("Train accuracy {}, Val accuracy: {}".format(train_acc, val_acc))
        if name is not None:
            torch.save(net.state_dict(), name)


def triplet_loss(features: torch.Tensor, labels: torch.Tensor, alpha=0.5):
    """
    triplet loss
    features [N, d] tensor of features for N data points
    labels [N] true labels of the data points

    Implement: max(0, d(a,p) - d(a,n) + alpha )) for a=0:10 and all valid p,n in the batch
    """
    L = 0
    # TODO: implement
    return L


def train_triplets(net, train_loader, epochs=20, name: str = None):
    """
    training with triplet loss
    """
    # TODO: implement
    pass


def smooth_AP_loss(features: torch.Tensor, labels: torch.Tensor, tau=0.5):
    """
    smoothAP loss
    features [N, d] tensor of features for N data points
    labels [N] true labels of the data points

    Implement smoothAP loss as defied in the lab for a=0:10 and all valid p,n in the batch
    """
    L = 0
    # TODO: implement
    return L


def train_smooth_AP(net, train_loader, epochs=20, name: str = None):
    """
    training with smoothAP loss
    """
    # TODO: implement
    pass


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--train", type=int, default=-1, help="run training: 0 -- classification loss, 1 -- triplet loss")
    ap.add_argument("--eval", type=int, default=-1, help="run evaluation: 0 -- classification loss, 1 -- triplet loss")
    ap.add_argument("-e", "--epochs", type=int, default=100, help="training epochs")
    args = ap.parse_args()

    if args.train == 0:
        net = ConvNet(10)
        net.to(dev)
        train_class(net, train_loader, val_loader, epochs=args.epochs, name=model_names[0])

    if args.train == 1:
        net = ConvNet(10)
        net.to(dev)
        train_triplets(net, train_loader, epochs=args.epochs, name=model_names[1])

    if args.train == 2:
        net = ConvNet(10)
        net.to(dev)
        train_smooth_AP(net, train_loader, epochs=args.epochs, name=model_names[2])

    if args.eval > -1:
        net = load_net(model_names[args.eval])

        mAP, mPrec, mRec = evaluate_mAP(net, test_set)
        print(f"Test mAP: {mAP:3.2f}")