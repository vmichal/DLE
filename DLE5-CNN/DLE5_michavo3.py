# %%
## Import
import os, sys, scipy, code, math, random, time
import numpy as np

import torch
#print(torch.__version__)
import torch.nn.functional
import torchvision.models
from torchvision import transforms,datasets
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
figsize = (6.0, 6.0 * 3 / 4)

#!%load_ext autoreload
#!%autoreload 2
#!%matplotlib inline


def save_jpg(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)

def show_tensor_image(img_t, where = None, normalize = True):
    """
    Function to show the image after initial transform, note: need to permute dimensions from the tensor CxWxH convention
    """
    # TODO alternative code from the assignment
    # grid = make_grid(x, nrow=10, normalize=True, padding=1)
    # image = grid.cpu().numpy().transpose(1, 2, 0)
    # plt.imshow(image)

    im = img_t.detach()
    im = torch.squeeze(im, 0) if im.dim()==4 else im
    im = im.permute([1,2,0]) if im.dim()==3 else im
    if normalize:
        im = im - im.min()
        im /= im.max()
    if where is None:
        plt.imshow(im.cpu().numpy())
    else:
        where.imshow(im.cpu().numpy())

def select_device():
    """ 
    Find the CUDA device with max available memory and set the global dev variable to it
    If less than 4GB memory on all devices, resort to dev='cpu'
    Repeated calls to the function select the same GPU previously selected
    """
    global dev
    if os.name == 'nt':
        dev = torch.device('cpu')
        return
    if 'my_gpu' in globals() and 'cuda' in str(my_gpu):
        dev = my_gpu
    else:
        # find free GPU
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
        #print(f'select_device: {memory_used = }')
        ii = np.arange(len(memory_used))
        mask = memory_used < 4000
        #print(f'select_device: {mask = }')
        if mask.any():
            mask_index = np.argmin(memory_used[mask])
            index = (ii[mask])[mask_index]
            my_gpu = torch.device(0) # TODO this should be torch.device(index) but it keeps crashing for me...
        else:
            my_gpu = torch.device('cpu')
        dev = my_gpu
        print(f'{dev = }')

select_device()

# %%
# Load Network
# load network
model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
model = model.eval().to(dev)
 
# we are not changing the network weights/biases in this lab
for param in model.parameters():
    param.requires_grad = False

# Load labels
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]

# reshape as [C x H x W], normalize
to_normalized_tensor = transforms.Compose([
    #transforms.Resize(224, antialias=True), # TODO consider using antialias
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# %% [markdown]
# ### Part 1: Classify dog.jpg, compute predictions and print top 10 with confidences  
# Load Test Image
img = Image.open("dog.jpg")
fig = plt.figure()
plt.imshow(img)
fig.axes[0].set_xticks([])
fig.axes[0].set_yticks([])
plt.draw()
save_jpg(f'figures/part1-dog.jpg')
plt.close()

# reshape image as [B C H W], move to device
x_dog = to_normalized_tensor(img).unsqueeze(0).to(dev)

scores = model(x_dog)
probs = torch.nn.functional.softmax(scores[0], dim=0)
probs, indices = torch.sort(probs, descending=True)
true_class = indices[0].item()
# Select most likely ten classes
print(f'10 most likely classes for dog.jpg:')
for i in range(10):
    print(f"{i+1} & {classes[indices[i]]} & {indices[i]} & {100*probs[i]:.2f} \\% \\\\")

fig = plt.figure()
show_tensor_image(x_dog)
fig.axes[0].set_xticks([])
fig.axes[0].set_yticks([])
plt.draw()
save_jpg(f'figures/part1-tensor.jpg')
plt.close()



# %%
# Task 2
x = x_dog
x.requires_grad = True
def plot_l2_norms(x, where, name):
    f = (x.detach()**2).sum(dim=1).sqrt()[0]
    where.imshow(f.cpu().numpy(), cmap='jet')
    where.set_axis_off()
    where.set_title(f"{i}:{name}")

gradients = []
plt.subplots_adjust
fig, axs = plt.subplots(nrows=3, ncols=7, constrained_layout=True)
for (i,l) in enumerate(model.features):
    x = l.forward(x)    
    plot_l2_norms(x, axs.flat[i], l.__class__.__name__)

    x.retain_grad()
    gradients.append(x)

plt.draw()
save_jpg(f'figures/part2-norms.jpg')
plt.close()

# Finish the forward pass through the model
x = model.avgpool(x)
x = torch.flatten(x, 1)
scores = model.classifier(x)
score = scores[0, indices[0]] # The score for the most probable output
score.backward()

fig, axs = plt.subplots(nrows=3, ncols=7, constrained_layout=True)
for i,x in enumerate(gradients):
    plot_l2_norms(x.grad, axs.flat[i], model.features[i].__class__.__name__)

plt.draw()
save_jpg(f'figures/part2-grads.jpg')
plt.close()

# %%
## Part 3
def get_layer_k_s(model, layer_index):
    layer = model.features[layer_index]
    if isinstance(layer, torch.nn.ReLU):
        return 1, 1
    if isinstance(layer, torch.nn.Conv2d):
        return layer.kernel_size[0], layer.stride[0]
    if isinstance(layer, torch.nn.MaxPool2d):
        return layer.kernel_size, layer.stride
    assert False # No other layers in our network
    
def receptive_field(model, target_layer_index):
    S = 1 # Size of the receptive field
    T = 1 # Stride of the receptive field
    for l in range(target_layer_index + 1):
        # Calculate T and S for l
        k, s = get_layer_k_s(model, l)
        
        S = k * T + (S - T)
        T = s * T
    return S


def activation_max(model, target_layer_index, require_smooth, epsilon):
    receptive_field_size = receptive_field(model, target_layer_index)
    print(f'{receptive_field_size = }')
    layers = model.features[0:target_layer_index+1]
    print(f'{layers = }')
    channels = layers[-1].out_channels
    print(f'{channels = }')
    x = torch.nn.Parameter(torch.zeros(channels, 3, receptive_field_size, receptive_field_size)).to(dev)

    optimizer = torch.optim.Adam([x], lr = 3e-4, maximize=True)
    costs = []
    apool = torch.nn.AvgPool2d(3, padding=0, stride=1)
    apad = torch.nn.ReplicationPad2d(1)
    for e in range(500):
        # Propagate the image through through features up to the target
        y = layers(x)
        sz = y.shape
        objective = y[:,:,sz[2]//2, sz[3]//2].diag().sum()
        if math.isnan(objective):
            print(f'Found NaN in training epoch {e}!')
            break # end this epoch of learning

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            x.data = x.data.clamp(min=-1.0, max=1.0)

            if require_smooth:
                xx = apool(apad(x))
                diff = x - xx
                dn = torch.linalg.norm(diff.flatten(2), dim=2, ord=1.0) / (receptive_field_size * receptive_field_size)
                if dn.max() > epsilon:
                    x.data[dn > epsilon] = xx[dn > epsilon]

        # Store the objective to plot later
        costs.append(objective.detach().numpy().item())
        print(f'{x.data.max() = }, {x.data.min() = }')

    return x, costs
    
x, smooth_costs = activation_max(model, 8, True, 1e-2)
print(f'{x.shape = }, {x.data.max() = }, {x.data.max() = }')
print(f'{len(smooth_costs) = }')


# Plot first 8
nrows = 4
ncols = 6
pattern_array = plt.figure().subplots(nrows=nrows, ncols=ncols, squeeze=True)
for i in range(nrows * ncols):
    to_plot = x[i, :,:,:]
    show_tensor_image(to_plot, pattern_array.flat[i])
    pattern_array.flat[i].set_xticks([])
    pattern_array.flat[i].set_yticks([])
plt.draw()
save_jpg(f'figures/part3-patterns-smooth.jpg')
plt.close()

x, nonsmooth_costs = activation_max(model, 8, False, 1e-2)
print(f'{x.shape = }, {x.data.max() = }, {x.data.max() = }')
print(f'{len(nonsmooth_costs) = }')


plt.figure(figsize=figsize)
plt.plot(np.arange(len(smooth_costs)) + 1, smooth_costs, label='Smooth pattern')
plt.plot(np.arange(len(nonsmooth_costs)) + 1, nonsmooth_costs, label='Non-smooth pattern')
plt.grid(True)
plt.xlabel("Epoch number")
plt.ylabel("Objective value")
plt.legend()
plt.draw()
save_jpg(f"figures/part3-objective.jpg")
plt.close()

# Plot first 8
pattern_array = plt.figure().subplots(nrows=nrows, ncols=ncols, squeeze=True)
for i in range(nrows * ncols):
    to_plot = x[i, :,:,:]
    show_tensor_image(to_plot, pattern_array.flat[i])
    pattern_array.flat[i].set_xticks([])
    pattern_array.flat[i].set_yticks([])
plt.draw()
save_jpg(f'figures/part3-patterns-nonsmooth.jpg')
plt.close()

    
# %% Task 4
target_class = 892
target_class_name = classes[target_class]
true_class_name = classes[true_class]

print(f'Target class {target_class_name} ({target_class}).')
print(f'True class {true_class_name} ({true_class}).')
epsilon = 1

def attack(model, x0, target_class, eps, true_class):
    x = x0.clone()
    x.requires_grad = True

    optimizer = torch.optim.Adam([x], lr = 3e-4, maximize=True)
    true_class_indices = []
    target_class_indices = []
    target_class_probs = []
    true_class_probs = []
    predicted_classes = []

    apool = torch.nn.AvgPool2d(3, padding=0, stride=1)
    apad = torch.nn.ReplicationPad2d(1)
    for e in range(250):
        # Propagate the image through through features up to the target
        scores = model(x)

        probs = torch.nn.functional.softmax(scores[0], dim=0)

        to_maximize = probs[target_class]

        optimizer.zero_grad()
        to_maximize.backward()
        optimizer.step()

        with torch.no_grad():
            dx = (x.detach() - x0)
            dn = dx.flatten().norm(p=float('inf'))
            div = torch.clamp(dn/eps, min=1.0)
            dx = dx / div
            x.data = x0 + dx

        # Store the objective to plot later
        _, indices = torch.sort(probs, descending=True)

        target_class_probs.append(probs[target_class].item())
        true_class_probs.append(probs[true_class].item())
        target_class_indices.append((indices == target_class).nonzero(as_tuple=True)[0].item())
        true_class_indices.append((indices == true_class).nonzero(as_tuple=True)[0].item())
        predicted_classes.append(indices[0].item())

    return x, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes

data = {}
eps_vector = [1e-3, 1e-2, 3e-2, 1e-1]
for eps in eps_vector:
    data[eps] = attack(model, x_dog, target_class, eps, true_class)


plt.figure(figsize=figsize)
x_axis = np.arange(len(data[eps_vector[-1]][1])) + 1
print(f'{x_axis.shape}')
for eps, (x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes) in data.items():
    print(f'{eps = }')
    print(f'{x_axis = }')
    print(f'{target_class_probs = }')
    plt.plot(x_axis, target_class_probs, label=f'$\\varepsilon = {eps}$')
plt.grid(True)
plt.xlabel("Epoch number")
plt.ylabel("Prediction confidence [-]")
plt.legend()
plt.draw()
save_jpg(f"figures/part4-probs-target.jpg")
plt.close()


plt.figure(figsize=figsize)
for eps, (x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes) in data.items():
    plt.plot(x_axis, true_class_probs, label=f'$\\varepsilon = {eps}$')
plt.grid(True)
plt.xlabel("Epoch number")
plt.ylabel("Prediction confidence [-]")
plt.legend()
plt.draw()
save_jpg(f"figures/part4-probs-true.jpg")
plt.close()



plt.figure(figsize=figsize)
for eps, (x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes) in data.items():
    plt.plot(x_axis, 1 + np.array(target_class_indices), label=f'$\\varepsilon = {eps}$')
plt.grid(True)
plt.xlabel("Epoch number")
plt.ylabel("Position")
plt.legend()
plt.draw()
save_jpg(f"figures/part4-indices-target.jpg")
plt.close()

plt.figure(figsize=figsize)
for eps, (x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes) in data.items():
    plt.plot(x_axis, 1 + np.array(true_class_indices), label=f'$\\varepsilon = {eps}$')
plt.grid(True)
plt.xlabel("Epoch number")
plt.ylabel("Position")
plt.legend()
plt.draw()
save_jpg(f"figures/part4-indices-true.jpg")
plt.close()


plt.figure(figsize=figsize)
for eps, (x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes) in data.items():
    plt.plot(x_axis, predicted_classes, label=f'$\\varepsilon = {eps}$')
plt.grid(True)
plt.xlabel("Epoch number")
plt.ylabel("Predicted class")
plt.legend()
plt.draw()
save_jpg(f"figures/part4-pred.jpg")
plt.close()

# Plot only for epsilon = 1
x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes = data[eps_vector[-1]]

fig = plt.figure(figsize=figsize)
show_tensor_image(x_attack)
fig.axes[0].set_xticks([])
fig.axes[0].set_yticks([])
plt.draw()
save_jpg(f'figures/part4-image.jpg')
plt.close()

fig = plt.figure(figsize=figsize)
show_tensor_image(x_dog - x_attack)
fig.axes[0].set_xticks([])
fig.axes[0].set_yticks([])
plt.draw()
save_jpg(f'figures/part4-image-difference-normalized.jpg')
plt.close()

diff = x_dog - x_attack
fig = plt.figure(figsize=figsize)
show_tensor_image(diff, None, False)
fig.axes[0].set_xticks([])
fig.axes[0].set_yticks([])
plt.draw()
save_jpg(f'figures/part4-image-difference.jpg')
plt.close()

for eps, (x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes) in data.items():
    diff = x_dog - x_attack
    mean = diff.mean(dim=[2,3]).detach().squeeze().numpy()
    std = diff.std(dim=[2,3]).detach().squeeze().numpy()
    print(f'{eps} & ( {mean[0]:.2e}, {mean[1]:.2e}, {mean[2]:.2e}) & ({std[0]:.2e}, {std[1]:.2e}, {std[2]:.2e}) \\\\')

x_attack, target_class_probs, target_class_indices, true_class_probs, true_class_indices, predicted_classes = data[eps_vector[-2]]
predicted_class_names = [f'{classes[pred]} ({pred})' for pred in predicted_classes]
pd.Series(predicted_class_names).value_counts(sort=False).plot(kind='bar')
plt.ylabel('# Occurences')
plt.draw()
save_jpg(f'figures/part4-pred-histogram.jpg')