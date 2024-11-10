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

if len(sys.argv) == 1:
    print('Will run all parts')
    parts_to_run = range(10000)
else:
    parts_to_run = list(map(int, sys.argv[1:]))
    print(f'Will run parts {parts_to_run}')

epochs = 50
print(f'Will run for {epochs} epochs')

figsize = (6.0, 6.0 * 3 / 4)
PAC5_cartoon_train = '/local/temporary/Datasets/PACS_cartoon/train'
PAC5_cartoon_test = '/local/temporary/Datasets/PACS_cartoon/test'
PAC5_cartoon_few_shot_train = '/local/temporary/Datasets/PACS_cartoon_few_shot/train'
PAC5_cartoon_few_shot_test = '/local/temporary/Datasets/PACS_cartoon_few_shot/test'

# Taken from the directory structure
classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

BEST_MODEL_PATH = 'best-model-saved'

#!%load_ext autoreload
#!%autoreload 2
#!%matplotlib inline

## Helper Functions
#def save_pdf(file_name):
#    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)
def save_jpg(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)

def show_tensor_image(img_t, where = None):
    """
    Function to show the image after initial transform, note: need to permute dimensions from the tensor CxWxH convention
    """
    im = img_t.detach()
    im = torch.squeeze(im, 0) if im.dim()==4 else im
    im = im.permute([1,2,0]) if im.dim()==3 else im
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
    global my_gpu
    if 'my_gpu' in globals() and 'cuda' in str(my_gpu):
        dev = my_gpu
    else:
        # find free GPU
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
        print(f'select_device: {memory_used = }')
        ii = np.arange(len(memory_used))
        mask = memory_used < 4000
        print(f'select_device: {mask = }')
        if mask.any():
            mask_index = np.argmin(memory_used[mask])
            index = (ii[mask])[mask_index]
            my_gpu = torch.device(0) # TODO this should be torch.device(index) but it keeps crashing for me...
        else:
            my_gpu = torch.device('cpu')
        dev = my_gpu
        print(f'{dev = }')


# %%
# Load Network
model = torchvision.models.squeezenet1_0(weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT)
# model = torchvision.models.resnet18(pretrained=True)

#print(model.features[0].weight.device)
dev = model.features[0].weight.device

# %% [markdown]
# ### Test Classification


model.eval()
# Load Test Image
img = Image.open("dog.jpg")
if 1 in parts_to_run:
    fig = plt.figure()
    plt.imshow(img)
    fig.axes[0].set_xticks([])
    fig.axes[0].set_yticks([])
    plt.draw()
    save_jpg(f'figures/part1-dog.jpg')
    plt.close()

# reshape as [C x H x W], normalize
to_normalized_tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias=True), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# reshape as [B C H W], move to device
x = to_normalized_tensor(img).unsqueeze(0).to(dev)

# Load labels
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]
    
# TASK PART 1: compute predictions and print top 5 with confidences  
# -------------Implement---------------
if 1 in parts_to_run:
    print('Running part 1')
    scores = model(x)
    probs = torch.nn.functional.softmax(scores[0], dim=0)
    values, indices = torch.sort(probs, descending=True)
    # Select most likely five classes
    for i in range(5):
        print(f"{i+1}-th class {classes[indices[i]]}, probability {100*values[i]:.2f} %")


    # %% [markdown]
    # ### Network First Conv Filters and Features
    # -------------Implement---------------

    kernel_array = plt.figure(figsize=figsize).subplots(8, 12, squeeze=False)
    for row in range(8):
        for col in range(12):
            weights = model.features[0].weight[row*12 + col, :, :, :]
            show_tensor_image(weights, kernel_array[row, col])
            kernel_array[row, col].set_xticks([])
            kernel_array[row, col].set_yticks([])
    plt.draw()
    save_jpg(f'figures/part1-kernels.jpg')
    plt.close()

    grid_array = plt.figure(figsize=figsize).subplots(4, 4, squeeze=False)
    output = model.features[0](x)
    for row in range(4):
        for col in range(4):
            i = row * 4 + col
            tmp = output[:, i]
            show_tensor_image(tmp, grid_array[row, col])
            grid_array[row, col].set_xticks([])
            grid_array[row, col].set_yticks([])
    plt.draw()
    save_jpg(f'figures/part1-lin.jpg')
    plt.close()

    grid_array = plt.figure(figsize=figsize).subplots(4, 4, squeeze=False)
    output = model.features[0:2](x)
    for row in range(4):
        for col in range(4):
            i = row * 4 + col
            tmp = output[:, i]
            show_tensor_image(tmp, grid_array[row, col])
            grid_array[row, col].set_xticks([])
            grid_array[row, col].set_yticks([])
    plt.draw()
    save_jpg(f'figures/part1-nonlin.jpg')
    plt.close()
    print('\n\n\n\n')

# %% [markdown]
# ### Part 2 Data Loader from Image Folders

# %%
train_data = datasets.ImageFolder(PAC5_cartoon_train, transforms.ToTensor())
# %%
# Normalization Statistics
# -------------Implement---------------
# Batch calculation is explained with derivation in https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

# Normalization Statistics
# -------------Implement---------------
def calculate_normalization_statistics(data):
    n = 0 # number of pixels so far
    mu = torch.tensor([0.0, 0.0, 0.0]) # mean so far
    sigma2 = torch.tensor([0.0, 0.0, 0.0]) # variance so far
    for (x, t) in data:
        # Calculate mean and std for this image
        mu_n = mu
        sigma2_n = sigma2

        mu_m = x.mean(dim=[1, 2])
        residual = x - mu_m[:, None, None]
        sigma2_m = residual.pow(2).mean(dim=[1,2])
        m = x[0].numel() # Number of pixels in each color channel

        mu = m / (m + n) * mu_m + n / (m + n) * mu_n
        sigma2 = m / (m + n) * sigma2_m + n / (m + n) * sigma2_n + m*n/(m+n)**2 * (mu_m - mu_n)**2
        n = n + m

    sigma = sigma2.pow(0.5)
    return mu, sigma

mu, sigma = calculate_normalization_statistics(train_data)
if 2 in parts_to_run:
    print('Will run part 2')
    print(f'{mu = }')
    print(f'{sigma = }')
    print('\n\n\n\n')


# %%



# split train data into training and validation
#indices = np.arange(len(train_data))
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(indices), num_workers=0)
#val_loader = torch.utils.data.DataLoader(train_data, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(indices), num_workers=0)

#  End of task 2


# %% [markdown]
# ### Finetune last layer
# %%

# %%
def evaluate(model, loader, show_misclassifications = False):
    """ Evaluate the model with the given dataset loader """
    #---------Implement-----------
    hits = 0
    total = 0
    loss = 0

    misclassifications = []

    model.eval()

    for (x,t) in loader:
        #print(f'{t.shape = }')
        #print(f'{t = }')
        #print(f'{x.shape = }')
        x = x.to(dev)
        t = t.to(dev)
        score = model(x)
        #print(f'{score.shape = }')
        log_p = torch.log_softmax(score, -1)
        _, indices = torch.max(log_p, dim=1)
        l = torch.nn.functional.nll_loss(log_p, t, reduction='sum')
        if torch.any(torch.isnan(l)):
            pass
            #print('!!!!!!!!!!!!!!!!!!!!!!!!!!NaN in evaluate()!!!!!!!!!!!!!!!')
            #print(f'{x.shape = }')
            #print(f'{t = }')
            #print(f'{score.shape = }')
            #print(f'{score = }')
            #print(f'{log_p.shape = }')
            #print(f'{log_p = }')

        loss += l.item() / x.shape[0]
        hit_mask = indices == t
        hits += sum(hit_mask).item()
        total += x.shape[0]

        if show_misclassifications:
            for i in range(x.shape[0]):
                if not hit_mask[i]:
                    probs = torch.exp(log_p[i, :])
                    #assert(abs(probs.sum() - 1) < 1e-3)
                    values, indices = torch.sort(probs, descending=True)
                    # Select most likely five classes
                    misclassifications.append((x[i], t[i], indices[0:5].tolist(), values[0:5].tolist()))

    if show_misclassifications:
        return loss, hits / total, misclassifications
    else:
        return loss, hits / total

# %%

def training_loop(model_fun, train_data, epochs, lr, best_validation_accuracy, best_epoch, allow_training, optimizer = torch.optim.Adam):
    
    torch.manual_seed(1375112354) # 1 Previously

    model = model_fun()
    
    optimizer = optimizer(model.parameters(), lr = lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(indices), num_workers=0)
    #val_loader = torch.utils.data.DataLoader(train_data, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(indices), num_workers=0)
    # Use 80 - 20 % split (Pareto Rule)
    dataset_train, dataset_val = torch.utils.data.random_split(train_data, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=8, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=8, num_workers=0)

    # training loop
    val_accs = []
    val_losses = []
    training_losses = []
    nans_detected = False
    execution_times = []
    for e in range(epochs):
        training_loss = 0
        model.train(allow_training) # Can't allow every time, parts 4 and 64 want training off
        training_start = time.time()
        for (x,t) in train_loader:
            x = x.to(dev)
            t = t.to(dev)
            score = model(x)
            log_p = torch.log_softmax(score, -1)
            l = torch.nn.functional.nll_loss(log_p, t, reduction='sum')
            l /= x.shape[0]
            training_loss += l.item()
            if math.isnan(training_loss) and not nans_detected:
                nans_detected = True
                print(f'Found NaN in training epoch {e}, skipping {epochs - len(training_losses) - 1} epochs!')
                break # end this epoch of learning

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        execution_times.append((time.time() - training_start) / len(dataset_train))  # TODO 
        training_losses.append(training_loss)
        # ----------Implement----------
        # Compute loss and accuracy using training set
        val_loss, val_acc = evaluate(model, val_loader)
        new_best = val_acc > best_validation_accuracy
        if new_best:
            best_validation_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_epoch = e
        #print(f'Epoch {e}: Evaluated {training_loss = :.8f}, {val_loss = :.8f}, {val_acc = :.8f}{" (new best, saved!)" if new_best else ""}')
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        # Compute loss and accuracy using validation set
        # Save so-far best model and its validation accuracy
    
    print(f'Trained: {lr = } {best_validation_accuracy = :.8f}')
    return best_validation_accuracy, best_epoch, (val_accs, val_losses, training_losses, execution_times)


def carry_out_part(all_data, model_fun, train_data, test_data, partname, epochs, allow_training=True):
    print(f'Starting experiment {partname}')
    # freeze all layers but the last one, re-initialze last layer
    # ----------Implement----------
    # optimizer
    data = {}
    all_data[partname] = data
    lr_vec = [0.01, 0.03, 0.001, 0.003, 0.0001]
    
    best_validation_accuracy = 0.0
    best_epoch = None
    best_lr = None

    for lr in lr_vec:

        new_best_validation_accuracy, best_epoch, training_results = training_loop(model_fun, train_data, epochs, lr, best_validation_accuracy, best_epoch, allow_training)
        if new_best_validation_accuracy > best_validation_accuracy:
            best_lr = lr
        best_validation_accuracy = new_best_validation_accuracy
        data[lr] = training_results
    # train and validation loaders

    x_axis = np.arange(epochs) + 1
    # data = (val_acc, val_losses, training_losses)

    plt.figure(figsize=figsize)
    for lr, d in data.items():
        label = f'{lr = }'
        plt.plot(x_axis, d[1], label=label)
    plt.grid(True)
    plt.xlabel("Epoch number")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{partname}-val-loss.jpg")
    plt.close()

    plt.figure(figsize=figsize)
    for lr, d in data.items():
        label = f'{lr = }'
        plt.plot(x_axis, d[0], label=label)
    plt.grid(True)
    plt.xlabel("Epoch number")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{partname}-val-acc.jpg")
    plt.close()

    plt.figure(figsize=figsize)
    for lr, d in data.items():
        label = f'{lr = }'
        plt.plot(x_axis, d[2], label=label)
    plt.grid(True)
    plt.xlabel("Epoch number")
    plt.ylabel("Training loss")
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{partname}-train-loss.jpg")
    plt.close()

    print(f'Best validation accuracy {best_validation_accuracy} at {best_lr = } and {best_epoch = }')
    
    global dev
    best_model = model_fun()
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    best_model.to(dev)

    # %%
    ## Test
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    if True:
        test_loss, test_acc = evaluate(best_model, test_loader)
        print(f'Testing of {partname}: {test_loss = }, {test_acc = }')
        print(f'Summary of {partname}: {best_lr} & {best_epoch} & {best_validation_accuracy*100:.2f} \\% & {test_acc*100:.2f} \\%')
        learning_times = [data_lr[3] for lr, data_lr in data.items()]
        learning_times = np.array(learning_times)

        training_time_mean = np.mean(learning_times)
        training_time_std = np.std(learning_times)
        print(f'Time {partname} & {training_time_mean:.2f} & {training_time_std:.3f} \\\\')
    else:
        test_loss, test_acc, misclassifications = evaluate(best_model, test_loader, True)
        print(f'Testing of {partname}: {test_loss = }, {test_acc = }')

        for i in random.sample(range(len(misclassifications)), 5):
            tensor, true_label, predicted_classes, probs = misclassifications[i]
            fig = plt.figure()
            show_tensor_image(tensor)
            fig.axes[0].set_xticks([])
            fig.axes[0].set_yticks([])
            plt.draw()
            save_jpg(f'figures/{partname}-misclassification-{i}.jpg')
            plt.close()

            print(f'Misclassification {i}, true label {classnames[true_label]}:')
            for prediction, prob in zip(predicted_classes, probs):
                print(f'\t{classnames[prediction]:20s} with prob {prob*100:5.2f} %')

def carry_out_part_test_optimizers(all_data, model_fun, train_data, test_data, partname, epochs, allow_training=True):
    print(f'Starting OPTIMIZER experiment {partname}')
    # freeze all layers but the last one, re-initialze last layer
    # ----------Implement----------
    # optimizer
    data = {}
    lr_vec = [0.01, 0.03, 0.001, 0.003, 0.0001]
    

    for optimizer_name, optimizer in {'SGD' : torch.optim.SGD, 'Adam':torch.optim.Adam}.items():
        data[optimizer_name] = { }
        best_validation_accuracy = 0.0
        best_epoch = None
        best_lr = None
        
        for lr in lr_vec:

            new_best_validation_accuracy, best_epoch, training_results = training_loop(model_fun, train_data, epochs, lr, best_validation_accuracy, best_epoch, allow_training, optimizer)
            if new_best_validation_accuracy > best_validation_accuracy:
                best_lr = lr
            best_validation_accuracy = new_best_validation_accuracy
            data[optimizer_name][lr] = training_results
        # train and validation loaders

        print(f'Best validation accuracy {best_validation_accuracy} at {best_lr = } and {best_epoch = }')

        global dev
        best_model = model_fun()
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
        best_model.to(dev)

        # %%
        ## Test
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)

        test_loss, test_acc = evaluate(best_model, test_loader)
        print(f'Testing of {partname}: {test_loss = }, {test_acc = }')
        print(f'Summary of {partname}: {optimizer_name} & {best_lr} & {best_epoch} & {best_validation_accuracy*100:.2f} \\% & {test_acc*100:.2f} \\%')
        learning_times = [data_lr[3] for lr, data_lr in data[optimizer_name].items()]
        learning_times = np.array(learning_times)

        training_time_mean = np.mean(learning_times)
        training_time_std = np.std(learning_times)
        print(f'Time {partname} & {optimizer_name} & {training_time_mean:.2f} & {training_time_std:.3f} \\\\')
    
    x_axis = np.arange(epochs) + 1

    plt.figure(figsize=figsize)
    colors = []
    for lr, d in data['SGD'].items():
        label = f'SGD {lr = }'
        line, = plt.plot(x_axis, d[0], linestyle='-', label=label)
        colors.append(line.get_color())

    for color, (lr, d) in zip(colors, data['Adam'].items()):
        label = f'Adam {lr = }'
        plt.plot(x_axis, d[0], linestyle='--', label=label, color = color)
    plt.grid(True)
    plt.xlabel("Epoch number")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{partname}-val-acc.jpg")
    plt.close()

    plt.figure(figsize=figsize)
    for lr, d in data['SGD'].items():
        label = f'SGD {lr = }'
        line, = plt.plot(x_axis, d[2], linestyle='-', label=label)
        colors.append(line.get_color())

    for color, (lr, d) in zip(colors, data['Adam'].items()):
        label = f'Adam {lr = }'
        plt.plot(x_axis, d[2], linestyle='--', label=label, color = color)
    plt.grid(True)
    plt.xlabel("Epoch number")
    plt.ylabel("Training loss")
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{partname}-train-loss.jpg")
    plt.close()


    # Error case analyzis
    # ----------Implement----------

# Move model to device
select_device()

to_normalized_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mu, std=sigma)])
train_data = datasets.ImageFolder(PAC5_cartoon_train, to_normalized_tensor)
test_data = datasets.ImageFolder(PAC5_cartoon_test, to_normalized_tensor)

all_data = {}

# Part 3
def prep_part3():
    global dev
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 7) # Only 7 destination classes
    model.to(dev)
    return model

if 3 in parts_to_run:
    carry_out_part(all_data,prep_part3, train_data, test_data, 'part3', epochs)
    print('\n\n\n\n')

#if 31 in parts_to_run:
#    carry_out_part_test_optimizers(all_data,prep_part3, train_data, test_data, 'part31', epochs)
#    print('\n\n\n\n')

# Part 4
def prep_part4():
    global dev
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.train(False)
    model.fc = torch.nn.Linear(512, 7) # Only 7 destination classes
    model.to(dev)
    return model

if 4 in parts_to_run:
    carry_out_part(all_data, prep_part4, train_data, test_data, 'part4', epochs, allow_training=False)
    print('\n\n\n\n')

# Part 5
def prep_part5():
    global dev
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 7)
    model.to(dev)
    return model

if 5 in parts_to_run:
    carry_out_part(all_data,prep_part5, train_data, test_data, 'part5', epochs)
    print('\n\n\n\n')

if 51 in parts_to_run:
    carry_out_part_test_optimizers(all_data,prep_part5, train_data, test_data, 'part51', epochs)
    print('\n\n\n\n')

# Part 6 - Few-Shot
train_data = datasets.ImageFolder(PAC5_cartoon_few_shot_train, transforms.ToTensor())
mu, sigma = calculate_normalization_statistics(train_data)
to_normalized_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mu, std=sigma)])
train_data = datasets.ImageFolder(PAC5_cartoon_few_shot_train, to_normalized_tensor)
test_data = datasets.ImageFolder(PAC5_cartoon_few_shot_test, to_normalized_tensor)


# Part 6 - 3
def prep_part63():
    global dev
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 7) # Only 7 destination classes
    model.to(dev)
    return model

if 63 in parts_to_run:
    carry_out_part(all_data,prep_part63, train_data, test_data, 'part63', epochs)
    print('\n\n\n\n')

# Part 6 - 4
def prep_part64():
    global dev
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.train(False)
    model.fc = torch.nn.Linear(512, 7) # Only 7 destination classes
    model.to(dev)
    return model

if 64 in parts_to_run:
    carry_out_part(all_data,prep_part64, train_data, test_data, 'part64', epochs, allow_training=False)
    print('\n\n\n\n')


# Part 6 - 5
def prep_part65():
    global dev
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 7) # TODO will probably fail
    model.to(dev)
    return model

if 65 in parts_to_run:
    carry_out_part(all_data,prep_part65, train_data, test_data, 'part65', epochs)
    print('\n\n\n\n')

# %% [markdown]
# ### Part 5: Data Augmentation

train_data = datasets.ImageFolder(PAC5_cartoon_few_shot_train, transforms.ToTensor())
mu, sigma = calculate_normalization_statistics(train_data)

to_normalized_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mu, std=sigma)])

augmentations = [
    [transforms.RandomHorizontalFlip(0.5), transforms.RandomAffine(10)]
]

for i in range(len(augmentations)):
    augmentation_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mu, std=sigma),
        ] + augmentations[i])
    print(f'Augmentation {i} with {augmentation_transforms = }')

    train_data = datasets.ImageFolder(PAC5_cartoon_few_shot_train, augmentation_transforms)
    test_data = datasets.ImageFolder(PAC5_cartoon_few_shot_test, to_normalized_tensor)

    # Part 7 - 5
    def prep_part75():
        global dev
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 7) # TODO will probably fail
        model.to(dev)
        return model

    if 75 in parts_to_run:
        carry_out_part(all_data,prep_part75, train_data, test_data, f'part75-{i}', epochs)
        print('\n\n\n\n')

print('Execution times:')
for partname, data in all_data.items():
    learning_times = [data_lr[3] for lr, data_lr in data.items()]
    learning_times = np.array(learning_times)

    training_time_mean = np.mean(learning_times)
    training_time_std = np.std(learning_times)
    
    print(f'{partname} & {training_time_mean:.2f} & {training_time_std:.2f} \\\\')
