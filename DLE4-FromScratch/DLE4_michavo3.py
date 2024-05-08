import numpy as np
import code, collections, math, sys, itertools, os

if len(sys.argv) == 1:
    parts_to_run = range(100)
    print(f'Will run all experimens')
else:
    parts_to_run = list(map(int, sys.argv[1:]))
    print(f'Will run {parts_to_run}')


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import matplotlib.pyplot as plt


class CircleDataGenerator():
    def one_class_generate(self, radius, y, n):
        angle = np.random.uniform(0, 2 * np.pi, n)
        noise = np.random.uniform(-1, 1, n)
        r = radius + noise
        x1 = np.cos(angle) * r + 10
        x2 = np.sin(angle) * r + 10
        x = np.stack([x1, x2], axis=1)
        t = np.ones((n,), dtype=np.int64) * int(y)
        return x, t

    def generate_sample(self, n):
        x1, t1 = self.one_class_generate(4, 1, n//2)
        x0, t0 = self.one_class_generate(1, 0, n//2)
        
        x = np.concatenate((x1, x0), axis=0)
        t = np.concatenate((t1, t0), axis=0)

        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.long)


        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        
        return x, t, mu, sigma


def plot_decision_boundary(gt_data, gt_target, model):
    step_size = 0.1
    x_distance = gt_data[:, 0].max().item() - gt_data[:, 0].min().item()
    y_distance = gt_data[:, 1].max().item() - gt_data[:, 1].min().item()
    xmin = gt_data[:, 0].min().item() - x_distance / 5
    xmax = gt_data[:, 0].max().item() + x_distance / 5
    ymin = gt_data[:, 1].min().item() - y_distance / 5
    ymax = gt_data[:, 1].max().item() + y_distance / 5
    xx, yy = torch.meshgrid(torch.arange(xmin, xmax, x_distance / 100), 
                            torch.arange(ymin, ymax, y_distance / 100))
    grid_data = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    model.eval()
    y = model.forward(grid_data.to(dev))
    y = torch.softmax(y, dim=1)
    y = y.detach().cpu().numpy()

    prob = y[:, 0].reshape(xx.shape)

    data = gt_data.detach().cpu().numpy()
    t = gt_target.detach().cpu().numpy()

    plt.imshow(prob.T, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='RdBu')
    plt.contour(xx, yy, prob, [0.5], origin='lower', colors='k')
    plt.plot(data[t == 0, 0], data[t == 0, 1], 'o', color='orange', alpha=1)
    plt.plot(data[t == 1, 0], data[t == 1, 1], 'o', color='lightgreen', alpha=1)
    

torch.manual_seed(1)

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

def save_model(model, name):
    torch.save(model.state_dict(), f'saved-model-{name}')

def load_model(model, name):
    model.load_state_dict(torch.load(f'saved-model-{name}'))
    model.to(dev)

def save_jpg(file_name):
    plt.savefig(file_name, bbox_inches='tight', dpi=199, pad_inches=0)

def init_weights(module, init_strategy):
    if not isinstance(module, torch.nn.Linear):
        return 
        
    n = module.in_features # Layer input size
    m = module.out_features # Layer output size

    if init_strategy == 'Part3 - N(0, 0.1)':
        # For part 3
        torch.nn.init.normal_(module.weight.data, 0, 0.1)
        module.bias = torch.nn.Parameter(torch.zeros_like(module.bias))
        
    elif init_strategy == 'Original heuristic':
        a = np.sqrt(1 / n)
        torch.nn.init.uniform_(module.weight.data, -a, a)

    elif init_strategy == 'Xavier uniform':
        a = np.sqrt(6 / (n + m))
        torch.nn.init.uniform_(module.weight.data, -a, a)

    elif init_strategy == 'Xavier normal':
        torch.nn.init.normal_(module.weight.data, 0, 2 / (n + m))

    elif init_strategy == 'Kaiming uniform':
        a = np.sqrt(3 / n)
        torch.nn.init.uniform_(module.weight.data, -a, a)

    elif init_strategy == 'Kaiming normal':
        #torch.nn.init.normal_(module.weight.data, 0, 2 / n)
        torch.nn.init.kaiming_normal_(module.weight.data)

    else:
        assert False

# Part 1
class Model1(torch.nn.Module):
    def __init__(self, num_layers, layer_width, init_strategy, activation_fun):
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=layer_width, out_features=layer_width, bias=False) for _ in range(num_layers)])

        self.init_strategy = init_strategy
        self.activation_fun = activation_fun

        self.apply(lambda m: init_weights(m, init_strategy))
        

    def init_weights(self, module):
        if not isinstance(module, torch.nn.Linear):
            return 
            
        n = self.layer_width # Layer input size
        m = self.layer_width # Layer output size

        assert module.in_features == n and module.out_features == m

        if self.init_strategy == 'Original heuristic':
            a = np.sqrt(1 / n)
            torch.nn.init.uniform_(module.weight.data, -a, a)

        elif self.init_strategy == 'Xavier uniform':
            a = np.sqrt(6 / (n + m))
            torch.nn.init.uniform_(module.weight.data, -a, a)

        elif self.init_strategy == 'Xavier normal':
            torch.nn.init.normal_(module.weight.data, 0, 2 / (n + m))

        elif self.init_strategy == 'Kaiming uniform':
            a = np.sqrt(3 / n)
            torch.nn.init.uniform_(module.weight.data, -a, a)

        elif self.init_strategy == 'Kaiming normal':
            torch.nn.init.normal_(module.weight.data, 0, 2 / n)
        else:
            assert False

    def forward(self, x):
        """ Forward pass through the neural network
        input:
        x : [batch_size x self.layer_width] 
        output:
        (score, stats), where stats = Stats(means, stds)
        score : [batch_size x self.layer_width] - score
        means : [batch_size x self.num_layers] - mean of activations
        stds : [batch_size x self.num_layers] - standard deviations of activations
        """
        assert x.shape[1] == self.layer_width

        means = np.zeros((self.num_layers, ))
        stds = np.zeros((self.num_layers, ))

        score = x

        for i, layer in enumerate(self.layers):
            score = layer(score)
            score = self.activation_fun(score)

            detached = score.detach().numpy()
            means[i] = abs(detached.mean())
            stds[i] = detached.std()

        return score, means, stds

experiments = [
    ('Original heuristic', torch.nn.functional.tanh, 'Tanh'),
    ('Xavier uniform', torch.nn.functional.tanh, 'Tanh'),
    ('Xavier normal', torch.nn.functional.tanh, 'Tanh'),
    ('Xavier uniform', torch.nn.functional.relu, 'ReLU'),
    ('Xavier normal', torch.nn.functional.relu, 'ReLU'),
    ('Kaiming uniform', torch.nn.functional.relu, 'ReLU'),
    ('Kaiming normal', torch.nn.functional.relu, 'ReLU')
]

unique_experiments = [
    'Original heuristic',
    'Xavier uniform',
    'Xavier normal',
    'Kaiming uniform',
    'Kaiming normal'
]

figsize = (6.0, 6.0 * 3 / 4)

def calculate_normalization_statistics(data):
    n = 0 # number of pixels so far
    mu = torch.tensor([0.0, 0.0]) # mean so far
    sigma2 = torch.tensor([0.0, 0.0]) # variance so far
    for (x, t) in data:
        # Calculate mean and std for this image
        mu = mu + x
        mu_n = mu
        sigma2_n = sigma2

        mu_m = x
        residual = x - mu_m
        sigma2_m = residual.pow(2).mean(dim=[1,2])
        m = x[0].numel() # Number of pixels in each color channel

        mu = m / (m + n) * mu_m + n / (m + n) * mu_n
        sigma2 = m / (m + n) * sigma2_m + n / (m + n) * sigma2_n + m*n/(m+n)**2 * (mu_m - mu_n)**2
        n = n + m

    sigma = sigma2.pow(0.5)
    return mu, sigma

def run_part1():
    print('Running part1')
    num_layers = 50
    layer_width = 512
    batch_size = int(1e4)
    
    x = torch.empty((batch_size, layer_width)).normal_(0, 1)
    t = torch.empty((batch_size, ), dtype=torch.long).random_(layer_width)

    print(f'Input data: {batch_size = }, mean = {x.mean().item()}, std = {x.std().item()}')
    
    fwd_mean = {}
    fwd_std = {}
    grad_mean = {}
    grad_std = {}
    
    for init_strategy, activation_fun, act_fun_name in experiments:
        print(f'Starting with {init_strategy = } and {act_fun_name}')
    
        model = Model1(num_layers, layer_width, init_strategy, activation_fun)
        model.to(dev)
    
        name = f'{init_strategy} with {act_fun_name}'
        score, fwd_mean[name ], fwd_std[name] = model(x)
        assert t.shape[0] == score.shape[0]
        l = torch.nn.functional.cross_entropy(score, t)
    
        l.backward()
    
        grad_mean[name] = np.zeros((num_layers, ))
        grad_std[name] = np.zeros((num_layers, ))
        for i, layer in enumerate(model.layers):
            grad = layer.weight.grad.detach().numpy()
            grad_mean[name][i] = abs(grad.mean())
            grad_std[name][i] = grad.std()

        #part1_plot(f'part1-{init_strategy}-fwd-mean', fwd_mean[init_strategy], plot_fun=plt.semilogy)
        #part1_plot(f'part1-{init_strategy}-fwd-std', fwd_std[init_strategy], plot_fun=plt.semilogy)
        #part1_plot(f'part1-{init_strategy}-bwd-mean', grad_mean[init_strategy])
        #part1_plot(f'part1-{init_strategy}-bwd-std', grad_std[init_strategy])

    part1_plot_shared(f'part1-shared-fwd-mean', fwd_mean, 'Activation mean', plot_fun=plt.semilogy)
    part1_plot_shared(f'part1-shared-fwd-std', fwd_std, 'Activation standard deviation', plot_fun=plt.semilogy)
    part1_plot_shared(f'part1-shared-bwd-mean', grad_mean, 'Gradient mean', plot_fun=plt.semilogy)
    part1_plot_shared(f'part1-shared-bwd-std', grad_std, 'Gradient standard deviation', plot_fun=plt.semilogy)
    print('Done with part1\n\n')



def part1_plot_shared(name, data, ylabel, plot_fun = plt.plot):
    plt.figure(figsize=figsize)
    for n, d in data.items():
        plot_fun(np.arange(len(d)), d, label=n)
    plt.grid(True)
    plt.xlabel("Layer index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{name}.jpg")
    plt.close()


def part1_plot(name, data, plot_fun=plt.plot):
    plt.figure(figsize=figsize)
    plot_fun(np.arange(len(data)), data)
    plt.grid(True)
    plt.xlabel("Layer index")
    plt.ylabel("Activation mean")
    #plt.legend()
    plt.draw()
    save_jpg(f"figures/{name}.jpg")
    plt.close()


# Part 2
class Model2(torch.nn.Module):
    def __init__(self, in_features, layer_widths, init_strategy):
        super().__init__()
        
        self.layer_widths = layer_widths

        in_features = [in_features] + layer_widths
        out_features = layer_widths + [2]

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=i, out_features=o)
            for i, o in zip(in_features, out_features)])

        self.init_strategy = init_strategy

        self.apply(lambda m: init_weights(m, init_strategy))

    def forward(self, x):
        """ Forward pass through the neural network
        input:
        x : [batch_size x self.layer_width] 
        output:
        (score, stats), where stats = Stats(means, stds)
        score : [batch_size x self.layer_width] - score
        means : [batch_size x self.num_layers] - mean of activations
        stds : [batch_size x self.num_layers] - standard deviations of activations
        """
        assert x.shape[1] == 2

        score = x
        for i, layer in enumerate(self.layers):
            score = layer(score)
            if i < len(self.layers) - 1:
#                score = torch.nn.functional.sigmoid(score
#            else:
                score = torch.nn.functional.relu(score)
        return score

def evaluate(model, loader):
    """ Evaluate the model with the given dataset loader """
    #---------Implement-----------
    hits = 0
    total = 0
    loss = 0

    model.eval()

    for (x,t) in loader:
        x = x.to(dev)
        t = t.to(dev)
        score = model(x)
        # TOOD use softmax in the end or signum?
        log_p = torch.log_softmax(score, -1)
        l = torch.nn.functional.nll_loss(log_p, t, reduction='sum')
        if torch.any(torch.isnan(l)):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!NaN in evaluate()!!!!!!!!!!!!!!!')

        loss += l.item()

        _, predictions = torch.max(log_p, dim=1)
        hit_mask = predictions == t
        hits += sum(hit_mask).item()
        total += x.shape[0]

    return loss, hits / total

def training_loop(model, train_loader, val_loader, epochs, lr, name_for_saving, print_epoch=False):
    optimizer = torch.optim.AdamW
    optimizer = optimizer(model.parameters(), lr = lr)

    # training loop
    val_accs = []
    val_losses = []
    training_losses = []
    training_accs = []

    best_epoch = None
    best_validation_accuracy = 0
    
    for e in range(epochs):
        training_loss = 0
        training_hits = 0
        model.train(True)
        for (x,t) in train_loader:
            x = x.to(dev)
            t = t.to(dev)
            score = model(x)
            # TOOD use softmax in the end or signum?
            log_p = torch.log_softmax(score, -1)
            l = torch.nn.functional.cross_entropy(log_p, t, reduction='sum')
            l /= x.shape[0]
            training_loss += l.item()
            if math.isnan(training_loss):
                print(f'Found NaN in training epoch {e}, skipping {epochs - len(training_losses) - 1} epochs!')
                break # end this epoch of learning

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            _, predictions = torch.max(log_p, dim=1)
            training_hits += sum(predictions == t).item()

        training_losses.append(training_loss)
        training_acc = training_hits / len(train_loader.dataset)
        training_accs.append(training_acc)
        # ----------Implement----------
        # Compute loss and accuracy using training set
        val_loss, val_acc = evaluate(model, val_loader)
        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
            best_epoch = e
            save_model(model, name_for_saving)
            #print(f'New best, saved model as {name_for_saving}')

        if print_epoch:
            print(f'Finished epoch {e}: {training_loss = }, {training_acc = }, {val_loss = }, {val_acc = }')

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        # Compute loss and accuracy using validation set
        # Save so-far best model and its validation accuracy
    
    return best_validation_accuracy, best_epoch, (val_accs, val_losses, training_losses, training_accs)

def normalize_dataset(data, mu, std, multiplication_factor):
    return (data - mu) / std * multiplication_factor

def run_part2(name, normalize, multiplication_factor):
    print(f'Runinng part 2 {name}, {normalize = }, {multiplication_factor = }')
    generator = CircleDataGenerator()
    training_set_size = 100
    test_set_size = 1000
    epochs = 5000
    # Training set of size 100
    x_train, t_train, mu, sigma = generator.generate_sample(training_set_size)
    
    train_data = torch.utils.data.TensorDataset(x_train, t_train)
    
    print(f'Train set normalization {mu = }, {sigma = }')

    if normalize:
        x_train = normalize_dataset(x_train, mu, sigma, multiplication_factor)

        print(f'After normalization {x_train.mean(dim=0) = }, {x_train.std(dim=0) = }')
        
    train_data = torch.utils.data.TensorDataset(x_train, t_train)
    # Use 80 - 20 % split (Pareto Rule)
    dataset_train, dataset_val = torch.utils.data.random_split(train_data, [0.8, 0.2])
    # Perform GD (batch size == dataset size)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train), num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=8, num_workers=0)

    x_test, t_test, _, _ = generator.generate_sample(test_set_size)
    if normalize:
        x_test = normalize_dataset(x_test, mu, sigma, multiplication_factor)
    test_data = torch.utils.data.TensorDataset(x_test, t_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)

    data = {}
    
    for init_strategy in unique_experiments:
        torch.manual_seed(3275)
        model = Model2(2, [6, 3], init_strategy)
        model.to(dev)
        
        # Before learning 
        test_loss, test_acc = evaluate(model, test_loader)
        #print(f'Before learning: {test_loss = }, {-np.log(1/2) = }')

        best_validation_accuracy, best_epoch, data[init_strategy] = training_loop(model, train_loader, val_loader, epochs, 3e-4, f'part2-{name}-{init_strategy}')
        test_loss, test_acc = evaluate(model, test_loader)
        print(f'Finished training {init_strategy = }: {best_epoch = }, {best_validation_accuracy = }, {test_loss = }, {test_acc = }')

        load_model(model, f'part2-{name}-{init_strategy}')
        plot_decision_boundary(x_test, t_test, model)
        plt.draw()
        save_jpg(f"figures/part2-{name}-decision-{init_strategy}.jpg")
        plt.close()

    part2_plot(data, 'Training loss', f'part2-{name}-training-loss', 2, epochs)
    part2_plot(data, 'Training accuracy', f'part2-{name}-training-acc', 3, epochs)
    part2_plot(data, 'Validation loss', f'part2-{name}-val-loss', 1, epochs)
    part2_plot(data, 'Validation accuracy', f'part2-{name}-val-acc', 0, epochs)
    print(f'Done with part2 - {name}\n\n')



def part2_plot(data, ylabel, name, index, epochs):
    plt.figure(figsize=figsize)
    x_axis = np.arange(epochs)
    for n, d in data.items():
        plt.plot(x_axis, d[index], label = n)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.draw()
    save_jpg(f"figures/{name}.jpg")
    plt.close()


# Part 3
class Model3(torch.nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            # Dropout on input
            torch.nn.Dropout(dropout_prob),
            
            # Dropout on both hidden layers
            torch.nn.Linear(in_features=784, out_features=800),
            torch.nn.Dropout(dropout_prob),
            
            torch.nn.Linear(in_features=800, out_features=800),
            torch.nn.Dropout(dropout_prob),
            
            torch.nn.Linear(in_features=800, out_features=10),
            ])

        # Init weights from N(0, 0.1) and biases to zero
        self.apply(lambda m: init_weights(m, 'Part3 - N(0, 0.1)'))

    def forward(self, x):
        """ Forward pass through the neural network
        input:
        x : [batch_size x self.layer_width] 
        output:
        (score, stats), where stats = Stats(means, stds)
        score : [batch_size x self.layer_width] - score
        means : [batch_size x self.num_layers] - mean of activations
        stds : [batch_size x self.num_layers] - standard deviations of activations
        """
        score = x
        for layer in self.layers:
            score = layer(score)

        return score


class MNISTData():
    def __init__(self, batch_size):
        # transforms
        transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,)),
                            transforms.Lambda(torch.flatten)])        
        self.train_set = torchvision.datasets.MNIST('./data', train=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST('./data', train=False, transform=transform)

        # split train_set into train_subset and val_subset
        self.train_subset = Subset(self.train_set, list(range(5000)))
        self.val_subset = Subset(self.train_set, list(range(5000, 15000)))

        # dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=0)

def run_part3():
    epochs = 70
    print(f'Runinng part 3 for {epochs} epochs')
    mnist_data = MNISTData(8)

    
    data = {}
    for dropout_p in (0.5, 0.2, 0):
        torch.manual_seed(3275)
        print(f'Starting with {dropout_p = }')
        model = Model3(dropout_p)
        model.to(dev)
        
        best_val_acc, best_epoch, data[f'Dropout probability p = {dropout_p}'] = training_loop(model, mnist_data.train_loader, mnist_data.val_loader, epochs, 3e-4, f'part3-{dropout_p}', print_epoch=True)
        test_loss, test_acc = evaluate(model, mnist_data.test_loader)
        print(f'Finished training {dropout_p = }: {best_epoch = }, {best_val_acc = }, {test_loss = }, {test_acc = }')

    part2_plot(data, 'Training loss', f'part3-training-loss', 2, epochs)
    part2_plot(data, 'Training accuracy', f'part3-training-acc', 3, epochs)
    part2_plot(data, 'Validation loss', f'part3-val-loss', 1, epochs)
    part2_plot(data, 'Validation accuracy', f'part3-val-acc', 0, epochs)
    print('Done with part 3\n\n')

def ensemble_evaluate(model, loader, ensemble_size):
    """ Evaluate the model with the given dataset loader """
    #---------Implement-----------
    hits = 0
    loss = 0

    if ensemble_size == 1:
        model.eval()
    else:
        model.train() # Activate dropout layers

    for (x,t) in loader:
        x = x.to(dev)
        t = t.to(dev)
        score = model(x)
        for _ in range(ensemble_size - 1):
            score += model(x)
        score /= ensemble_size

        log_p = torch.log_softmax(score, -1)
        l = torch.nn.functional.nll_loss(log_p, t, reduction='sum')

        loss += l.item()

        _, predictions = torch.max(log_p, dim=1)
        hit_mask = predictions == t
        hits += sum(hit_mask).item()

    total = len(loader.dataset)
    return hits / total

def run_ensemble():
    print('Starting ensemble')

    mnist_data = MNISTData(8)

    for ensemble_size, dropout_p in itertools.product((5,50,500, 1000), (0, 0.2, 0.5)):
        model = Model3(dropout_p)
        load_model(model, f'part3-{dropout_p}')
        test_acc = ensemble_evaluate(model, mnist_data.test_loader, ensemble_size)
        print(f'{dropout_p} & {ensemble_size} & {test_acc} \\\\')


    print('Done with ensemble\n\n')


if 1 in parts_to_run:
    run_part1()

if 21 in parts_to_run:
    run_part2('nonorm', False, 1.0)
if 22 in parts_to_run:
    run_part2('norm', True, 1.0)
if 23 in parts_to_run:
    run_part2('normmul', True, 0.01)

if 3 in parts_to_run:
    run_part3()

if 31 in parts_to_run:
    run_ensemble()