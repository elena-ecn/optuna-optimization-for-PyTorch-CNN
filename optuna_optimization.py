"""Optimization study for a PyTorch CNN with Optuna.

Hyperparameter optimization example of a PyTorch Convolutional Neural Network
for the MNIST dataset of handwritten digits using the hyperparameter
optimization framework Optuna.

The MNIST dataset contains 60,000 training images and 10,000 testing images,
where each sample is a small, square, 28×28 pixel grayscale image of handwritten
single digits between 0 and 9.

This script requires installing the following packages: torch, optuna

Author: Elena Oikonomou
Author: elena-ecn
Date: 2021
"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState


class Net(nn.Module):
    """CNN for the MNIST dataset of handwritten digits.

    Attributes:
        - convs (torch.nn.modules.container.ModuleList):   List with the convolutional layers
        - conv2_drop (torch.nn.modules.dropout.Dropout2d): Dropout for conv layer 2
        - out_feature (int):                               Size of flattened features
        - fc1 (torch.nn.modules.linear.Linear):            Fully Connected layer 1
        - fc2 (torch.nn.modules.linear.Linear):            Fully Connected layer 2
        - p1 (float):                                      Dropout ratio for FC1

    Methods:
        - forward(x): Does forward propagation
    """
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1):
        """Parameters:
            - trial (optuna.trial._trial.Trial): Optuna trial
            - num_conv_layers (int):             Number of convolutional layers
            - num_filters (list):                Number of filters of conv layers
            - num_neurons (int):                 Number of neurons of FC layers
            - drop_conv2 (float):                Dropout ratio for conv layer 2
            - drop_fc1 (float):                  Dropout ratio for FC1
        """
        super(Net, self).__init__()                                                     # Initialize parent class
        in_size = 28                                                                    # Input image size (28 pixels)
        kernel_size = 3                                                                 # Convolution filter size

        # Define the convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], kernel_size=(3, 3))])  # List with the Conv layers
        out_size = in_size - kernel_size + 1                                            # Size of the output kernel
        out_size = int(out_size / 2)                                                    # Size after pooling
        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=(3, 3)))
            out_size = out_size - kernel_size + 1                                       # Size of the output kernel
            out_size = int(out_size/2)                                                  # Size after pooling

        self.conv2_drop = nn.Dropout2d(p=drop_conv2)                                    # Dropout for conv2
        self.out_feature = num_filters[num_conv_layers-1] * out_size * out_size         # Size of flattened features
        self.fc1 = nn.Linear(self.out_feature, num_neurons)                             # Fully Connected layer 1
        self.fc2 = nn.Linear(num_neurons, 10)                                           # Fully Connected layer 2
        self.p1 = drop_fc1                                                              # Dropout ratio for FC1

        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        """Forward propagation.

        Parameters:
            - x (torch.Tensor): Input tensor of size [N,1,28,28]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,10]
        """
        for i, conv_i in enumerate(self.convs):  # For each convolutional layer
            if i == 2:  # Add dropout if layer 2
                x = F.relu(F.max_pool2d(self.conv2_drop(conv_i(x)), 2))  # Conv_i, dropout, max-pooling, RelU
            else:
                x = F.relu(F.max_pool2d(conv_i(x), 2))                   # Conv_i, max-pooling, RelU

        x = x.view(-1, self.out_feature)                     # Flatten tensor
        x = F.relu(self.fc1(x))                              # FC1, RelU
        x = F.dropout(x, p=self.p1, training=self.training)  # Apply dropout after FC1 only when training
        x = self.fc2(x)                                      # FC2

        return F.log_softmax(x, dim=1)                       # log(softmax(x))


def train(network, optimizer):
    """Trains the model.

    Parameters:
        - network (__main__.Net):              The CNN
        - optimizer (torch.optim.<optimizer>): The optimizer for the CNN
    """
    network.train()  # Set the module in training mode (only affects certain modules)
    for batch_i, (data, target) in enumerate(train_loader):  # For each batch

        # Limit training data for faster computation
        if batch_i * batch_size_train > number_of_train_examples:
            break

        optimizer.zero_grad()                                 # Clear gradients
        output = network(data.to(device))                     # Forward propagation
        loss = F.nll_loss(output, target.to(device))          # Compute loss (negative log likelihood: −log(y))
        loss.backward()                                       # Compute gradients
        optimizer.step()                                      # Update weights


def test(network):
    """Tests the model.

    Parameters:
        - network (__main__.Net): The CNN

    Returns:
        - accuracy_test (torch.Tensor): The test accuracy
    """
    network.eval()         # Set the module in evaluation mode (only affects certain modules)
    correct = 0
    with torch.no_grad():  # Disable gradient calculation (when you are sure that you will not call Tensor.backward())
        for batch_i, (data, target) in enumerate(test_loader):  # For each batch

            # Limit testing data for faster computation
            if batch_i * batch_size_test > number_of_test_examples:
                break

            output = network(data.to(device))               # Forward propagation
            pred = output.data.max(1, keepdim=True)[1]      # Find max value in each row, return indexes of max values
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()  # Compute correct predictions

    accuracy_test = correct / len(test_loader.dataset)

    return accuracy_test


def objective(trial):
    """Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 3)  # Number of convolutional layers
    num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16))
                   for i in range(num_conv_layers)]              # Number of filters for the convolutional layers
    num_neurons = trial.suggest_int("num_neurons", 10, 400, 10)  # Number of neurons of FC1 layer
    drop_conv2 = trial.suggest_float("drop_conv2", 0.2, 0.5)     # Dropout for convolutional layer 2
    drop_fc1 = trial.suggest_float("drop_fc1", 0.2, 0.5)         # Dropout for FC1 layer

    # Generate the model
    model = Net(trial, num_conv_layers, num_filters, num_neurons, drop_conv2,  drop_fc1).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer)  # Train the model
        accuracy = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 10                         # Number of training epochs
    batch_size_train = 64                 # Batch size for training data
    batch_size_test = 1000                # Batch size for testing data
    number_of_trials = 100                # Number of Optuna trials
    limit_obs = True                      # Limit number of observations for faster computation

    # *** Note: For more accurate results, do not limit the observations.
    #           If not limited, however, it might take a very long time to run.
    #           Another option is to limit the number of epochs. ***

    if limit_obs:  # Limit number of observations
        number_of_train_examples = 500 * batch_size_train  # Max train observations
        number_of_test_examples = 5 * batch_size_test      # Max test observations
    else:
        number_of_train_examples = 60000                   # Max train observations
        number_of_test_examples = 10000                    # Max test observations
    # -------------------------------------------------------------------------

    # Make runs repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)

    # Create directory 'files', if it doesn't exist, to save the dataset
    directory_name = 'files'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    # Download MNIST dataset to 'files' directory and normalize it
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_test, shuffle=True)

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of completed and pruned trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Compute and show the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))

    # Show results in a dataframe and save to file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    print("\nOverall Results (ordered by accuracy):\n {}".format(df))
