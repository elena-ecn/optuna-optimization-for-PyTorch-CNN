# Hyperparameter optimization study for a PyTorch CNN with Optuna.

This project includes a hyperparameter optimization study of a PyTorch 
Convolutional Neural Network for the MNIST dataset of handwritten digits using 
the hyperparameter optimization framework Optuna.

It shows how to use Optuna with a PyTorch CNN that uses classes (OOP) in order 
to maximize test accuracy.

The CNN hyperparameters chosen to be optimized are: 
- type of optimizer 
- learning rate
- dropout values
- number of convolutional layers
- number of filters of convolutional layers 
- number of neurons of fully connected layers

After the optimization is completed, the program will provide some statistics
about the study and it will show the parameters of the best trial. It will also
display the overall results and save them in a .csv file for future reference. Lastly, it will find and display the most important hyperparameters based on completed trials in the given study.

#### About the dataset
The MNIST dataset contains 60,000 training images and 10,000 testing images,
where each sample is a small, square, 28×28 pixel grayscale image of 
handwritten single digits between 0 and 9.


Technologies
------------
The project is created with:
* Python 3.8
* Torch 1.9.0
* Optuna 2.6.0
* Pandas 1.2.4

Installation
------------

To run this project, install it locally using:
```
git clone https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN.git
```

The following dependencies need to be installed:
* PyTorch: [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
* Optuna: ```pip install optuna```
* Pandas

License
-------
The contents of this repository are covered under the [MIT License](LICENSE).

References
----------
* [optuna.org](https://optuna.org/)
