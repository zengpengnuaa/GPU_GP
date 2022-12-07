# GPU-based Genetic Programming for Image Classification

This repository is for the paper "Large Scale Image Classification Using GPU-based Genetic Programming".


# Preparation

### Dependencies
This code is tested on Python 3.9.5, and the dependent packages are listed:

- numpy
- sklearn
- deap
- torchvision
- numba

### Datasets
We provide the two datasets for performing the experiments at hand:
- KTH
- CIFAR-10

You can also prepare your own datasets imitating the format of the two datasets above.

# Getting started
#### To search for the programs on KTH dataset, run: 
```
python experiment_kth.py 
```
#### To search for the programs on CIFAR-10 dataset, run: 
```
python experiment_cifar10.py
```
If you have any questions, please feel free to raise "issues" for discussion.