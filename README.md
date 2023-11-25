# Restricted Boltzmann Machine (RBM) Implementation with Gibbs Sampling

## Overview

This repository provides an implementation of the Restricted Boltzmann Machine (RBM) using Gibbs sampling. RBM, belonging to the generative Energy-based model family, employs an energy function to calculate a score for a given configuration of visible units. The model bipartite interactions between visible and hidden layers, induce high-order interactions among visible nodes.

The RBM structure is characterized by the independence of each unit in the visible (hidden) layer, given the state of the hidden (visible) layer. This independence facilitates the use of a quick sampling scheme known as Gibbs sampling.

## Contents

### 1. RBM_torch.ipynb

This notebook serves as the main file containing the complete implementation of the RBM. It covers the initialization of model parameters, Gibbs sampling for training using contrastive divergence, and functions for computing free energy, log-likelihood, and reconstruction error. The notebook also includes practical examples, such as training the model on the MNIST dataset and generating reconstructed images.


### 2. AIS_rbm

This notebook introduces the Annealed Importance Sampling (AIS) algorithm to estimate the log partition function. The AIS algorithm plays a crucial role in evaluating the progress of log-likelihood in energy-based models, providing valuable insights into the model's performance.

### 3. Langevian_NB_RBM

This notebook deviates from the standard practice for training the Restricted Boltzmann Machine (RBM). In this unconventional approach, Gibbs sampling is replaced by Langevian Monte Carlo sampling. This adventurous technique explores alternative avenues for model training, potentially offering unique advantages or insights.


