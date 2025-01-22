# Markov Switching Regression with Hierarchical Priors

This project implements a Markov Switching Regression model using PyMC for Bayesian inference. The model estimates state-specific parameters (means and standard deviations) and transition probabilities for a time series. Hierarchical priors are used to incorporate higher-level uncertainty in state-specific parameters.

## Features
- **Hierarchical Priors**:
  - State-specific means and standard deviations are modeled using hyperpriors, reflecting uncertainty and allowing parameter sharing across states.
- **Transition Probabilities**:
  - Transition probabilities between states are estimated and extracted from the posterior distribution.
- **State Inference**:
  - Hidden states (0 or 1) are inferred for the entire time series.
- **Visualization**:
  - Observed data and inferred states are plotted for easy analysis.

## File Structure
- **`main.py`**: Contains the main script to load data, define the model, sample from the posterior, and visualize results.
- **`utils.py`**: Utility functions for loading data, extracting states, computing transition probabilities, and plotting results.
- **`config.py`**: Configuration file for specifying input file paths and model parameters.
