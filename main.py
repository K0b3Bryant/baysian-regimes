# Main script: main.py
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from utils import generate_markov_switching_data, plot_results
from config import CONFIG

def main():
    # Load configuration
    n = CONFIG["n"]
    true_state_means = CONFIG["true_state_means"]
    true_state_stds = CONFIG["true_state_stds"]
    p_switch = CONFIG["p_switch"]

    # Generate synthetic data
    data, true_states = generate_markov_switching_data(n, p_switch, true_state_means, true_state_stds)

    # Define the Markov Switching Regression model
    with pm.Model() as model:
        # Transition probabilities (logit parameterization)
        logit_p_00 = pm.Normal("logit_p_00", mu=0, sigma=2)
        logit_p_11 = pm.Normal("logit_p_11", mu=0, sigma=2)

        p_00 = pm.Deterministic("p_00", pm.math.invlogit(logit_p_00))
        p_11 = pm.Deterministic("p_11", pm.math.invlogit(logit_p_11))
        p_01 = 1 - p_00
        p_10 = 1 - p_11

        # Initial state probabilities
        pi_0 = pm.Dirichlet("pi_0", a=np.ones(2))

        # Emission model (state-specific means and stds)
        state_means = pm.Normal("state_means", mu=0, sigma=5, shape=2)
        state_stds = pm.Exponential("state_stds", lam=1, shape=2)

        # Hidden states (categorical)
        states = pm.Categorical("states", p=pi_0, shape=n)

        # Observations
        likelihood = pm.Normal(
            "y_obs",
            mu=pm.math.switch(states, state_means[1], state_means[0]),
            sigma=pm.math.switch(states, state_stds[1], state_stds[0]),
            observed=data,
        )

        # Sampling
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.95)

    # Plot results
    plot_results(data, true_states, trace)

if __name__ == "__main__":
    main()
