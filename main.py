import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from utils import load_time_series, categorize_states, plot_results
from config import CONFIG

def main():
    # Load the time series data
    data = load_time_series(CONFIG["time_series_path"])

    # Define the Markov Switching Regression model
    n = len(data)
    with pm.Model() as model:
        # Transition probabilities (logit parameterization)
        # The prior assumes a normal distribution centered at 0 with standard deviation 2.
        # This reflects a neutral assumption, allowing flexibility in transition probabilities
        # while not overly constraining the model to specific values.
        logit_p_00 = pm.Normal("logit_p_00", mu=0, sigma=2)
        logit_p_11 = pm.Normal("logit_p_11", mu=0, sigma=2)

        p_00 = pm.Deterministic("p_00", pm.math.invlogit(logit_p_00))
        p_11 = pm.Deterministic("p_11", pm.math.invlogit(logit_p_11))
        p_01 = 1 - p_00
        p_10 = 1 - p_11

        # Initial state probabilities
        # The prior assumes a uniform Dirichlet distribution, reflecting equal uncertainty
        # about the probability of starting in either state.
        pi_0 = pm.Dirichlet("pi_0", a=np.ones(2))

        # Emission model (state-specific means and stds)
        # The prior for state means assumes a normal distribution centered at 0 with a wide standard deviation of 5.
        # This choice reflects a diffuse prior, indicating limited prior knowledge about the actual means.
        # It allows the data to primarily influence the posterior estimates.
        state_means = pm.Normal("state_means", mu=0, sigma=5, shape=2)

        # The prior for state standard deviations assumes an exponential distribution with a rate parameter of 1.
        # This reflects a preference for smaller standard deviations but still allows for larger values if supported by the data.
        # Consider whether this aligns with expectations about variability in the states.
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

    # Categorize states and plot results
    predicted_states = categorize_states(trace, n)
    plot_results(data, predicted_states)

if __name__ == "__main__":
    main()
