import numpy as np
import matplotlib.pyplot as plt

def generate_markov_switching_data(n, p_switch, state_means, state_stds):
    states = np.zeros(n, dtype=int)
    values = np.zeros(n)

    for t in range(1, n):
        if np.random.rand() < p_switch:
            states[t] = 1 - states[t - 1]  # Switch state
        else:
            states[t] = states[t - 1]  # Stay in the same state

        values[t] = np.random.normal(state_means[states[t]], state_stds[states[t]])

    return values, states

def plot_results(data, true_states, trace):
    import arviz as az

    # Plot posterior distribution
    az.plot_trace(trace, var_names=["state_means", "state_stds", "p_00", "p_11"])
    plt.show()

    # Extract hidden states
    posterior_states = trace.posterior["states"].mean(axis=(0, 1))
    predicted_states = np.round(posterior_states)

    # Plot true and predicted states
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Observed Data", color="black")
    plt.plot(true_states, label="True States", linestyle="--", color="blue")
    plt.plot(predicted_states, label="Predicted States", linestyle="--", color="red")
    plt.legend()
    plt.title("Markov Switching Regression: True vs Predicted States")
    plt.show()
