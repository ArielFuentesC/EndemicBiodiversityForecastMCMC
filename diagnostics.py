import os
import stan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Path to the folder containing species data
species_folder = '/home/arielfuentes/Desktop/Statistics/species'

# List all CSV files in the species folder
species_files = [f for f in os.listdir(species_folder) if f.endswith('.csv')]

# Dictionary to store compiled Stan models
stan_models = {}

# Dictionary to store data for each species
species_data = {}

# Load data and compile Stan models for each species
for species_file in species_files:
    # Load species data from CSV
    species_name = os.path.splitext(species_file)[0]
    data = pd.read_csv(os.path.join(species_folder, species_file))

    # Drop rows where 'fechacolecta' is missing
    data = data.dropna(subset=['fechacolecta'])

    # Extract year from 'fechacolecta' column and count occurrences per year
    data['year'] = pd.to_datetime(data['fechacolecta']).dt.year.astype(int)
    grouped_data = data.groupby('year').size().reset_index(name='occurrences')

    print(f"grouped: {grouped_data}")
    # Store data for the species
    species_data[species_name] = {
        'N': len(grouped_data),
        'T': max(grouped_data['year']),
        'year': grouped_data['year'].values,
        'count': grouped_data['occurrences'].values,
    }

    # Define the Stan model code
    stan_code = """
    // Species Occurrence Bayesian Model

    data {
    int<lower=0> N;               // Number of observations
    int<lower=1> T;               // Number of years
    int<lower=1, upper=T> year[N]; // Year of each observation
    int<lower=0> count[N];         // Number of occurrences for each observation
    }

    parameters {
    real<lower=0> alpha;           // Intercept
    real<lower=0> beta_year;       // Slope for year effect
    real<lower=0> sigma;           // Observation error
    real<lower=0> lambda;          // Additional parameter for count distribution
    }

    model {
    // Priors
    alpha ~ normal(0, 10);
    beta_year ~ normal(0, 5);
    sigma ~ cauchy(0, 5);
    lambda ~ normal(0, 1);

    // Likelihood
    for (i in 1:N) {
        count[i] ~ neg_binomial_2_log(alpha + beta_year * year[i], (sigma + lambda * count[i])+1e-6);
    }
    }

    generated quantities {
    // Predictions for future years
    real predicted_counts[T];
    for (t in 1:T) {
        predicted_counts[t] = neg_binomial_2_log_rng(alpha + beta_year * (t - 1), sigma + lambda * count[N]);
    }
    }
    """

    # Compile the Stan model
    stan_models[species_name] = stan.build(stan_code, data=species_data[species_name])

# Function to predict occurrences for a future year
    
def predict_occurrences(species_name):
    model = stan_models[species_name]

    # Fit the Stan model
    fit = model.sample(
        num_chains=3,
        num_samples=700
    )

    # Extract posterior samples
    samples = fit['predicted_counts']

    # Predict occurrences for the next three years
    future_years = range(species_data[species_name]['T'] + 1, species_data[species_name]['T'] + 4)
    predicted_counts = np.mean(samples[:, np.array(future_years) - 1], axis=0)
    prediction_interval = np.percentile(samples[:, np.array(future_years) - 1], [2.5, 97.5], axis=0)

    # Plot observed and predicted occurrences
    plt.figure(figsize=(12, 8))

    # Plot occurrences per year
    plt.plot(species_data[species_name]['year'], species_data[species_name]['count'], label='Observed', marker='o', linestyle='-', color='blue')
    plt.plot(future_years, predicted_counts, marker='o', markersize=8, label='Predicted', color='green')
    plt.fill_between(future_years, prediction_interval[0], prediction_interval[1], color='green', alpha=0.2)

    plt.title(species_name)
    plt.xlabel('Year')
    plt.ylabel('Occurrences')
    plt.legend()

    plt.show()

    return fit

def check_divergences(fit):
    divergent_iter = []
    num_chains = fit['divergent__'].shape[1]

    for chain in range(num_chains):
        divergent_iter_chain = fit['divergent__'][:, chain].nonzero()[0]
        divergent_iter.extend(divergent_iter_chain)

    if divergent_iter:
        print(f"Warning: {len(divergent_iter)} divergent iterations detected.")
        plt.scatter(range(len(divergent_iter)), divergent_iter, marker='x', color='red')
        plt.title('Divergent Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Divergent Transition')
        plt.show()

# Function to plot diagnostics
def plot_diagnostics(fit):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

    # Trace Plots
    az.plot_trace(fit, var_names=["alpha", "beta_year", "sigma", "lambda"])

    # Gelman-Rubin Statistic (Rhat)
    r_hat = az.rhat(fit)
    print("Gelman-Rubin Statistic (Rhat):", r_hat)

    # Effective Sample Size (ESS)
    ess = az.ess(fit)
    print("Effective Sample Size (ESS):", ess)

    # Autocorrelation Plots
    az.plot_autocorr(fit)

    # Energy Plot
    az.plot_energy(fit)

    # Divergences Check
    check_divergences(fit)

    plt.tight_layout()
    plt.show()

# Loop over each species
for species_name, data in species_data.items():
    print(f"Species: {species_name}")
    fit = predict_occurrences(species_name)
    plot_diagnostics(fit)
