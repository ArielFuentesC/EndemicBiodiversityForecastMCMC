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

    # Define the Stan model code (use the previous Stan code)
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


def plot_diagnostics(fit, species_name):
    # Plot trace plots for selected parameters
    az.plot_trace(fit, var_names=["alpha", "beta_year", "sigma", "lambda"])
    plt.suptitle(f"Trace Plots for {species_name}")

    # Plot Gelman-Rubin statistic
    az.plot_forest(fit, var_names=["alpha", "beta_year", "sigma", "lambda"], kind="ridgeplot")
    plt.suptitle(f"Gelman-Rubin Statistic for {species_name}")

    # Plot effective sample size (ESS)
    az.plot_ess(fit, var_names=["alpha", "beta_year", "sigma", "lambda"])
    plt.suptitle(f"Effective Sample Size (ESS) for {species_name}")

    # Plot autocorrelation plots
    az.plot_autocorr(fit, var_names=["alpha", "beta_year", "sigma", "lambda"])
    plt.suptitle(f"Autocorrelation Plots for {species_name}")

    # Check divergences using az.summary
    divergences = az.summary(fit, kind="diagnostics")
    print(divergences)

    plt.show()

    
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

species_fits = {}
for i, (species_name, data) in enumerate(species_data.items()):
    model = stan_models[species_name]

    # Fit the Stan model
    fit = model.sample(
        num_chains=4,
        num_samples=2000
    )
    species_fits[species_name] = fit

    # Extract posterior samples
    samples = fit['predicted_counts']

    # Plot occurrences per year
    axes[i].plot(data['year'], data['count'], label='Observed', marker='o', linestyle='-', color='blue')

    fut_pred = [data['count'][-1]]

    # Highlight predicted occurrences for the next 3 years
    for future_year in range(data['T'] + 1, data['T'] + 4):
        # Predict occurrences for the future year
        predicted_counts = samples[:, future_year - data['T'] - 1]
        mean_prediction = predicted_counts.mean()
        prediction_interval = (np.percentile(predicted_counts, 2.5), np.percentile(predicted_counts, 97.5))
        fut_pred.append(mean_prediction)

        # Print observed data statistics
        print(f"Observed Data - Mean: {np.mean(data['count'])}, Standard Deviation: {np.std(data['count'])}")
        print(f"Predicted Data for {future_year} - Mean: {mean_prediction}, Prediction Interval: {prediction_interval}")
        print("\n")

    # Plot occurrences and predictions
    axes[i].plot(range(data['T'], data['T'] + 4), fut_pred, marker='o', markersize=8, label='Predicted', color='green')
    axes[i].fill_between(range(data['T'], data['T'] + 4), prediction_interval[0], prediction_interval[1], color='green', alpha=0.3)
    axes[i].set_title(species_name)
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Occurrences')
    axes[i].legend()

plt.tight_layout()
plt.show()


for species_name, fit in species_fits.items():
    plot_diagnostics(fit, species_name)

def gelman_rubin_statistic(chains):
    n = len(chains)
    m = len(chains[0])

    # Calculate between-chain variance
    B = m / (n - 1) * np.sum((np.mean(chain) - np.mean(chains))**2 for chain in chains)

    # Calculate within-chain variance
    W = 1 / n * np.sum(np.var(chain) for chain in chains)

    # Calculate estimated variance
    Var_hat = (m - 1) / m * W + B / m

    # Calculate potential scale reduction factor
    Rhat = np.sqrt(Var_hat / W)

    return Rhat

parameters = ['alpha', 'beta_year', 'sigma', 'lambda']

for specie, fit in species_fits.items():
    chains = [fit[parameter] for parameter in parameters]
    print(f"Gelman-Rubin Statistic {specie} (Rhat): {gelman_rubin_statistic(chains)}")
