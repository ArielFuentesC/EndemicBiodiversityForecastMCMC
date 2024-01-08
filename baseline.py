import os
import stan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Function to predict occurrences for a future year
    
def predict_occurrences(species_name, future_year):
    model = stan_models[species_name]
    data = species_data[species_name]

    # Fit the Stan model with existing data and provide additional data for prediction
    fit = model.sample(
        num_chains=4,
        num_samples=2000
    )

    # Extract posterior samples
    samples = fit['predicted_counts']

    # Predict occurrences for the future year
    predicted_counts = samples[:, future_year - 1]

    # Return mean and uncertainty (e.g., 95% credible interval) of predictions
    return predicted_counts.mean(), (np.percentile(predicted_counts, 2.5), np.percentile(predicted_counts, 97.5))

# Plot occurrences per year for each species
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Flatten the axes array for easier indexing
axes = axes.flatten()

for i, (species_name, data) in enumerate(species_data.items()):
    # Plot occurrences per year
    axes[i].plot(data['year'], data['count'], label='Observed', marker='o', linestyle='-', color='blue')
    print(f"Specie:{species_name}\n{data}")
    print(f"Statistics for {species_name}:")
    fut_pred = [data['count'][-1]]

    # Highlight predicted occurrences for the next 3 years
    for future_year in range(data['T'] + 1, data['T'] + 4):
        mean_prediction, prediction_interval = predict_occurrences(species_name, future_year)
        fut_pred.append(mean_prediction)

        # Print observed data statistics
        print(f"Observed Data - Mean: {np.mean(data['count'])}, Standard Deviation: {np.std(data['count'])}")
        print(f"Predicted Data for {future_year} - Mean: {mean_prediction}, Prediction Interval: {prediction_interval}")
        print("\n")
    print(f"ranges:{range(data['T'], data['T'] + 4)}, fut:{fut_pred}, T:{data['T']}")
    axes[i].plot(range(data['T'], data['T'] + 4), fut_pred, marker='o', markersize=8, label=f'Predicted {future_year}', color='green')

    axes[i].set_title(species_name)
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Occurrences')
    axes[i].legend()


# Remove empty subplots if there are fewer than 9 species
for j in range(len(species_files), 9):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()