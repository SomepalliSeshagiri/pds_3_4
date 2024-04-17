import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Set a seed for reproducibility
np.random.seed(42)

# a) Random sample of 25 observations
sample = data.sample(n=25)

# Calculate mean and highest Glucose values for the sample
sample_mean_glucose = sample['Glucose'].mean()
sample_highest_glucose = sample['Glucose'].max()

# Population statistics
population_mean_glucose = data['Glucose'].mean()
population_highest_glucose = data['Glucose'].max()

# Create comparison chart for mean Glucose
plt.bar(['Sample', 'Population'], [sample_mean_glucose, population_mean_glucose])
plt.title('Mean Glucose Comparison')
plt.ylabel('Glucose')
plt.savefig('mean_glucose_comparison.png')  # Save the plot as an image
plt.close()  # Close the plot to release memory

# Create comparison chart for highest Glucose
plt.bar(['Sample', 'Population'], [sample_highest_glucose, population_highest_glucose])
plt.title('Highest Glucose Comparison')
plt.ylabel('Glucose')
plt.savefig('highest_glucose_comparison.png')  # Save the plot as an image
plt.close()  # Close the plot to release memory

# b) Find the 98th percentile of BMI
sample_bmi_98th_percentile = np.percentile(sample['BMI'], 98)
population_bmi_98th_percentile = np.percentile(data['BMI'], 98)

# Create comparison chart for 98th percentile of BMI
plt.bar(['Sample', 'Population'], [sample_bmi_98th_percentile, population_bmi_98th_percentile])
plt.title('98th Percentile of BMI Comparison')
plt.ylabel('BMI')
plt.savefig('bmi_98th_percentile_comparison.png')  # Save the plot as an image
plt.close()  # Close the plot to release memory

# c) Bootstrap sampling
n_samples = 500
sample_size = 150
bootstrap_means = []
bootstrap_stds = []
bootstrap_percentiles = []

for _ in range(n_samples):
    bootstrap_sample = np.random.choice(data['BloodPressure'], size=sample_size, replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))
    bootstrap_stds.append(np.std(bootstrap_sample))
    bootstrap_percentiles.append(np.percentile(bootstrap_sample, 98))

# Calculate mean, standard deviation, and 98th percentile for BloodPressure in the population
population_bp_mean = data['BloodPressure'].mean()
population_bp_std = data['BloodPressure'].std()
population_bp_98th_percentile = np.percentile(data['BloodPressure'], 98)

# Create comparison charts
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(bootstrap_means, bins=30, alpha=0.5, label='Bootstrap Samples')
plt.axvline(population_bp_mean, color='red', linestyle='dashed', linewidth=1, label='Population Mean')
plt.title('Mean Blood Pressure Comparison')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(bootstrap_stds, bins=30, alpha=0.5, label='Bootstrap Samples')
plt.axvline(population_bp_std, color='red', linestyle='dashed', linewidth=1, label='Population Std')
plt.title('Standard Deviation Comparison')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(bootstrap_percentiles, bins=30, alpha=0.5, label='Bootstrap Samples')
plt.axvline(population_bp_98th_percentile, color='red', linestyle='dashed', linewidth=1, label='Population 98th percentile')
plt.title('98th Percentile Comparison')
plt.legend()

plt.tight_layout()
plt.savefig('blood_pressure_comparison.png')  # Save the plot as an image
plt.close()  # Close the plot to release memory

# Report findings
print("Population Mean Glucose:", population_mean_glucose)
print("Sample Mean Glucose:", sample_mean_glucose)
print("Population Highest Glucose:", population_highest_glucose)
print("Sample Highest Glucose:", sample_highest_glucose)
print("Population 98th Percentile of BMI:", population_bmi_98th_percentile)
print("Sample 98th Percentile of BMI:", sample_bmi_98th_percentile)
print("Population Mean Blood Pressure:", population_bp_mean)
print("Bootstrap Mean Blood Pressure:", np.mean(bootstrap_means))
print("Population Standard Deviation of Blood Pressure:", population_bp_std)
print("Bootstrap Standard Deviation of Blood Pressure:", np.mean(bootstrap_stds))
print("Population 98th Percentile of Blood Pressure:", population_bp_98th_percentile)
print("Bootstrap 98th Percentile of Blood Pressure:", np.mean(bootstrap_percentiles))
