import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the folder and file
#folder_path = 'rms_mu_changing_parameters'
#file_name = 'rms_mu_combined_31011-31091.csv'
#file_name_2 = 'rms_mu_combined_8630.csv'
#file_path = os.path.join(folder_path, file_name_2)

# Read the CSV file
#df = pd.read_csv(file_path)

# Calculate the number of columns
#number_of_rms_categories = len(df.columns)

# Print the number of 'rms' categories
#print(f"Total number of 'rms' categories (columns): {number_of_rms_categories}")

# Calculate the number of elements for each 'rms' category
#rms_counts = df.count()

# Print the counts in a readable format
#print("Counts of each 'rms' category (column):")
#for rms_category, count in rms_counts.items():
 #   print(f"{rms_category}: {count}")





def set_plot_template():
    plt.rcParams.update({
        'figure.figsize': (10, 6),        # Size of the figure
        'axes.titlesize': 35,             # Title font size
        'axes.labelsize': 30,             # X and Y label font size
        'xtick.labelsize': 25,            # X tick label font size
        'ytick.labelsize': 25,            # Y tick label font size
        'axes.titlepad': 25,              # Distance of the title from the plot
        'axes.labelpad': 20,              # Distance of the labels from the plot
        'legend.fontsize': 25,            # Legend font size
        'legend.frameon': True,           # Legend frame
        'legend.loc': 'best',             # Legend location
        'lines.linewidth': 2,             # Line width
        'grid.linestyle': '--',           # Grid line style
        'grid.alpha': 0.7,                # Grid line transparency
        'savefig.dpi': 300,               # Dots per inch for saved figures
        'savefig.bbox': 'tight',          # Bounding box for saved figures
        'savefig.format': 'png',          # Default save format
        'figure.autolayout': True         # Automatic layout adjustment
    })
# Function to get all files and group them by noise-alpha pairs
def get_files_by_noise_alpha(directory):
    files_by_noise_alpha = {}
    # Regular expression to match the filenames
    #pattern = re.compile(r"mux_muy_noise_(?P<noise>[\d\.e-]+)_alpha_(?P<alpha>[\d\.e-]+)(\.tfs)?\.npy")
    #pattern = re.compile(r"mux_muy_noise_(?P<noise>[\d\.e-]+)_alpha_(?P<alpha>[\d\.e-]+)(\.tfs)?(?:.try2)?(?:.try3)?(?:.try4)?(?:.try5)?\.npy")
    #pattern = re.compile(r"mux_muy_noise_(?P<noise>[\d\.e-]+)_alpha_(?P<alpha>[\d\.e-]+)(?:.compare)?(?:.compare1)?(?:.compare2)?(?:.compare4)?(?:.compare5)?(?:.compare6)?(?:.compare3)?\.npy")
    pattern = re.compile(r"mux_muy_b1_b2_noise_(?P<noise>[\d\.e-]+)_alpha_(?P<alpha>[\d\.e-]+)(?:.compare)?(?:.compare1)?(?:.compare2)?(?:.compare3)?(?:.compare4)?(?:.compare5)?(?:.compare6)?(?:.compare7)?(?:.compare8)?(?:.compare9)?\.npy")


    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            noise = match.group("noise")
            alpha = match.group("alpha")
            key = (noise, alpha)
            if key not in files_by_noise_alpha:
                files_by_noise_alpha[key] = []
            files_by_noise_alpha[key].append(os.path.join(directory, filename))

    return files_by_noise_alpha

# Function to calculate RMS
def calculate_rms(array):
    #array = array.astype(np.float64)  # Ensure the array is of type float64
    #return np.sqrt(np.mean(np.square(array), axis=1))
    return np.sqrt(np.mean(np.square(array)))

# Function to extract delta phi in x and in y for beam 1 and beam 2, calculate RMS, and return mean RMS values
def combine_extract_calculate_mean_rms(files, noise, alpha):
    data_arrays = [np.load(file, allow_pickle=True) for file in files]
    combined_data = np.concatenate(data_arrays, axis=0)  # Combine along the first axis
    print(f"Noise: {noise}, Alpha: {alpha}, Combined data shape: {combined_data.shape}")  # Print the shape of the combined data with noise and alpha
    #first_term = combined_data[:, 0, :]  # Extract first term of the second dimension
    #second_term = combined_data[:, 1, :]  # Extract second term of the second dimension
    delta_mu_x_b1 = combined_data[:, 0]
    delta_mu_x_b2 = combined_data[:, 1]
    delta_mu_y_b1 = combined_data[:, 2]
    delta_mu_y_b2 = combined_data[:, 3]
    
    
    #first_term_rms = calculate_rms(first_term)
    #second_term_rms = calculate_rms(second_term)
    delta_mu_x_b1_rms = calculate_rms(delta_mu_x_b1)
    delta_mu_x_b2_rms =  calculate_rms(delta_mu_x_b2)
    delta_mu_y_b1_rms = calculate_rms(delta_mu_y_b1)
    delta_mu_y_b2_rms = calculate_rms(delta_mu_y_b2)
    
    # Calculate mean RMS values
    #mean_first_term_rms = np.mean(first_term_rms)
    #mean_second_term_rms = np.mean(second_term_rms)
    mean_delta_mu_x_b1 = np.mean(delta_mu_x_b1_rms)
    mean_delta_mu_x_b2 = np.mean(delta_mu_x_b2_rms)
    mean_delta_mu_y_b1 = np.mean(delta_mu_y_b1_rms)
    mean_delta_mu_y_b2 = np.mean(delta_mu_y_b2_rms)
    
    
    #return mean_first_term_rms, mean_second_term_rms
    return mean_delta_mu_x_b1, mean_delta_mu_x_b2,  mean_delta_mu_y_b1, mean_delta_mu_y_b2

#directory = 'rms_mu_when_noisy_data'  # Specify the directory containing the files
#directory = 'compare_ip5'
directory = 'IP8_sbs'

# Get the files grouped by noise-alpha pairs
files_grouped = get_files_by_noise_alpha(directory)

# Prepare a list to store results
results = []

# Calculate mean RMS values for each noise-alpha pair
for key, files in files_grouped.items():
    noise = float(key[0])
    alpha = float(key[1])
    if noise >= 0.00 and alpha >= 0.00:
        #mean_first_term_rms, mean_second_term_rms = combine_extract_calculate_mean_rms(files, noise, alpha)
        mean_delta_mu_x_b1, mean_delta_mu_x_b2,  mean_delta_mu_y_b1, mean_delta_mu_y_b2 = combine_extract_calculate_mean_rms(files, noise, alpha)
        
        results.append({
            'noise': noise,
            'alpha': alpha,
            #'rms_mux_mean': mean_first_term_rms,
            #'rms_muy_mean': mean_second_term_rms
            'rms_mux_mean_b1': mean_delta_mu_x_b1,
            'rms_mux_mean_b2': mean_delta_mu_x_b2,
            'rms_muy_mean_b1': mean_delta_mu_y_b1,
            'rms_muy_mean_b2': mean_delta_mu_y_b2
        })

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Ensure alphas are sorted in increasing order
sorted_alphas = sorted(results_df['alpha'].unique())

# Pivot the data for heatmap plotting
pivot_table_mux_b1 = results_df.pivot(index='alpha', columns='noise', values='rms_mux_mean_b1').reindex(sorted_alphas) * 1000
pivot_table_muy_b1 = results_df.pivot(index='alpha', columns='noise', values='rms_muy_mean_b1').reindex(sorted_alphas) * 1000
pivot_table_mux_b2 = results_df.pivot(index='alpha', columns='noise', values='rms_mux_mean_b2').reindex(sorted_alphas) * 1000
pivot_table_muy_b2 = results_df.pivot(index='alpha', columns='noise', values='rms_muy_mean_b2').reindex(sorted_alphas) * 1000

# Plotting heatmaps
set_plot_template()

plt.figure(figsize=(24, 16))

# Heatmap for RMS Mux Mean Beam 1
plt.subplot(2, 2, 1)
sns.heatmap(pivot_table_mux_b1, annot=True, fmt=".4f", cmap='viridis', annot_kws={"size": 18})
plt.title(r'RMS $\Delta \phi_{x}$ $[10^{{-3}}]$')
plt.xlabel('Noise', fontsize=25)
plt.ylabel(r'$\alpha$', fontsize=25)
plt.text(-0.6,  0.8, r'Beam 1', ha='center', va='bottom', rotation='vertical', fontsize=30)
plt.gca().invert_yaxis()

# Heatmap for RMS Muy Mean Beam 1
plt.subplot(2, 2, 2)
sns.heatmap(pivot_table_muy_b1, annot=True, fmt=".4f", cmap='viridis', annot_kws={"size": 18})
plt.title(r'RMS $\Delta \phi_{y}$ $[10^{{-3}}]$')
plt.xlabel('Noise', fontsize=25)
plt.ylabel(r'$\alpha$', fontsize=25)
plt.gca().invert_yaxis()

# Heatmap for RMS Mux Mean Beam 2
plt.subplot(2, 2, 3)
sns.heatmap(pivot_table_mux_b2, annot=True, fmt=".4f", cmap='viridis', annot_kws={"size": 18})
#plt.title(r'RMS $\Delta \phi_{x}$ $[10^{{-3}}]$')
plt.xlabel('Noise', fontsize=25)
plt.ylabel(r'$\alpha$', fontsize=25)
plt.text(-0.6,  0.8, r'Beam 2', ha='center', va='bottom', rotation='vertical', fontsize=30)
plt.gca().invert_yaxis()

# Heatmap for RMS Muy Mean Beam 2
plt.subplot(2, 2, 4)
sns.heatmap(pivot_table_muy_b2, annot=True, fmt=".4f", cmap='viridis', annot_kws={"size": 18})
#plt.title(r'RMS $\Delta \phi_{y}$ Beam 2 $[10^{{-3}}]$')
plt.xlabel('Noise', fontsize=25)
plt.ylabel(r'$\alpha$', fontsize=25)
plt.gca().invert_yaxis()

# Adjust layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()
