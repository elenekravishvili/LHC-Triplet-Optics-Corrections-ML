# Workflow for Machine Learning-Based Optics Correction

### Step 1: Data Generation
To generate the initial dataset, run the following script:
`generate_data.py`
This script internally calls `madx_jobs.py` to simulate and generate the required data. The generated data will be saved in the specified directories, ready for further processing.

### Step 2: Model Training
Once the data is generated, the next step is to train the machine learning models. Use the following script:
`model_training.py`
This script will create and train the estimators needed for predicting errors.

### Step 3: Reconstruct Twiss Parameters
After training the model, you can reconstruct the Twiss parameters based on the predicted values. There are two options:
- **Without offset**: Run the script:
    ```
    python reconstruct_twiss.py
    ```
- **With offset**: Run the script:
    ```
    python recons_twiss.py
    ```
These scripts will generate the reconstructed Twiss parameters and save the results accordingly.

### Step 4: Data Analysis and Results Visualization
To visualize the results from the previous steps, you can use the plotting and analysis functions located in `data_analysis_plot.py`.
This script generates various visualizations that will help assess the performance and results of the optics correction process.

## SBS Validation Workflow
The following steps outline the process for validating the machine learning predictions using the Segment-by-Segment (SBS) method.

### Step 1: Generate SBS Data
To begin, generate the SBS data using the script: `sbs_generate.py`
This script will run `madx_job_sbs.py`, utilizing the predicted values from the estimators created in the earlier model training step. The phase advances for both **B1** and **B2** beams (in both **x** and **y** directions) will be saved.

### Step 2: Calculate and Plot RMS Values
Next, read the SBS-generated data and calculate the RMS values. Use the following script to calculate and plot these values:
`sbs_plot.py`

### Step 3: Additional Data Analysis
To further analyze the SBS validation data and plot specific results, you can call relevant functions from within the `data_analysis_plot.py` script.
This script will help visualize the results and perform additional checks to ensure the model's accuracy.
