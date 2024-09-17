# Workflow for Machine Learning-Based Optics Correction

### Step 1: Data Generation
To generate the initial dataset, run the following script:
generate_data.py
This script internally calls `madx_jobs.py` to simulate and generate the required data. The generated data will be saved in the specified directories, ready for further processing.

### Step 2: Model Training
Once the data is generated, the next step is to train the machine learning models. Use the following script:
model_training.py
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



