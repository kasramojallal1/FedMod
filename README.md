# FedMod

## Reconstructing the Conda Environment and Running the Code

This guide explains how to reconstruct the Conda environment and run the code, including how to select datasets and adjust parameters.

### Method 1: Using `environment.yml`

1. **Create the environment**:
   conda env create -f environment.yml

2. **Activate the environment**:
   conda activate <your-environment-name>

### Method 2: Using `requirements.txt`

1. **Create an empty environment**:
   conda create --name <your-environment-name> python=<python-version>

2. **Install dependencies**:
   conda install --name <your-environment-name> --file requirements.txt

3. **Activate the environment**:
   conda activate <your-environment-name>

### Instructions to Run the Code

1. **Run the main program**:
   The code can be executed by running the `main.py` file. Use the following command to run it:
   
   python main.py

2. **Select the dataset**:
   In the `main.py` file, you can specify which dataset to use by setting the corresponding dataset flag to `True`. Ensure only the dataset you want to run is set to `True`.

3. **Modify parameters**:
   Parameters for running the code, such as training settings or model configurations, can be adjusted in the `config.py` file. Open `config.py` and modify the parameters as needed before running the code.
