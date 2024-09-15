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
   Parameters for running the code, such as learning rate or model settings, can be adjusted for each individual dataset in the `main.py` file.

### Running Different Approaches

By default, only the FedMod approach will be executed when the program is initiated. If you want to run additional approaches, such as Homomorphic Encryption (HE), Functional Encryption (FE), Differential Privacy (DP), FedV, or Centralized, you will need to modify the `main.py` file.

1. **Uncomment the desired approaches**: 
   In the `main.py` file, locate the list called `all_results`. The code sections corresponding to the approaches you want to run (e.g., HE, FE, DP, FedV, Centralized) should be uncommented.

2. **Update the `name_list`**:
   Similarly, in the list called `name_list`, add the names of the approaches you want to run.

### Important Notes

- Some parameters, such as the learning rate, batch size, and number of epochs, can be specified individually for each dataset directly within the `main.py` file. Be sure to configure these parameters based on your dataset before running the code.
