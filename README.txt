
Welcome to the KLDA Model Repository!

This repository contains code and resources to reproduce the experiments presented in our paper "Achieving Joint Training Accuracy in Continual Learning". Follow the instructions below to set up the environment.

## Dependencies

1. **Install PyTorch:**
   Make sure you have PyTorch installed. You can follow the installation guide from (https://pytorch.org/get-started/locally/).

2. **Install Required Packages:**
   After installing PyTorch, install the remaining dependencies using the provided `requirements.txt` file. Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

## Note on GPU Usage

It is recommended to use a GPU for faster computation. If a GPU is not available, the code will automatically switch to CPU, though it will run slower.

## Running the Model

To evaluate the model on one of the supported datasets, you can use the provided `evaluate.py` script. Below is an example of how to run the script:

```bash
python3 evaluate.py CLINC --D 5000 --sigma 1e-4 --num_ensembles 5 --seed 0 --model_name 'facebook/bart-base'
```

### Arguments

- **`dataset_name` (Positional)**: The name of the dataset to evaluate on. Available options are:
  - `CLINC`
  - `Banking`
  - `DBpedia`
  - `HWU`
  The argument is case-insensitive.

- **`--D` (Optional)**: The dimension of the transformed features using Random Fourier Features (RFF). Default is 5000.

- **`--sigma` (Optional)**: The bandwidth parameter for the Radial Basis Function (RBF) kernel. Default is `1e-4`.

- **`--num_ensembles` (Optional)**: The number of models used in the ensemble. A higher number of models can increase performance but may require more computation and memory. Default is 5.

- **`--seed` (Optional)**: The random seed used to ensure reproducibility in the model initialization. Default is 0.

- **`--model_name` (Optional)**: The name of the pre-trained model to use as the backbone for generating text embeddings. The default is `'facebook/bart-base'`. Other models available in the `sentence-transformers` library can also be specified here.

For each model, it is essential to use the appropriate sigma value, as the performance of the model can be sensitive to this parameter. Below are the sigma values we used for our experiments: Paraphrase-MiniLM: 1e-2, BART-base: 1e-4, BERT-base: 5e-3, RoBERTa-large: 5e-3, T5-3b: 5e-2, and Mistral-7B: 5e-6.

This command will train the classifier on the CLINC dataset and print the accuracy of the model on the test set.

Happy experimenting!
