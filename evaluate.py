import pandas as pd
import warnings
import torch
import argparse
from classifier import Classifier

warnings.simplefilter(action='ignore', category=FutureWarning)

# Dataset paths
DATASET_PATHS = {
    "CLINC": {'train': 'data/clinc_train.csv', 'test': 'data/clinc_test.csv'},
    "BANKING": {'train': 'data/banking_train.csv', 'test': 'data/banking_test.csv'},
    "DBPEDIA": {'train': 'data/dbpedia_train.csv', 'test': 'data/dbpedia_test.csv'},
    "HWU": {'train': 'data/hwu_train.csv', 'test': 'data/hwu_test.csv'},
}

def load_data_from_csv(train_filename, test_filename):
    """
    Loads training and testing data from CSV files.
    Args:
        train_filename (str): Path to the training data CSV file.
        test_filename (str): Path to the testing data CSV file.
    Returns:
        tuple: A tuple containing the training DataFrame and testing DataFrame.
    """
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)

    return train_df, test_df

def compute_accuracy(model, test_df):
    """
    Computes the accuracy for the model on the test dataset.
    Args:
        model (Classifier): The classifier model to evaluate.
        test_df (pd.DataFrame): DataFrame containing the test dataset.
    Returns:
        float: The accuracy of the model on the test dataset.
    """
    correct_predictions = 0
    total_predictions = test_df.shape[0]

    for _, row in test_df.iterrows():
        actual_label = row['label']
        text = str(row['text'])
        predicted_label = model.predict(text)

        if actual_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def main(args):
    """
    Main function to execute the evaluation of the classifier.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    dataset_name = args.dataset_name.upper()
    
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Available datasets are: {', '.join(DATASET_PATHS.keys())}")

    train_filename = DATASET_PATHS[dataset_name]['train']
    test_filename = DATASET_PATHS[dataset_name]['test']
    train_df, test_df = load_data_from_csv(train_filename, test_filename)

    model = Classifier(D=args.D, sigma=args.sigma, num_ensembles=args.num_ensembles, seed=args.seed, model_name=args.model_name)
    model.fit(train_df)

    accuracy = compute_accuracy(model, test_df)

    print(f"Accuracy on the {dataset_name} dataset: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help="Name of the dataset to use (CLINC, Banking, DBpedia, HWU).")
    parser.add_argument('--D', type=int, default=5000, help="Dimension of the transformed features using RFF.")
    parser.add_argument('--sigma', type=float, default=1e-4, help="Bandwidth parameter for the RBF kernel.")
    parser.add_argument('--num_ensembles', type=int, default=5, help="Number of ensemble models to use.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument('--model_name', type=str, default='facebook/bart-base', help="Name of the pre-trained model to use as the backbone.")

    args = parser.parse_args()
    main(args)
