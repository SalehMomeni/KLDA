import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from KLDA import KLDA_E

class Classifier:
    def __init__(self, D, sigma, num_ensembles, seed=0, model_name='facebook/bart-base'):
        """
        Args:
            D (int): Dimension of Random Fourier Features (RFF).
            sigma (float): Bandwidth parameter for the RBF kernel.
            num_ensembles (int): Number of models in the ensemble.
            seed (int, optional): Seed for reproducibility. Defaults to 0.
            model_name (str, optional): Pre-trained model name for SentenceTransformer. Defaults to 'facebook/bart-base'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = SentenceTransformer(model_name).to(self.device)
        self.backbone.tokenizer.pad_token = self.backbone.tokenizer.eos_token
        self.D = D
        self.sigma = sigma
        self.num_ensembles = num_ensembles
        self.seed = seed
        self.model = None
        self.labels = []

    def fit(self, df):
        """
        Trains the ensemble model using the provided dataframe.
        Args:
            df (pd.DataFrame): DataFrame containing 'text' and 'label' columns.
        """
        grouped = df.groupby('label')['text'].apply(list).reset_index()
        self.labels = grouped['label'].tolist()
        class_texts = grouped['text'].tolist()
        
        num_classes = len(self.labels)
        d = self.backbone.get_sentence_embedding_dimension()
        self.model = KLDA_E(num_classes, d, self.D, self.sigma, self.num_ensembles, self.seed, self.device)
        for label, texts in zip(self.labels, class_texts):
            text_embeddings = self.get_embeddings(texts)
            self.model.batch_update(text_embeddings, self.labels.index(label))
        self.model.fit()

    def get_embeddings(self, sentences):
        """
        Generates embeddings for the given sentences using the SentenceTransformer backbone.
        Args:
            sentences (list or str): List of sentences or a single sentence.
        Returns:
            torch.Tensor: Tensor of sentence embeddings.
        """
        embeddings = self.backbone.encode(sentences, convert_to_tensor=True, device=self.device)
        return embeddings

    def predict(self, sentence):
        """
        Predicts the label of a given sentence.
        Args:
            sentence (str): The input sentence to classify.
        Returns:
            str: The predicted label.
        """
        input_embedding = self.get_embeddings(sentence)
        idx = self.model.predict(input_embedding)
        return self.labels[idx]