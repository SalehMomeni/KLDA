import torch
from collections import defaultdict

class KLDA:
    def __init__(self, num_classes, d, D, sigma, seed, device):
        """
        Args:
            num_classes (int): Total number of classes.
            d (int): Input feature dimension.
            D (int): RFF dimension.
            sigma (float): Bandwidth parameter for RBF kernel.
            seed (int): Seed for RFF initialization to ensure different transformations.
            device (torch.device): Device to run the models on.
        """
        self.num_classes = num_classes
        self.d = d
        self.D = D
        self.sigma = sigma
        self.seed = seed
        self.device = device

        torch.manual_seed(self.seed)
        self.omega = torch.normal(
            0, 
            torch.sqrt(torch.tensor(2 * self.sigma, dtype=torch.float32)),
            (self.d, self.D)
        ).to(self.device)
        self.b = (torch.rand(self.D) * 2 * torch.pi).to(self.device)

        self.class_means = defaultdict(lambda: torch.zeros(self.D, device=self.device))
        self.class_counts = defaultdict(int)

        self.sigma = torch.zeros((self.D, self.D), device=self.device)
        self.sigma_inv = None
        self.class_mean_matrix = None

    def _compute_rff(self, X):
        """
        Computes the Random Fourier Features for input data X.
        Args:
            X (torch.Tensor): Input feature tensor of shape (n_samples, d).
        Returns:
            torch.Tensor: Transformed feature tensor of shape (n_samples, D).
        """
        scaling_factor = torch.sqrt(torch.tensor(2.0 / self.D, dtype=torch.float32, device=self.device))
        return scaling_factor * torch.cos(X @ self.omega + self.b)

    def batch_update(self, X, y):
        """
        Updates the model with a batch of data for a specific class.
        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, d).
            y (int): Class label.
        """
        X = X.to(self.device)
        n = X.size(0)
        phi_X = self._compute_rff(X)  # Shape: (n, D)
        phi_X_mean = torch.mean(phi_X, dim=0)

        # Update class mean
        previous_count = self.class_counts[y]
        self.class_counts[y] += n
        self.class_means[y] = (self.class_means[y] * previous_count + phi_X_mean * n) / self.class_counts[y]

        # Update covariance matrix sigma
        centered_phi_X = phi_X - self.class_means[y]
        self.sigma += centered_phi_X.t() @ centered_phi_X

    def fit(self):
        """
        Finalizes the model by computing the inverse of the covariance matrix
        and stacking class means into a matrix for efficient distance computation.
        """
        self.sigma_inv = torch.pinverse(self.sigma)
        self.class_mean_matrix = torch.stack([self.class_means[i] for i in range(self.num_classes)]).to(self.device)

    def get_logits(self, x):
        """
        Computes the logits for all the classes.
        Args:
            x (torch.Tensor): Feature tensor of shape (d,).
        Returns:
            torch.Tensor: Logits for each class of shape (num_classes,).
        """
        x = x.to(self.device)
        phi_x = self._compute_rff(x.unsqueeze(0))  # Shape: (1, D)
        diff = self.class_mean_matrix - phi_x      # Shape: (num_classes, D)
        # Note:
        # Mahalanobis distance is used here instead of the original LDA because it provides a more intuitive
        # measure of distance. Under reasonable assumptions, Mahalanobis distance can be proven to be equivalent to LDA.
        logits = -torch.sum((diff @ self.sigma_inv) * diff, dim=1)  # Shape: (num_classes,)
        return logits

class KLDA_E:
    def __init__(self, num_classes, d, D, sigma, num_ensembles, seed, device=None):
        """
        Args:
            num_classes (int): Total number of classes.
            d (int): Input feature dimension.
            D (int): RFF dimension.
            sigma (float): Bandwidth parameter for RBF kernel.
            num_ensembles (int): Number of models in the ensemble.
            seed (int): Base seed for reproducibility.
            device (torch.device, optional): Device to run the models on. Defaults to CUDA if available.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ensembles = num_ensembles
        self.device = device
        self.models = [
            KLDA(num_classes, d, D, sigma, seed=seed+i, device=self.device) for i in range(self.num_ensembles)
        ]

    def batch_update(self, X, y):
        """
        Updates all models in the ensemble with the given batch of data.
        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, d).
            y (int): Class label.
        """
        for model in self.models:
            model.batch_update(X, y)

    def fit(self):
        """
        Finalizes all models in the ensemble by computing necessary matrices.
        """
        for model in self.models:
            model.fit()

    def predict(self, x):
        """
        Predicts the class label for a single instance by aggregating normalized probabilities from all ensemble members.
        Args:
            x (torch.Tensor): Feature tensor of shape (d,).
        Returns:
            int: Predicted class label.
        """
        total_probabilities = torch.zeros(self.models[0].num_classes, device=self.device)

        for model in self.models:
            logits = model.get_logits(x)
            probs = torch.softmax(logits, dim=0)
            total_probabilities += probs

        predicted_class = torch.argmax(total_probabilities).item()
        return predicted_class