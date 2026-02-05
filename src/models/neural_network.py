"""
Neural Network Model
====================

PyTorch-based deep learning model for flight price prediction.
"""

import numpy as np
from typing import Tuple, Optional, List
from .base import BasePriceModel

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PriceNet(nn.Module):
        """Deep neural network for price prediction."""

        def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                     dropout: float = 0.2):
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [128, 64, 32]

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)


class NeuralNetworkModel(BasePriceModel):
    """
    Neural network model for flight price prediction.

    Features:
    - Multi-layer perceptron architecture
    - Batch normalization and dropout for regularization
    - Early stopping to prevent overfitting
    - Monte Carlo dropout for uncertainty estimation
    """

    def __init__(self, hidden_dims: List[int] = None,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 dropout: float = 0.2):
        super().__init__("NeuralNetworkModel")

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for NeuralNetworkModel")

        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout

        self.model: Optional[PriceNet] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._input_dim: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.1) -> "NeuralNetworkModel":
        """
        Train the neural network.

        Args:
            X: Feature matrix
            y: Target prices
            validation_split: Fraction for validation

        Returns:
            Self for method chaining
        """
        self._input_dim = X.shape[1]

        # Normalize targets
        self._mean = y.mean()
        self._std = y.std()
        y_normalized = (y - self._mean) / (self._std + 1e-8)

        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_normalized[train_idx], y_normalized[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Initialize model
        self.model = PriceNet(
            input_dim=self._input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val).to(self.device)
                val_y = torch.FloatTensor(y_val).to(self.device)
                val_pred = self.model(val_X)
                val_loss = criterion(val_pred, val_y).item()

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict flight prices."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        # Denormalize
        return predictions * self._std + self._mean

    def predict_with_confidence(self, X: np.ndarray,
                                n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty using Monte Carlo dropout.

        Runs multiple forward passes with dropout enabled to estimate
        prediction uncertainty.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Enable dropout for MC sampling
        self.model.train()

        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_tensor).cpu().numpy()
                predictions_list.append(pred)

        predictions = np.array(predictions_list)

        # Denormalize
        predictions = predictions * self._std + self._mean

        # Mean prediction and uncertainty
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        # Convert std to confidence (lower std = higher confidence)
        max_expected_std = self._std * 0.5  # Expected maximum uncertainty
        confidence = 1.0 - np.clip(std_pred / max_expected_std, 0, 1)

        return mean_pred, confidence


# Fallback if PyTorch not available
if not HAS_TORCH:
    class NeuralNetworkModel(BasePriceModel):
        """Fallback when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for NeuralNetworkModel. "
                "Install with: pip install torch"
            )

        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

        def predict_with_confidence(self, X):
            pass
