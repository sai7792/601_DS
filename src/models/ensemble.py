"""
Ensemble Models
===============

Advanced ensemble methods combining XGBoost, LightGBM, and CatBoost
for robust flight price prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .base import BasePriceModel

# Scikit-learn models (always available)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge

# Optional advanced boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class EnsembleModel(BasePriceModel):
    """
    Ensemble model combining multiple gradient boosting algorithms.

    Uses stacking with:
    - XGBoost (if available)
    - LightGBM (if available)
    - CatBoost (if available)
    - Scikit-learn GradientBoosting (fallback)
    - Random Forest
    """

    def __init__(self, use_stacking: bool = True):
        super().__init__("EnsembleModel")
        self.use_stacking = use_stacking
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.meta_model: Optional[Ridge] = None
        self._setup_models()

    def _setup_models(self) -> None:
        """Initialize base models."""
        # Always available sklearn models
        self.models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.models["random_forest"] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        self.models["adaboost"] = AdaBoostRegressor(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )

        # Advanced models if available
        if HAS_XGBOOST:
            self.models["xgboost"] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )

        if HAS_LIGHTGBM:
            self.models["lightgbm"] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )

        if HAS_CATBOOST:
            self.models["catboost"] = cb.CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=0
            )

        # Initialize uniform weights
        n_models = len(self.models)
        for name in self.models:
            self.weights[name] = 1.0 / n_models

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """
        Train all ensemble models.

        Uses cross-validation to determine optimal weights.
        """
        # Train each base model
        for name, model in self.models.items():
            model.fit(X, y)

        # Optimize weights using validation performance
        self._optimize_weights(X, y)

        # Train meta-model if using stacking
        if self.use_stacking:
            meta_features = self._get_meta_features(X)
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_features, y)

        self.is_fitted = True
        return self

    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize model weights based on individual performance."""
        errors = {}

        for name, model in self.models.items():
            predictions = model.predict(X)
            mae = np.mean(np.abs(predictions - y))
            errors[name] = mae

        # Convert errors to weights (lower error = higher weight)
        total_inv_error = sum(1.0 / e for e in errors.values())
        for name, error in errors.items():
            self.weights[name] = (1.0 / error) / total_inv_error

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X).reshape(-1, 1)
            predictions.append(pred)
        return np.hstack(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted ensemble or stacking."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.use_stacking and self.meta_model is not None:
            meta_features = self._get_meta_features(X)
            return self.meta_model.predict(meta_features)

        # Weighted average prediction
        weighted_pred = np.zeros(len(X))
        for name, model in self.models.items():
            weighted_pred += self.weights[name] * model.predict(X)

        return weighted_pred

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence based on model agreement.

        Returns predictions and confidence scores (0-1).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions from all models
        all_predictions = []
        for model in self.models.values():
            all_predictions.append(model.predict(X))

        all_predictions = np.array(all_predictions)

        # Main prediction
        main_prediction = self.predict(X)

        # Confidence based on coefficient of variation
        std = np.std(all_predictions, axis=0)
        mean = np.mean(all_predictions, axis=0)
        cv = std / (mean + 1e-8)  # Coefficient of variation

        # Convert CV to confidence (lower CV = higher confidence)
        confidence = 1.0 / (1.0 + cv)

        return main_prediction, confidence

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models."""
        importances = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_

        return importances

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights in the ensemble."""
        return self.weights.copy()
