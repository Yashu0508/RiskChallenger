"""Fixed-configuration regressors for predicting next-period variance."""

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from riskchallenger.ml.features import FEATURE_COLUMNS


class VarianceRegressor(Protocol):
    """Small common interface shared by baseline and ML estimators."""

    model_name: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> dict[str, float]: ...


class NaiveRollingVarianceRegressor:
    """Naive variance forecast from the trailing 20-return sample variance."""

    model_name = "naive_rolling_volatility"

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        if features.empty or target.empty or len(features) != len(target):
            raise ValueError("naive model requires aligned nonempty training data")
        self._fitted = True

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("naive volatility model must be fitted first")
        if "volatility_20d" not in features:
            raise ValueError("naive model requires the volatility_20d feature")
        volatility = features["volatility_20d"].to_numpy(dtype=float)
        if not np.isfinite(volatility).all():
            raise ValueError("naive volatility feature must be finite")
        return np.square(volatility)

    def feature_importance(self) -> dict[str, float]:
        return {"volatility_20d": 1.0}


class RidgeVarianceRegressor:
    """Ridge regression with training-only standardization in a Pipeline."""

    model_name = "ridge_variance"

    def __init__(self, alpha: float = 1.0) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and greater than zero")
        self.alpha = float(alpha)
        self.estimator = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._validate_training(features, target)
        self.estimator.fit(features.loc[:, list(FEATURE_COLUMNS)], target)
        self._fitted = True

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Ridge variance model must be fitted first")
        return np.asarray(
            self.estimator.predict(features.loc[:, list(FEATURE_COLUMNS)]), dtype=float
        )

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Ridge variance model must be fitted first")
        coefficients = self.estimator.named_steps["ridge"].coef_
        return dict(zip(FEATURE_COLUMNS, map(float, coefficients), strict=True))

    @staticmethod
    def _validate_training(features: pd.DataFrame, target: pd.Series) -> None:
        if features.empty or len(features) != len(target):
            raise ValueError(
                "training features and target must be nonempty and aligned"
            )
        if (
            not np.isfinite(features.to_numpy(dtype=float)).all()
            or not np.isfinite(target.to_numpy(dtype=float)).all()
        ):
            raise ValueError("training features and target must be finite")


class RandomForestVarianceRegressor(RidgeVarianceRegressor):
    """Small deterministic random forest; input scaling is not used."""

    model_name = "random_forest_variance"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        if n_estimators < 1 or min_samples_leaf < 1:
            raise ValueError("forest estimator and leaf counts must be positive")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be positive or None")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._validate_training(features, target)
        self.estimator.fit(features.loc[:, list(FEATURE_COLUMNS)], target)
        self._fitted = True

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Random Forest variance model must be fitted first")
        return np.asarray(
            self.estimator.predict(features.loc[:, list(FEATURE_COLUMNS)]), dtype=float
        )

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Random Forest variance model must be fitted first")
        return dict(
            zip(
                FEATURE_COLUMNS,
                map(float, self.estimator.feature_importances_),
                strict=True,
            )
        )


class XGBoostVarianceRegressor(RandomForestVarianceRegressor):
    """Compact deterministic XGBoost regressor with fixed conservative defaults."""

    model_name = "xgboost_variance"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if n_estimators < 1 or max_depth < 1:
            raise ValueError("XGBoost estimator count and max_depth must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("XGBoost learning_rate must be finite and positive")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self._fitted = False

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("XGBoost variance model must be fitted first")
        return dict(
            zip(
                FEATURE_COLUMNS,
                map(float, self.estimator.feature_importances_),
                strict=True,
            )
        )
