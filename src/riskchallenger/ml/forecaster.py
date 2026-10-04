"""ML model adapters that convert predicted variance into monetary VaR."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskchallenger.backtesting.forecasters import ForecastTrainingData
from riskchallenger.backtesting.validation import validate_count
from riskchallenger.ml.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RiskFeatureBuilder,
)
from riskchallenger.ml.models import (
    NaiveRollingVarianceRegressor,
    VarianceRegressor,
)
from riskchallenger.risk.validation import validate_confidence_level


@dataclass(frozen=True)
class MLRiskForecast:
    """Audit record for one ML-derived conditional Gaussian VaR forecast."""

    model_name: str
    forecast_date: pd.Timestamp
    predicted_volatility: float
    mean_forecast: float
    var_forecast: float
    portfolio_value: float
    training_observations: int
    negative_variance_projected_to_zero: bool

    def to_backtest_metadata(self) -> dict[str, object]:
        return {
            "predicted_volatility": self.predicted_volatility,
            "mean_forecast": self.mean_forecast,
            "portfolio_value": self.portfolio_value,
            "training_observations": self.training_observations,
            "negative_variance_projected_to_zero": (
                self.negative_variance_projected_to_zero
            ),
        }


class MLVaRForecaster:
    """Train a variance regressor and provide daily features between refits."""

    def __init__(
        self,
        model: VarianceRegressor,
        *,
        feature_builder: RiskFeatureBuilder | None = None,
        minimum_training_examples: int = 50,
        refit_frequency: int = 20,
    ) -> None:
        minimum_training_examples = validate_count(
            minimum_training_examples, "minimum_training_examples", minimum=2
        )
        refit_frequency = validate_count(refit_frequency, "refit_frequency")
        self.model = model
        self.model_name = model.model_name
        self.feature_builder = feature_builder or RiskFeatureBuilder()
        self.minimum_training_examples = minimum_training_examples
        self.refit_frequency = refit_frequency
        self._fitted = False
        self._current_features: pd.DataFrame | None = None
        self._current_date: pd.Timestamp | None = None
        self._current_portfolio_value: float | None = None
        self._current_mean: float | None = None
        self._training_observations = 0
        self.last_forecast: MLRiskForecast | None = None

    def fit(self, training_data: ForecastTrainingData) -> None:
        returns = self._returns_series(training_data)
        supervised = self.feature_builder.build_features(returns)
        training = supervised.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
        if len(training) < self.minimum_training_examples:
            raise ValueError(
                "insufficient labeled observations for ML fit: "
                f"need {self.minimum_training_examples}, received {len(training)}"
            )
        x_train = training.loc[:, list(FEATURE_COLUMNS)]
        y_train = training[TARGET_COLUMN]
        if (
            not np.isfinite(x_train.to_numpy(dtype=float)).all()
            or not np.isfinite(y_train.to_numpy(dtype=float)).all()
        ):
            raise ValueError("ML training data contain non-finite values")
        self.model.fit(x_train, y_train)
        self._training_observations = len(training)
        self._fitted = True

    def prepare_forecast(self, training_data: ForecastTrainingData) -> None:
        """Refresh current features/value/mean without retraining parameters."""
        if not self._fitted:
            raise RuntimeError("ML model must be fitted before forecast preparation")
        returns = self._returns_series(training_data)
        self._current_features = self.feature_builder.build_latest_features(returns)
        self._current_date = training_data.forecast_date
        self._current_portfolio_value = float(training_data.portfolio_value)
        self._current_mean = float(np.mean(returns.to_numpy(dtype=float)))

    def forecast_var(self, confidence_level: float) -> float:
        alpha = validate_confidence_level(confidence_level)
        if (
            not self._fitted
            or self._current_features is None
            or self._current_date is None
            or self._current_portfolio_value is None
            or self._current_mean is None
        ):
            raise RuntimeError("ML VaR forecaster needs fit/forecast preparation first")
        predicted_variance = float(self.model.predict(self._current_features)[0])
        if not np.isfinite(predicted_variance):
            raise ValueError("ML model predicted non-finite conditional variance")
        projected = predicted_variance < 0.0
        if projected:
            predicted_variance = 0.0
        predicted_volatility = float(np.sqrt(predicted_variance))
        if not np.isfinite(predicted_volatility) or predicted_volatility < 0:
            raise ValueError("predicted volatility must be finite and non-negative")
        lower_tail_z = float(norm.ppf(1.0 - alpha))
        var_forecast = (
            -(self._current_mean + predicted_volatility * lower_tail_z)
            * self._current_portfolio_value
        )
        if not np.isfinite(var_forecast):
            raise ValueError("conditional Gaussian VaR forecast is non-finite")
        self.last_forecast = MLRiskForecast(
            model_name=self.model_name,
            forecast_date=self._current_date,
            predicted_volatility=predicted_volatility,
            mean_forecast=self._current_mean,
            var_forecast=float(var_forecast),
            portfolio_value=self._current_portfolio_value,
            training_observations=self._training_observations,
            negative_variance_projected_to_zero=projected,
        )
        return float(var_forecast)

    def forecast_metadata(self) -> Mapping[str, object]:
        """Return details from the immediately preceding VaR calculation."""
        if self.last_forecast is None:
            raise RuntimeError("no ML forecast metadata is available yet")
        return self.last_forecast.to_backtest_metadata()

    def feature_importance(self) -> dict[str, float]:
        """Expose model coefficients/importances using the common feature names."""
        return self.model.feature_importance()

    def _returns_series(self, training_data: ForecastTrainingData) -> pd.Series:
        if training_data.dates is None:
            raise ValueError("ML forecaster requires dates in forecast training data")
        if len(training_data.dates) != len(training_data.returns):
            raise ValueError("training dates and returns must have equal length")
        if training_data.forecast_date is None:
            raise ValueError("ML forecaster requires a forecast date")
        dates = pd.DatetimeIndex(training_data.dates)
        if len(dates) == 0 or dates[-1] >= training_data.forecast_date:
            raise ValueError(
                "training observations must strictly precede forecast date"
            )
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("training dates must be unique and chronological")
        return self.feature_builder.validate_returns(
            pd.Series(training_data.returns, index=dates, name="portfolio_return")
        )


class NaiveVolatilityForecaster(MLVaRForecaster):
    """Phase 7 naive 20-session rolling volatility baseline adapter."""

    def __init__(self, *, refit_frequency: int = 20) -> None:
        super().__init__(
            NaiveRollingVarianceRegressor(), refit_frequency=refit_frequency
        )
