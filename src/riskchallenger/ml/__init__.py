"""Machine-learning volatility challengers for portfolio VaR."""

from riskchallenger.ml.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RiskFeatureBuilder,
)
from riskchallenger.ml.forecaster import (
    MLRiskForecast,
    MLVaRForecaster,
    NaiveVolatilityForecaster,
)
from riskchallenger.ml.models import (
    RandomForestVarianceRegressor,
    RidgeVarianceRegressor,
    XGBoostVarianceRegressor,
)

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "MLRiskForecast",
    "MLVaRForecaster",
    "NaiveVolatilityForecaster",
    "RandomForestVarianceRegressor",
    "RidgeVarianceRegressor",
    "RiskFeatureBuilder",
    "XGBoostVarianceRegressor",
]
