from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskchallenger.backtesting.engine import BacktestConfig, run_backtest
from riskchallenger.backtesting.forecasters import ForecastTrainingData
from riskchallenger.ml.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RiskFeatureBuilder,
)
from riskchallenger.ml.forecaster import MLVaRForecaster, NaiveVolatilityForecaster
from riskchallenger.ml.models import (
    NaiveRollingVarianceRegressor,
    RandomForestVarianceRegressor,
    RidgeVarianceRegressor,
    XGBoostVarianceRegressor,
)
from riskchallenger.portfolio.engine import PortfolioCalculationResult
from riskchallenger.portfolio.portfolio import Portfolio
from riskchallenger.portfolio.statistics import PortfolioStatistics
from riskchallenger.risk.result import RiskUnit


def returns_series(values: np.ndarray) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=dates, name="portfolio_return")


def training_data(
    values: np.ndarray, *, forecast_offset: int = 0
) -> ForecastTrainingData:
    series = returns_series(values)
    forecast_date = series.index[-1] + pd.offsets.BDay(1 + forecast_offset)
    return ForecastTrainingData(
        losses=-values * 1_000_000,
        returns=values,
        portfolio_value=1_000_000,
        unit=RiskUnit.MONETARY,
        refit_number=0,
        dates=series.index,
        forecast_date=forecast_date,
    )


def test_feature_target_is_next_session_squared_return() -> None:
    values = np.arange(1, 41, dtype=float) / 1000
    frame = RiskFeatureBuilder().build_features(returns_series(values))
    assert frame.iloc[19][TARGET_COLUMN] == pytest.approx(values[20] ** 2)
    assert frame.iloc[-1][TARGET_COLUMN] != frame.iloc[-1][TARGET_COLUMN]
    assert (
        tuple(column for column in frame if column != TARGET_COLUMN) == FEATURE_COLUMNS
    )


def test_features_at_origin_are_causal_and_future_change_only_changes_target() -> None:
    values = np.linspace(-0.02, 0.02, 50)
    changed = values.copy()
    changed[31] = 0.40
    builder = RiskFeatureBuilder()
    first = builder.build_features(returns_series(values))
    second = builder.build_features(returns_series(changed))
    pd.testing.assert_series_equal(
        first.loc[first.index[30], list(FEATURE_COLUMNS)],
        second.loc[second.index[30], list(FEATURE_COLUMNS)],
    )
    assert first.iloc[30][TARGET_COLUMN] != second.iloc[30][TARGET_COLUMN]


def test_feature_validation_rejects_nonfinite_and_unordered_returns() -> None:
    builder = RiskFeatureBuilder()
    values = returns_series(np.full(25, 0.01))
    invalid = values.copy()
    invalid.iloc[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        builder.build_features(invalid)
    with pytest.raises(ValueError, match="chronological"):
        builder.build_features(values.iloc[::-1])


@pytest.mark.parametrize(
    "model",
    [
        NaiveRollingVarianceRegressor(),
        RidgeVarianceRegressor(),
        RandomForestVarianceRegressor(n_estimators=12),
        XGBoostVarianceRegressor(n_estimators=12),
    ],
)
def test_variance_regressors_fit_predict_finite_and_expose_importance(model) -> None:
    rng = np.random.default_rng(13)
    values = rng.normal(0.0002, 0.012, 240)
    frame = RiskFeatureBuilder().build_features(returns_series(values))
    train = frame.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    model.fit(train.loc[:, list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    prediction = model.predict(train.loc[:, list(FEATURE_COLUMNS)].tail(5))
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
    assert model.feature_importance()


def test_ridge_scaler_is_fitted_only_on_training_features() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(0.0, 0.01, 180)
    frame = RiskFeatureBuilder().build_features(returns_series(values))
    train = frame.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).iloc[:80]
    model = RidgeVarianceRegressor()
    model.fit(train.loc[:, list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    scaler = model.estimator.named_steps["scaler"]
    np.testing.assert_allclose(
        scaler.mean_, train.loc[:, list(FEATURE_COLUMNS)].mean().to_numpy()
    )


def test_forecaster_converts_variance_to_right_tail_monetary_loss() -> None:
    values = np.random.default_rng(19).normal(0.001, 0.02, 180)
    model = MLVaRForecaster(RidgeVarianceRegressor())
    data = training_data(values)
    model.fit(data)
    model.prepare_forecast(data)
    value = model.forecast_var(0.95)
    forecast = model.last_forecast
    assert forecast is not None
    expected = (
        -(forecast.mean_forecast + forecast.predicted_volatility * norm.ppf(0.05))
        * 1_000_000
    )
    assert value == pytest.approx(expected)
    assert forecast.predicted_volatility >= 0
    assert forecast.training_observations > 50


def test_forecaster_requires_dates_strictly_before_forecast() -> None:
    data = training_data(np.random.default_rng(2).normal(0, 0.01, 100))
    model = NaiveVolatilityForecaster()
    model.minimum_training_examples = 10
    model.fit(data)
    invalid = ForecastTrainingData(**{**data.__dict__, "forecast_date": data.dates[-1]})
    with pytest.raises(ValueError, match="strictly precede"):
        model.prepare_forecast(invalid)


def test_future_realized_observation_does_not_change_prior_ml_forecasts() -> None:
    from riskchallenger.portfolio.engine import calculate_portfolio

    values = np.random.default_rng(22).normal(0.0002, 0.01, 110)
    dates = pd.date_range("2022-01-01", periods=len(values), freq="B")
    returns = pd.DataFrame({"A": values}, index=dates)
    portfolio = Portfolio(["A"], {"A": 1.0}, 1_000_000)
    original = calculate_portfolio(portfolio, returns)
    changed_returns = returns.copy()
    changed_returns.iloc[-1, 0] = -0.15
    changed = calculate_portfolio(portfolio, changed_returns)
    config = BacktestConfig(minimum_training_observations=60, refit_frequency=10)
    original_forecasts = (
        run_backtest(
            original,
            [MLVaRForecaster(RidgeVarianceRegressor(), minimum_training_examples=10)],
            config,
        )
        .model_results["ridge_variance"]
        .forecasts
    )
    changed_forecasts = (
        run_backtest(
            changed,
            [MLVaRForecaster(RidgeVarianceRegressor(), minimum_training_examples=10)],
            config,
        )
        .model_results["ridge_variance"]
        .forecasts
    )
    np.testing.assert_allclose(
        original_forecasts["var_forecast"].iloc[:-1],
        changed_forecasts["var_forecast"].iloc[:-1],
        rtol=0,
        atol=0,
    )
    assert original_forecasts["var_forecast"].iloc[-1] == pytest.approx(
        changed_forecasts["var_forecast"].iloc[-1]
    )


class TrackingForecaster:
    model_name = "tracking"
    refit_frequency = 3

    def __init__(self) -> None:
        self.fit_lengths: list[int] = []
        self.prepared_lengths: list[int] = []
        self._forecast = 0.0

    def fit(self, training_data: ForecastTrainingData) -> None:
        self.fit_lengths.append(len(training_data.returns))

    def prepare_forecast(self, training_data: ForecastTrainingData) -> None:
        self.prepared_lengths.append(len(training_data.returns))
        self._forecast = 100.0

    def forecast_var(self, confidence_level: float) -> float:
        return self._forecast

    def forecast_metadata(self) -> dict[str, int]:
        return {"training_length": self.prepared_lengths[-1]}


def test_backtest_per_model_refit_and_daily_forecast_update() -> None:
    n = 15
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    returns = pd.Series(np.zeros(n), index=dates, name="portfolio_return")
    values = pd.Series(np.full(n, 1000.0), index=dates, name="portfolio_value")
    pnl = pd.Series(np.zeros(n), index=dates, name="pnl")
    losses = -pnl
    stats = PortfolioStatistics(0, 0, 0, 0, 0, 1000, 1000, 0)
    portfolio_result = PortfolioCalculationResult(returns, values, pnl, losses, stats)
    forecaster = TrackingForecaster()
    output = run_backtest(
        portfolio_result,
        [forecaster],
        BacktestConfig(minimum_training_observations=5, refit_frequency=9),
    ).model_results["tracking"]
    assert output.refit_frequency == 3
    assert len(forecaster.fit_lengths) == 4
    assert forecaster.prepared_lengths == list(range(5, 15))
    assert output.forecasts["training_length"].tolist() == list(range(5, 15))


def test_naive_baseline_uses_trailing_twenty_session_sample_volatility() -> None:
    values = np.linspace(-0.02, 0.02, 80)
    model = NaiveVolatilityForecaster()
    data = training_data(values)
    model.fit(data)
    model.prepare_forecast(data)
    forecast = model.forecast_var(0.95)
    expected_sigma = np.std(values[-20:], ddof=1)
    expected_mu = np.mean(values)
    assert model.last_forecast.predicted_volatility == pytest.approx(expected_sigma)
    assert forecast == pytest.approx(
        -(expected_mu + expected_sigma * norm.ppf(0.05)) * 1_000_000
    )


def test_forest_learns_higher_variance_in_persistent_high_volatility_regime() -> None:
    rng = np.random.default_rng(2027)
    scales = np.tile(np.repeat([0.004, 0.035], 150), 4)
    values = rng.normal(0.0, scales)
    frame = RiskFeatureBuilder().build_features(returns_series(values))
    labeled = frame.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    train = labeled.loc[labeled.index < labeled.index[750]]
    test = labeled.loc[labeled.index >= labeled.index[750]]
    model = RandomForestVarianceRegressor(n_estimators=80, random_state=2027)
    model.fit(train.loc[:, list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    predictions = model.predict(test.loc[:, list(FEATURE_COLUMNS)])
    high_volatility = test["volatility_20d"].to_numpy() > 0.02
    assert high_volatility.any() and (~high_volatility).any()
    assert predictions[high_volatility].mean() > predictions[~high_volatility].mean()


class NegativeVarianceRegressor:
    model_name = "negative_variance_test"

    def fit(self, features, target) -> None:
        return None

    def predict(self, features) -> np.ndarray:
        return np.full(len(features), -0.01)

    def feature_importance(self) -> dict[str, float]:
        return {}


def test_negative_predicted_variance_is_projected_and_audited() -> None:
    values = np.random.default_rng(42).normal(0.0, 0.01, 100)
    model = MLVaRForecaster(NegativeVarianceRegressor(), minimum_training_examples=10)
    data = training_data(values)
    model.fit(data)
    model.prepare_forecast(data)
    model.forecast_var(0.95)
    assert model.last_forecast.predicted_volatility == 0.0
    assert model.last_forecast.negative_variance_projected_to_zero is True


def test_ml_pipeline_integration_on_repository_returns() -> None:
    from riskchallenger.portfolio.engine import calculate_portfolio, load_return_dataset

    assets = ("HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS")
    portfolio = Portfolio(
        assets=assets,
        weights={assets[0]: 0.5, assets[1]: 0.3, assets[2]: 0.2},
        initial_value=1_000_000,
    )
    calculated = calculate_portfolio(portfolio, load_return_dataset())
    forecaster = NaiveVolatilityForecaster(refit_frequency=20)
    data = ForecastTrainingData(
        losses=calculated.loss.iloc[:-1].to_numpy(),
        returns=calculated.portfolio_returns.iloc[:-1].to_numpy(),
        portfolio_value=float(calculated.portfolio_value.iloc[-2]),
        unit=RiskUnit.MONETARY,
        refit_number=0,
        dates=pd.DatetimeIndex(calculated.portfolio_returns.index[:-1]),
        forecast_date=pd.Timestamp(calculated.portfolio_returns.index[-1]),
    )
    forecaster.fit(data)
    forecaster.prepare_forecast(data)
    assert np.isfinite(forecaster.forecast_var(0.95))
