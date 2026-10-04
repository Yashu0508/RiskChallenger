"""Streamlit AppTest checks for dashboard views and portfolio interaction."""

from streamlit.testing.v1 import AppTest

APP_PATH = "dashboard/app.py"


def test_dashboard_views_render_and_expose_complete_asset_universe() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=90)
    assert not app.exception
    assert len(app.multiselect[0].options) == 50
    assert app.multiselect[0].value == ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
    assert [item.value for item in app.title] == [
        "Quantitative Risk Management Terminal"
    ]
    default_weights = {
        item.key: item.value
        for item in app.number_input
        if item.key and item.key.startswith("portfolio_weight_")
    }
    assert default_weights == {
        "portfolio_weight_HDFCBANK.NS": 0.5,
        "portfolio_weight_ICICIBANK.NS": 0.3,
        "portfolio_weight_RELIANCE.NS": 0.2,
    }

    for page in (
        "Portfolio",
        "Risk",
        "Stress Testing",
        "Backtesting",
        "Model Diagnostics",
        "Overview",
    ):
        app.radio[0].set_value(page).run(timeout=90)
        assert not app.exception, (
            f"{page} view raised: {[e.message for e in app.exception]}"
        )


def test_alternative_portfolio_does_not_trigger_or_mislabel_reference_backtest() -> (
    None
):
    app = AppTest.from_file(APP_PATH).run(timeout=90)
    app.multiselect[0].set_value(["TCS.NS", "INFY.NS"]).run(timeout=90)
    assert not app.exception
    assert any("sum to 100%" in error.value for error in app.error)

    app.button[0].click().run(timeout=90)
    assert not app.exception
    selected_weights = {
        item.key: item.value
        for item in app.number_input
        if item.key and item.key.startswith("portfolio_weight_")
    }
    assert selected_weights["portfolio_weight_TCS.NS"] == 0.5
    assert selected_weights["portfolio_weight_INFY.NS"] == 0.5

    app.radio[0].set_value("Backtesting").run(timeout=90)
    assert not app.exception
    assert any(
        "available for the stored reference portfolio" in item.value
        for item in app.info
    )
