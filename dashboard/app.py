"""Local quantitative risk and trading analytics terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data_access import (
    DEFAULT_ASSETS,
    DEFAULT_WEIGHTS,
    DEMO_VALUE,
    build_portfolio_result,
    build_stress_results,
    calculate_risk_results,
    create_portfolio,
    discover_asset_universe,
    format_risk_results,
    format_stress_results,
    is_reference_portfolio,
    load_clean_prices,
    load_phase4_snapshot,
    load_phase7_snapshot,
    overview_metrics,
    validate_dashboard_weights,
)

PAGES = (
    "Overview",
    "Portfolio",
    "Risk",
    "Stress Testing",
    "Backtesting",
    "Model Diagnostics",
)
DEMO_ASSET_SHOCKS = {
    "HDFCBANK.NS": -0.10,
    "ICICIBANK.NS": -0.15,
    "RELIANCE.NS": -0.20,
}

st.set_page_config(
    page_title="RiskChallenger | Risk Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --terminal-bg: #0b0f14;
        --terminal-panel: #111821;
        --terminal-panel-2: #151e28;
        --terminal-border: #263341;
        --terminal-muted: #8e9baa;
        --terminal-text: #e6edf3;
        --terminal-accent: #68a9ff;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--terminal-bg);
        color: var(--terminal-text);
    }
    [data-testid="stSidebar"] {
        background: #0e141b;
        border-right: 1px solid var(--terminal-border);
    }
    .block-container {max-width: 1580px; padding-top: 1.35rem; padding-bottom: 2rem;}
    h1, h2, h3 {color: var(--terminal-text); letter-spacing: -0.025em;}
    h1 {font-size: 1.8rem;}
    h2 {font-size: 1.25rem;}
    h3 {font-size: 1.02rem;}
    p, label, [data-testid="stCaptionContainer"] {color: #aab6c2;}
    [data-testid="stMetric"] {
        background: var(--terminal-panel);
        border: 1px solid var(--terminal-border);
        border-radius: 7px;
        padding: 0.82rem 1rem;
    }
    [data-testid="stMetricLabel"] {color: var(--terminal-muted);}
    [data-testid="stMetricValue"] {color: var(--terminal-text);
        font-variant-numeric: tabular-nums;}
    [data-testid="stDataFrame"] {border: 1px solid var(--terminal-border);
        border-radius: 6px;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--terminal-border) !important;
        background: var(--terminal-panel);
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {
        background: var(--terminal-panel-2);
        border-color: var(--terminal-border);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {font-size: 0.93rem;}
    [data-testid="stSidebar"] .terminal-kicker {font-size: 0.95rem;}
    hr {border-color: var(--terminal-border);}
    .terminal-kicker {color: var(--terminal-accent); font-size: 0.73rem;
        font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;}
    .terminal-subtitle {color: var(--terminal-muted); margin-top: -0.7rem;}
    .terminal-status {color: #8fc5ff; font-family: monospace; font-size: 0.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading canonical price and return coverage…")
def cached_asset_universe() -> tuple[tuple[str, ...], pd.DataFrame]:
    return discover_asset_universe()


@st.cache_data(show_spinner="Calculating portfolio series through Phase 2…")
def cached_portfolio_result(
    assets: tuple[str, ...], weights: tuple[float, ...], initial_value: float
):
    portfolio = create_portfolio(
        assets, dict(zip(assets, weights, strict=True)), initial_value
    )
    return build_portfolio_result(portfolio)


@st.cache_data(show_spinner="Loading canonical prices…")
def cached_prices() -> pd.DataFrame:
    return load_clean_prices()


@st.cache_data(show_spinner="Calculating current-portfolio VaR and ES…")
def cached_risk_results(
    assets: tuple[str, ...],
    weights: tuple[float, ...],
    initial_value: float,
    confidence: float,
):
    result = cached_portfolio_result(assets, weights, initial_value)
    return calculate_risk_results(result, confidence)


@st.cache_data(show_spinner="Calculating scenario outcomes…")
def cached_stress_results(
    assets: tuple[str, ...],
    weights: tuple[float, ...],
    initial_value: float,
    base_value: float,
    asset_shock_pairs: tuple[tuple[str, float], ...],
):
    portfolio = create_portfolio(
        assets, dict(zip(assets, weights, strict=True)), initial_value
    )
    return build_stress_results(
        portfolio,
        prices=cached_prices(),
        base_value=base_value,
        asset_shocks=dict(asset_shock_pairs),
    )


@st.cache_data(show_spinner=False)
def cached_backtest_snapshot() -> dict:
    return load_phase4_snapshot()


@st.cache_data(show_spinner=False)
def cached_ml_snapshot() -> dict:
    return load_phase7_snapshot()


def render_sidebar(
    available_assets: tuple[str, ...],
) -> tuple[tuple[str, ...], dict[str, float], float, float, str, dict[str, float]]:
    st.sidebar.markdown(
        '<div class="terminal-kicker">RiskChallenger</div>', unsafe_allow_html=True
    )
    st.sidebar.caption("QUANTITATIVE RISK · PORTFOLIO ANALYTICS")
    page = st.sidebar.radio("Workspace", PAGES, key="workspace_page")
    st.sidebar.divider()

    if "selected_assets" not in st.session_state:
        demo = [asset for asset in DEFAULT_ASSETS if asset in available_assets]
        st.session_state.selected_assets = demo or list(available_assets[:3])
    assets = tuple(
        st.sidebar.multiselect(
            "Asset universe",
            options=available_assets,
            key="selected_assets",
            placeholder="Search ticker or company symbol",
            help=(
                f"{len(available_assets)} assets discovered from canonical prices "
                "and returns."
            ),
        )
    )
    if not assets:
        st.sidebar.warning("Select one or more assets to configure a portfolio.")

    if assets and st.sidebar.button("Set equal weights", use_container_width=True):
        equal_weight = 1.0 / len(assets)
        for asset in assets:
            st.session_state[f"portfolio_weight_{asset}"] = equal_weight

    weights: dict[str, float] = {}
    if assets:
        with st.sidebar.expander("Allocation weights", expanded=False):
            st.caption("Long-only allocation · values are never normalized")
            for asset in assets:
                key = f"portfolio_weight_{asset}"
                if key not in st.session_state:
                    st.session_state[key] = DEFAULT_WEIGHTS.get(asset, 0.0)
                weights[asset] = st.number_input(
                    asset,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.4f",
                    key=key,
                )
            total_weight = float(sum(weights.values()))
            st.metric("Allocation total", f"{total_weight:.2%}")

    initial_value = st.sidebar.number_input(
        "Portfolio value (₹)",
        min_value=1.0,
        value=float(st.session_state.get("portfolio_value", DEMO_VALUE)),
        step=50_000.0,
        key="portfolio_value",
    )
    confidence = st.sidebar.selectbox(
        "VaR confidence",
        [0.90, 0.95, 0.975, 0.99],
        index=1,
        format_func=lambda value: f"{value:.1%}",
    )

    shocks: dict[str, float] = {}
    if page == "Stress Testing" and assets:
        with st.sidebar.expander("Asset-specific scenario shocks", expanded=False):
            st.caption("Enter a simple return shock for each selected asset")
            for asset in assets:
                shocks[asset] = st.number_input(
                    asset,
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(DEMO_ASSET_SHOCKS.get(asset, -0.10)),
                    step=0.01,
                    format="%.2f",
                    key=f"stress_shock_{asset}",
                )
    st.sidebar.divider()
    st.sidebar.caption(f"Canonical coverage · {len(available_assets)} assets")
    st.sidebar.caption("Local data only · no external services")
    return assets, weights, float(initial_value), float(confidence), page, shocks


def render_header() -> None:
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            '<div class="terminal-kicker">RiskChallenger</div>', unsafe_allow_html=True
        )
        st.title("Quantitative Risk Management Terminal")
        st.markdown(
            '<div class="terminal-subtitle">Portfolio exposure · VaR / ES · '
            "stress testing · backtesting · model diagnostics</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="terminal-status">LOCAL DATA · RESEARCH MODE</div>',
            unsafe_allow_html=True,
        )
        st.caption("Risk forecasts are analytical outputs, not investment advice.")
    st.divider()


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def portfolio_is_valid(
    assets: tuple[str, ...], weights: dict[str, float]
) -> tuple[bool, str]:
    return validate_dashboard_weights(assets, weights)


def get_model_status(snapshot: dict, model_name: str) -> dict:
    return next(
        (row for row in snapshot.get("models", []) if row["model_name"] == model_name),
        {},
    )


def render_overview(result, portfolio, snapshot: dict, reference: bool) -> None:
    st.subheader("Portfolio overview")
    st.caption("Historical portfolio analytics from the Phase 2 engine.")
    metrics = overview_metrics(result)
    columns = st.columns(4)
    columns[0].metric("Portfolio value", money(metrics["final_portfolio_value"]))
    columns[1].metric("Selected assets", str(len(portfolio.assets)))
    columns[2].metric(
        "Annualized volatility", f"{metrics['annualized_volatility']:.2%}"
    )
    columns[3].metric("Maximum drawdown", f"{metrics['maximum_drawdown']:.2%}")

    left, right = st.columns([1.75, 1])
    with left:
        st.markdown("### Portfolio value · historical")
        st.line_chart(result.portfolio_value.rename("Portfolio value (₹)"), height=300)
        st.markdown("### Daily portfolio return")
        st.line_chart(result.portfolio_returns.rename("Daily return"), height=210)
    with right:
        st.markdown("### Allocation")
        allocation = pd.DataFrame(
            {"Asset": portfolio.assets, "Weight": portfolio.weight_vector}
        ).set_index("Asset")
        st.bar_chart(allocation, horizontal=True, height=300)
        st.metric("Cumulative return", f"{metrics['cumulative_return']:.2%}")
        st.metric(
            "Annualized arithmetic return", f"{metrics['annualized_mean_return']:.2%}"
        )
    st.caption(
        "Annualized return uses arithmetic mean × 252; volatility uses sample "
        "standard deviation × √252. The displayed historical value path starts "
        "at the configured portfolio value."
    )
    champion = snapshot.get("champion_challenger_summary", {}).get(
        "champion", "Unavailable"
    )
    garch = get_model_status(snapshot, "garch_1_1_gaussian")
    violation_rate = garch.get("violation_rate")
    a, b = st.columns([1, 2])
    with a:
        st.metric("Stored full-period Champion", champion)
    with b:
        if reference and violation_rate is not None:
            st.metric("Reference GARCH violation rate", f"{violation_rate:.2%}")
        else:
            st.info(
                "Stored backtest/Champion metrics refer to the reference portfolio, "
                "not this allocation."
            )


def render_portfolio(result, portfolio, coverage: pd.DataFrame) -> None:
    st.subheader("Portfolio construction & analytics")
    st.caption(
        "Selected allocation uses explicit Phase 2 weights; no automatic normalization."
    )
    allocation = pd.DataFrame(
        {"Asset": portfolio.assets, "Weight": portfolio.weight_vector}
    )
    selected_coverage = coverage[coverage["Asset"].isin(portfolio.assets)]
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("### Allocation by asset")
        st.bar_chart(allocation.set_index("Asset"), horizontal=True, height=340)
        st.dataframe(
            allocation.style.format({"Weight": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        metrics = overview_metrics(result)
        st.markdown("### Performance statistics")
        stats = pd.DataFrame(
            [
                ("Mean daily return", metrics["mean_daily_return"], "percent"),
                (
                    "Daily sample volatility",
                    metrics["standard_deviation_daily_return"],
                    "percent",
                ),
                (
                    "Annualized return (arithmetic)",
                    metrics["annualized_mean_return"],
                    "percent",
                ),
                ("Annualized volatility", metrics["annualized_volatility"], "percent"),
                ("Cumulative return", metrics["cumulative_return"], "percent"),
                ("Maximum drawdown", metrics["maximum_drawdown"], "percent"),
                ("Final portfolio value", metrics["final_portfolio_value"], "money"),
                ("Total P&L", metrics["total_pnl"], "money"),
            ],
            columns=["Statistic", "Value", "Format"],
        )
        stats["Value"] = stats.apply(
            lambda row: (
                f"{row['Value']:.2%}"
                if row["Format"] == "percent"
                else money(row["Value"])
            ),
            axis=1,
        )
        st.dataframe(
            stats[["Statistic", "Value"]], hide_index=True, use_container_width=True
        )
        st.markdown("### Canonical data coverage")
        st.dataframe(selected_coverage, hide_index=True, use_container_width=True)
    value_path = np.concatenate(
        ([portfolio.initial_value], result.portfolio_value.to_numpy(dtype=float))
    )
    drawdown = value_path / np.maximum.accumulate(value_path) - 1.0
    drawdown_series = pd.Series(
        drawdown[1:], index=result.portfolio_value.index, name="Drawdown"
    )
    st.markdown("### Drawdown profile")
    st.line_chart(drawdown_series, height=220)
    st.caption(
        f"{len(result.portfolio_returns):,} observations · "
        f"{result.portfolio_returns.index[0].date()} to "
        f"{result.portfolio_returns.index[-1].date()} · "
        "Phase 1’s undefined initial return is handled by the Phase 2 engine."
    )


def render_risk(portfolio, confidence: float) -> None:
    st.subheader("Value at Risk & Expected Shortfall")
    st.caption(
        f"{confidence:.1%} confidence · positive values denote losses · "
        "monetary estimates on the current portfolio’s Phase 2 P&L series."
    )
    try:
        risk_results = cached_risk_results(
            tuple(portfolio.assets),
            tuple(portfolio.weight_vector),
            portfolio.initial_value,
            confidence,
        )
        table = format_risk_results(risk_results)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        st.error(f"Risk estimates unavailable for this portfolio: {error}")
        return
    display_names = {
        "historical": "Historical",
        "gaussian": "Gaussian",
        "student_t": "Student-t",
        "monte_carlo": "Monte Carlo",
        "garch_1_1_gaussian": "GARCH(1,1) Gaussian",
    }
    table["Model"] = table["Model"].map(display_names).fillna(table["Model"])
    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(
            table.style.format(
                {"VaR": "₹{:,.0f}", "ES": "₹{:,.0f}", "Confidence": "{:.1%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.bar_chart(table.set_index("Model")[["VaR", "ES"]], height=330)
    st.info(
        "These are on-demand full-sample model estimates, not backtest forecasts. "
        "Monte Carlo uses its existing deterministic bootstrap settings; GARCH "
        "uses the Phase 3 one-step conditional model."
    )


def render_backtesting(snapshot: dict, reference: bool) -> None:
    st.subheader("Persisted historical backtesting")
    st.caption(
        "Phase 4 reference run · loaded from saved results; never run from this "
        "dashboard."
    )
    if not snapshot.get("available", True):
        st.warning(snapshot.get("message", "Stored backtest summary unavailable."))
        return
    if not reference:
        st.info(
            "Historical backtest results are available for the stored reference "
            "portfolio. Run the backtesting pipeline separately to generate "
            "results for this portfolio."
        )
    st.warning(
        "All figures below belong to the stored reference portfolio and its "
        "validated run."
    )
    period = snapshot["period"]
    st.caption(
        f"{period['forecast_count']:,} forecasts · "
        f"{period['confidence_level']:.0%} confidence · "
        f"{period['window_mode']} window · refit frequency {period['refit_frequency']}"
    )
    frame = pd.DataFrame(snapshot["models"]).rename(
        columns={
            "model_name": "Model",
            "violations": "Violations",
            "violation_rate": "Violation rate",
            "expected_rate": "Expected rate",
            "kupiec_p_value": "Kupiec p",
            "independence_p_value": "Christoffersen p",
            "conditional_coverage_p_value": "Conditional coverage p",
            "runtime_seconds": "Runtime (s)",
        }
    )
    summary = snapshot.get("champion_challenger_summary", {})
    statuses = {
        row["model_name"]: (
            "Champion"
            if row["model_name"] == summary.get("champion")
            else "Challenger"
            if row["eligible"]
            else "Ineligible"
        )
        for row in summary.get("ranking", [])
    }
    frame["Status"] = frame["Model"].map(statuses).fillna("Unavailable")
    st.dataframe(
        frame.style.format(
            {
                "Violation rate": "{:.2%}",
                "Expected rate": "{:.2%}",
                "Kupiec p": "{:.3g}",
                "Christoffersen p": "{:.3g}",
                "Conditional coverage p": "{:.3g}",
                "Runtime (s)": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    champion = summary.get("champion", "Unavailable")
    st.metric("Stored Champion", champion)
    st.caption(snapshot.get("source", "Persisted Phase 4 summary."))
    st.info(
        "Forecast-level dates were not persisted. This view does not recalculate "
        "violations or rerun Phase 4."
    )


def render_model_diagnostics(phase4: dict, phase7: dict, reference: bool) -> None:
    st.subheader("Model diagnostics")
    st.caption(
        "Saved diagnostics are evidence from prior runs, not guaranteed predictions."
    )
    if not phase4.get("available", True):
        st.warning(phase4.get("message", "Champion result unavailable."))
    else:
        summary = phase4.get("champion_challenger_summary", {})
        if not reference:
            st.info(
                "Champion–Challenger results below are for the stored reference "
                "portfolio only."
            )
        st.markdown("### Champion–Challenger · full-period reference")
        st.metric("Champion", summary.get("champion", "Unavailable"))
        ranking = pd.DataFrame(summary.get("ranking", []))
        if not ranking.empty:
            labels = {
                "model_name": "Model",
                "eligible": "Eligible",
                "coverage_score": "Coverage",
                "exception_rate_score": "Exception accuracy",
                "stability_score": "Stability",
                "operational_score": "Operational",
                "total_score": "Final score",
            }
            ranking = ranking.rename(columns=labels)
            st.dataframe(
                ranking.style.format(
                    {
                        "Coverage": "{:.3f}",
                        "Exception accuracy": "{:.3f}",
                        "Stability": lambda value: (
                            "Unavailable" if pd.isna(value) else f"{value:.3f}"
                        ),
                        "Operational": "{:.3f}",
                        "Final score": "{:.3f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        garch = get_model_status(phase4, "garch_1_1_gaussian")
        if garch:
            st.markdown("### GARCH reference backtest diagnostics")
            cols = st.columns(4)
            cols[0].metric("Violation rate", f"{garch['violation_rate']:.2%}")
            cols[1].metric("Kupiec p-value", f"{garch['kupiec_p_value']:.3g}")
            cols[2].metric(
                "Independence p-value", f"{garch['independence_p_value']:.3g}"
            )
            cols[3].metric(
                "Conditional coverage p-value",
                f"{garch['conditional_coverage_p_value']:.3g}",
            )
    st.divider()
    st.markdown("### Phase 7 ML recent-period diagnostic")
    if not phase7.get("available", True):
        st.info(phase7.get("message", "Recent-period diagnostic unavailable."))
        return
    st.warning(
        phase7.get("source", "Saved Phase 7 result")
        + " This recent-period diagnostic is not a replacement for the "
        "full-period backtest."
    )
    rows = pd.DataFrame(phase7.get("models", []))
    if not rows.empty:
        rows = rows.rename(
            columns={
                "model_name": "Model",
                "violations": "Violations",
                "violation_rate": "Violation rate",
                "expected_rate": "Expected rate",
                "kupiec_p_value": "Kupiec p",
                "independence_p_value": "Christoffersen p",
                "conditional_coverage_p_value": "Conditional coverage p",
                "average_predicted_volatility": "Avg predicted volatility",
                "average_var": "Avg VaR (₹)",
                "runtime_seconds": "Runtime (s)",
                "eligible": "Eligible",
            }
        )
        diagnostic_columns = [
            "Model",
            "Eligible",
            "Violations",
            "Violation rate",
            "Expected rate",
            "Kupiec p",
            "Christoffersen p",
            "Conditional coverage p",
            "Avg predicted volatility",
            "Avg VaR (₹)",
            "Runtime (s)",
        ]
        shown_columns = [column for column in diagnostic_columns if column in rows]
        st.dataframe(
            rows[shown_columns].style.format(
                {
                    "Violation rate": "{:.2%}",
                    "Expected rate": "{:.2%}",
                    "Kupiec p": "{:.3g}",
                    "Christoffersen p": "{:.3g}",
                    "Conditional coverage p": "{:.3g}",
                    "Avg predicted volatility": "{:.3%}",
                    "Avg VaR (₹)": "₹{:,.0f}",
                    "Runtime (s)": "{:.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        importance = next(
            (row for row in phase7["models"] if row.get("feature_importance")), None
        )
        if importance:
            selected_model = st.selectbox(
                "Saved feature importance / coefficient",
                [
                    row["model_name"]
                    for row in phase7["models"]
                    if row.get("feature_importance")
                ],
            )
            selected = next(
                row for row in phase7["models"] if row["model_name"] == selected_model
            )
            importance_series = pd.Series(
                selected["feature_importance"], name="Importance / coefficient"
            ).sort_values(key=abs, ascending=False)
            st.bar_chart(importance_series, horizontal=True)
            st.caption(
                "Saved diagnostic feature importance/coefficients are not causal "
                "explanations."
            )


def render_stress(
    portfolio,
    current_value: float,
    asset_shocks: dict[str, float],
) -> None:
    st.subheader("Scenario stress testing")
    st.caption(
        "Scenario outcomes are conditional what-if analyses, not probability "
        "forecasts or VaR."
    )
    try:
        results = cached_stress_results(
            tuple(portfolio.assets),
            tuple(portfolio.weight_vector),
            portfolio.initial_value,
            current_value,
            tuple(asset_shocks.items()),
        )
    except (KeyError, ValueError, TypeError) as error:
        st.error(f"Stress scenarios unavailable for this selection: {error}")
        return
    table = format_stress_results(results).rename(
        columns={
            "scenario_name": "Scenario",
            "base_value": "Base value",
            "stressed_value": "Stressed value",
            "pnl": "P&L",
            "loss": "Loss",
            "loss_percentage": "Loss %",
        }
    )
    styled_table = table[
        ["Scenario", "Base value", "Stressed value", "P&L", "Loss", "Loss %"]
    ].style.format(
        {
            "Base value": "₹{:,.0f}",
            "Stressed value": "₹{:,.0f}",
            "P&L": "₹{:,.0f}",
            "Loss": "₹{:,.0f}",
            "Loss %": "{:.2f}%",
        }
    )
    styled_table = styled_table.map(
        lambda value: "color: #ff7777" if value < 0 else "color: #61d095",
        subset=["P&L"],
    ).map(
        lambda value: "color: #ff7777" if value > 0 else "color: #61d095",
        subset=["Loss", "Loss %"],
    )
    st.dataframe(
        styled_table,
        hide_index=True,
        use_container_width=True,
    )
    selected_name = st.selectbox(
        "Scenario attribution", [item.scenario_name for item in results]
    )
    detail = next(item for item in results if item.scenario_name == selected_name)
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown("### Asset P&L contribution")
        attribution = detail.asset_impacts_frame.set_index("asset")[
            ["pnl_contribution"]
        ]
        st.bar_chart(
            attribution.rename(columns={"pnl_contribution": "P&L (₹)"}), horizontal=True
        )
    with right:
        st.markdown("### Scenario attribution detail")
        st.dataframe(
            detail.asset_impacts_frame[
                [
                    "asset",
                    "weight",
                    "return_shock",
                    "pnl_contribution",
                    "loss_contribution",
                ]
            ].style.format(
                {
                    "weight": "{:.2%}",
                    "return_shock": "{:.2%}",
                    "pnl_contribution": "₹{:,.0f}",
                    "loss_contribution": "₹{:,.0f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    if detail.start_date is not None:
        st.caption(
            f"Historical window: {detail.start_date.date()} to {detail.end_date.date()}"
        )
    with st.expander("Scenario assumptions"):
        for assumption in detail.assumptions:
            st.write(f"• {assumption}")
    st.caption(
        "Historical replay applies the selected current weights to the canonical "
        "event-return vector."
    )


def main() -> None:
    try:
        assets_available, coverage = cached_asset_universe()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Canonical asset universe unavailable: {error}")
        return
    assets, weights, initial_value, confidence, page, asset_shocks = render_sidebar(
        assets_available
    )
    render_header()
    phase4 = cached_backtest_snapshot()
    phase7 = cached_ml_snapshot()
    if not assets:
        st.warning(
            "Choose assets in the sidebar to enable portfolio-dependent analysis."
        )
        if page == "Backtesting":
            render_backtesting(phase4, False)
        elif page == "Model Diagnostics":
            render_model_diagnostics(phase4, phase7, False)
        return

    is_valid, validation_message = portfolio_is_valid(assets, weights)
    try:
        portfolio = (
            create_portfolio(assets, weights, initial_value) if is_valid else None
        )
    except (TypeError, ValueError) as error:
        portfolio = None
        is_valid = False
        validation_message = str(error)
    if not is_valid:
        st.sidebar.error(validation_message)
    else:
        st.sidebar.success(validation_message)

    reference = portfolio is not None and is_reference_portfolio(portfolio)
    if page == "Backtesting":
        render_backtesting(phase4, reference)
        return
    if page == "Model Diagnostics":
        render_model_diagnostics(phase4, phase7, reference)
        return
    if portfolio is None:
        st.warning(
            "Portfolio-dependent analytics are paused until the allocation validates."
        )
        st.markdown("### Allocation controls")
        st.write(validation_message)
        return

    try:
        result = cached_portfolio_result(
            tuple(portfolio.assets),
            tuple(portfolio.weight_vector),
            portfolio.initial_value,
        )
    except (FileNotFoundError, ValueError, TypeError) as error:
        st.error(f"Portfolio analytics unavailable for this selection: {error}")
        return
    if page == "Overview":
        render_overview(result, portfolio, phase4, reference)
    elif page == "Portfolio":
        render_portfolio(result, portfolio, coverage)
    elif page == "Risk":
        render_risk(portfolio, confidence)
    elif page == "Stress Testing":
        try:
            render_stress(
                portfolio, float(result.portfolio_value.iloc[-1]), asset_shocks
            )
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Canonical price history unavailable for stress replay: {error}")
    st.caption(
        f"Selected portfolio · {len(portfolio.assets)} assets · "
        f"{result.portfolio_returns.index[0].date()} to "
        f"{result.portfolio_returns.index[-1].date()} · "
        "positive loss values indicate adverse outcomes."
    )


if __name__ == "__main__":
    main()
