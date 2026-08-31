import time
from datetime import timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import (
    DEMAND_SCALING_FACTOR,
    build_locality_demand_scores,
    create_city_dataset,
    create_daily_dataset,
    create_features,
    get_cities,
    get_localities,
    get_forecast_results,
    run_decision_support,
)

st.set_page_config(
    page_title="Zomato Demand Forecasting",
    page_icon="🍽️",
    layout="wide",
)

st.title("Zomato Demand Forecasting & Rider Planning")
st.markdown("Demand forecasting and rider allocation decision-support system")
st.divider()

model = joblib.load("forecast_model.pkl")

# ------------------------------------------------------------
# LOCATION SELECTION
# ------------------------------------------------------------
st.subheader("Select Location")

cities = get_cities()
col1, col2 = st.columns(2)

with col1:
    selected_city = st.selectbox("City", cities)

with col2:
    localities = get_localities(selected_city)
    selected_locality = st.selectbox("Locality", localities)

generate_forecast = st.button("🔍 Generate Forecast")

if generate_forecast:
    city_data = create_city_dataset(selected_city)
    city_daily = create_daily_dataset(city_data)

    last_available_date = city_daily["order_date"].max()
    forecast_date = last_available_date + timedelta(days=1)

    st.write(
        f"📅 Forecast Date: {forecast_date.strftime('%d %B %Y')} "
        "(Based on latest available operational data)"
    )
    st.caption(
        f"Latest historical data available: "
        f"{last_available_date.strftime('%d %B %Y')}"
    )

    # --------------------------------------------------------
    # FORECAST PROCESSING
    # --------------------------------------------------------
    with st.status(
        "🔄 Generating operational forecast...",
        expanded=True,
    ) as status:
        st.write("📊 Preparing historical demand data...")
        time.sleep(0.8)

        st.write("⚙️ Creating forecasting features...")
        city_daily = create_features(city_daily)
        time.sleep(0.8)

        st.write("🤖 Generating next-day demand forecast...")
        forecast_orders, _ = get_forecast_results(model, city_daily)
        time.sleep(1)

        st.write("📍 Allocating demand across localities...")
        locality_orders = build_locality_demand_scores(city_data)
        time.sleep(0.8)

        st.write("🛵 Calculating rider requirements...")
        results = run_decision_support(
            city_name=selected_city,
            locality_name=selected_locality,
            forecast_orders=forecast_orders,
            locality_orders=locality_orders,
            city_daily=city_daily,
        )
        time.sleep(0.8)

        status.update(
            label="✅ Forecast generated successfully",
            state="complete",
            expanded=False,
        )

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------
    st.divider()
    main_col, insights_col = st.columns([3, 1])

    with main_col:
        st.subheader("📊 Forecast Summary")
        st.caption(
            f"📅 Forecast Date: {forecast_date.strftime('%d %B %Y')}  |  "
            f"📍 Location: {selected_city}"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Expected Orders",
                f"{results['forecast_orders']:.0f}",
            )
        with c2:
            st.metric(
                "Historical Average",
                f"{results['historical_average']:.2f}",
            )
        with c3:
            st.metric("Forecast Reliability", results["confidence"])

        demand_status = results["demand_status"].upper()
        demand_indicator = {
            "HIGH": "🔴 HIGH",
            "MEDIUM": "🟡 MODERATE",
            "MODERATE": "🟡 MODERATE",
            "LOW": "🟢 LOW",
        }.get(demand_status, "🔴 UNAVAILABLE")
        st.write(f"**Demand Level:** {demand_indicator}")

        st.subheader("📍 Locality Planning")
        locality_info = results["selected_locality"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Expected Orders",
                f"{locality_info['forecast_orders']:.0f}",
            )
        with c2:
            st.metric(
                "Recommended Riders",
                f"{locality_info['recommended_riders']:.0f}",
            )
        with c3:
            st.metric(
                "Standby Riders",
                f"{locality_info['standby_riders']:.0f}",
            )

        st.write(f"**Priority:** {locality_info['priority']}")
        st.info(locality_info["recommendation"])

        st.markdown("---")
        st.subheader("📈 Demand Analysis")
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown("### Forecast Trend")

            trend_data = city_daily.copy()
            trend_data["orders"] *= DEMAND_SCALING_FACTOR
            trend_data["order_date"] = pd.to_datetime(
                trend_data["order_date"]
            )
            trend_data = trend_data.tail(30)

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_data["order_date"],
                    y=trend_data["orders"],
                    mode="lines+markers",
                    name="Historical Demand",
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>Date:</b> %{x|%d %b %Y}<br>"
                        "<b>Orders:</b> %{y}<extra></extra>"
                    ),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=[forecast_date],
                    y=[forecast_orders],
                    mode="markers",
                    name="Model Forecast",
                    marker=dict(size=14, symbol="circle"),
                    hovertemplate=(
                        "<b>Forecast Date:</b> %{x|%d %b %Y}<br>"
                        "<b>Expected Orders:</b> %{y}<extra></extra>"
                    ),
                )
            )
            fig_trend.add_hline(
                y=results["historical_average"],
                line_dash="dash",
                annotation_text="Historical Average",
                annotation_position="top left",
            )
            fig_trend.update_layout(
                template="plotly_dark",
                height=380,
                xaxis_title="Date",
                yaxis_title="Orders",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                xaxis=dict(showgrid=False),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                ),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with graph_col2:
            st.markdown("### 🔥 Demand Hotspots by Locality")
            allocation = results.get("allocation")

            if allocation is not None and not allocation.empty:
                top_localities = (
                    allocation.sort_values(
                        "forecast_orders",
                        ascending=False,
                    )
                    .head(8)
                    .sort_values(
                        "forecast_orders",
                        ascending=True,
                    )
                )

                fig_locality = go.Figure()
                fig_locality.add_trace(
                    go.Bar(
                        x=top_localities["forecast_orders"],
                        y=top_localities["locality"],
                        orientation="h",
                        name="Expected Demand",
                        text=top_localities["forecast_orders"],
                        textposition="outside",
                        hovertemplate=(
                            "<b>Locality:</b> %{y}<br>"
                            "<b>Expected Orders:</b> %{x}<extra></extra>"
                        ),
                    )
                )
                fig_locality.update_layout(
                    template="plotly_dark",
                    height=380,
                    xaxis_title="Expected Orders",
                    yaxis_title="Locality",
                    margin=dict(l=20, r=35, t=20, b=20),
                    showlegend=False,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.10)",
                    ),
                    yaxis=dict(showgrid=False),
                    hoverlabel=dict(namelength=-1),
                )
                st.plotly_chart(
                    fig_locality,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Locality-level demand could not be generated "
                    "because sufficient data was not available."
                )

    # ------------------------------------------------------------
    # PLANNING INSIGHTS
    # ------------------------------------------------------------
    with insights_col:
        st.markdown(
            "<h2 style='font-size:30px;margin-bottom:15px;'>"
            "💡 Planning Insights</h2>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='font-size:22px;margin-bottom:8px;'>"
            "🚴 Rider Allocation</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:15px;line-height:1.35;'>"
            f"{results['rider_insight']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='margin:12px 0;border:0;"
            "border-top:1px solid #333;'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='font-size:22px;margin-bottom:8px;'>"
            "⚠️ Operational Risk</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:15px;line-height:1.35;'>"
            f"{results['risk_insight']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='margin:12px 0;border:0;"
            "border-top:1px solid #333;'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='font-size:22px;margin:0 0 5px 0;'>"
            "📍 Locality Priority</h3>",
            unsafe_allow_html=True,
        )

        if allocation is not None and not allocation.empty:
            priority_localities = (
                allocation.sort_values(
                    "forecast_orders",
                    ascending=False,
                )
                .head(3)["locality"]
                .tolist()
            )
            locality_text = ", ".join(
                f"<span style='color:#4da6ff;font-weight:700'>{x}</span>"
                for x in priority_localities
            )
            st.markdown(
                f"<div style='font-size:15px;line-height:1.3;'>"
                f"Monitor demand concentration in:"
                f"<div style='margin-top:8px;margin-bottom:4px;'>"
                f"{locality_text}</div>"
                f"<div style='margin-top:16px;margin-bottom:16px;'>"
                f"These zones require higher rider availability due "
                f"to comparatively higher expected demand.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.write("Locality priority is unavailable.")

        st.markdown(
            "<hr style='margin:8px 0;border:0;"
            "border-top:1px solid #333;'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='font-size:22px;margin-bottom:8px;'>"
            "📊 Decision Confidence</h3>",
            unsafe_allow_html=True,
        )
        st.progress(results["decision_confidence"] / 100)
        st.markdown(
            f"<div style='margin-top:5px;margin-bottom:15px;'>"
            f"<span style='font-size:22px;font-weight:700;"
            f"color:#4da6ff;'>{results['decision_confidence']}%</span>"
            f"<br><span style='font-size:14px;'>"
            f"Confidence in the operational recommendation"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='font-size:22px;margin-bottom:8px;'>"
            "Why should this work?</h3>",
            unsafe_allow_html=True,
        )
        for reason in results["confidence_reasons"]:
            st.markdown(
                f"<div style='font-size:14px;line-height:1.35;"
                f"margin-bottom:8px;'>✓ {reason}</div>",
                unsafe_allow_html=True,
            )
