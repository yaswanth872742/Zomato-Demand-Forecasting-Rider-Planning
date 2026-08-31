from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# FILES + CASE-STUDY RULES
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Datasets" / "DATA1_Cleaned.xlsx"
MODEL_FILE = BASE_DIR / "forecast_model.pkl"

MIN_RECORDS = 30
LOW_DEMAND_THRESHOLD = 20
DEMAND_SCALING_FACTOR = 100
TOP_LOCALITIES = 10
ACTIVE_DEMAND_SHARE = 0.90

# ------------------------------------------------------------
# LOAD + VALIDATE
# ------------------------------------------------------------
df = pd.read_excel(DATA_FILE)
model = joblib.load(MODEL_FILE)

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

required_columns = ["city", "locality", "order_date"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(
        "Dataset validation failed. "
        f"Missing required columns: {', '.join(missing_columns)}"
    )

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date", "city", "locality"]).copy()

# ------------------------------------------------------------
# LOCATION / VALIDATION
# ------------------------------------------------------------
def get_cities():
    return sorted(df["city"].dropna().unique().tolist())


def get_localities(selected_city):
    return sorted(
        df.loc[df["city"] == selected_city, "locality"]
        .dropna()
        .unique()
        .tolist()
    )


def validate_locality(selected_city, selected_locality):
    data = df[
        (df["city"] == selected_city)
        & (df["locality"] == selected_locality)
    ]
    return data, len(data)


def forecast_available(records):
    return records >= MIN_RECORDS


# ------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------
def create_city_dataset(selected_city):
    return (
        df[df["city"] == selected_city]
        .copy()
        .sort_values("order_date")
    )


def create_daily_dataset(city_data):
    daily = (
        city_data.groupby("order_date")
        .size()
        .reset_index(name="orders")
        .set_index("order_date")
        .asfreq("D")
        .fillna(0)
        .reset_index()
    )
    daily["orders"] = daily["orders"].astype(int)
    return daily


def create_features(data):
    data = data.copy()

    data["year"] = data["order_date"].dt.year
    data["month"] = data["order_date"].dt.month
    data["day"] = data["order_date"].dt.day
    data["day_of_week"] = data["order_date"].dt.dayofweek
    data["quarter"] = data["order_date"].dt.quarter
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    data["lag_1"] = data["orders"].shift(1)
    data["lag_7"] = data["orders"].shift(7)
    data["lag_30"] = data["orders"].shift(30)
    data["rolling_mean_7"] = data["orders"].rolling(7).mean()
    data["rolling_std_7"] = data["orders"].rolling(7).std()

    return data.dropna().reset_index(drop=True)


# ------------------------------------------------------------
# FORECASTING
# ------------------------------------------------------------
def create_tomorrow_features(city_daily):
    if len(city_daily) < 30:
        raise ValueError(
            "At least 30 daily observations are required for forecasting."
        )

    last_row = city_daily.iloc[-1]
    tomorrow = last_row["order_date"] + pd.Timedelta(days=1)

    return pd.DataFrame(
        [{
            "year": tomorrow.year,
            "month": tomorrow.month,
            "day": tomorrow.day,
            "day_of_week": tomorrow.dayofweek,
            "quarter": ((tomorrow.month - 1) // 3) + 1,
            "is_weekend": int(tomorrow.dayofweek >= 5),
            "lag_1": last_row["orders"],
            "lag_7": city_daily.iloc[-7]["orders"],
            "lag_30": city_daily.iloc[-30]["orders"],
            "rolling_mean_7": city_daily["orders"].tail(7).mean(),
            "rolling_std_7": city_daily["orders"].tail(7).std(),
        }]
    )


def calibrate_demand(raw_prediction):
    return max(0, round(raw_prediction * DEMAND_SCALING_FACTOR))


def predict_tomorrow(forecast_model, city_daily):
    features = create_tomorrow_features(city_daily)
    return calibrate_demand(forecast_model.predict(features)[0])


def forecast_summary(city_daily, forecast_orders):
    historical_average = (
        city_daily["orders"].mean() * DEMAND_SCALING_FACTOR
    )
    rolling_average = (
        city_daily["orders"].tail(7).mean() * DEMAND_SCALING_FACTOR
    )
    rolling_std = (
        city_daily["orders"].tail(7).std() * DEMAND_SCALING_FACTOR
    )

    if rolling_std < 75:
        confidence = "High"
    elif rolling_std < 150:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "Forecast Orders": forecast_orders,
        "Historical Average": round(historical_average, 2),
        "7-Day Average": round(rolling_average, 2),
        "Confidence": confidence,
    }


def get_forecast_results(forecast_model, city_daily):
    forecast_orders = predict_tomorrow(forecast_model, city_daily)
    return forecast_orders, forecast_summary(city_daily, forecast_orders)


# ------------------------------------------------------------
# LOCALITY DEMAND SCORE
# FILTER FIRST -> NORMALIZE -> SCORE
# ------------------------------------------------------------
def build_locality_demand_scores(city_data):
    latest_date = city_data["order_date"].max()
    recent_cutoff = latest_date - pd.Timedelta(days=30)

    recent_data = city_data[
        city_data["order_date"] >= recent_cutoff
    ]

    historical = (
        city_data.groupby("locality")
        .size()
        .rename("historical_orders")
    )
    recent = (
        recent_data.groupby("locality")
        .size()
        .rename("recent_orders")
    )

    locality_orders = (
        pd.concat([historical, recent], axis=1)
        .fillna(0)
        .reset_index()
    )

    # Important: insufficiently supported localities do not
    # participate in hotspot allocation.
    locality_orders = locality_orders[
        locality_orders["historical_orders"] >= MIN_RECORDS
    ].copy()

    if locality_orders.empty:
        return locality_orders

    historical_max = locality_orders["historical_orders"].max()
    recent_max = locality_orders["recent_orders"].max()

    locality_orders["historical_norm"] = (
        locality_orders["historical_orders"] / historical_max
        if historical_max > 0 else 0
    )
    locality_orders["recent_norm"] = (
        locality_orders["recent_orders"] / recent_max
        if recent_max > 0 else 0
    )

    locality_orders["demand_score"] = (
        0.75 * locality_orders["historical_norm"]
        + 0.25 * locality_orders["recent_norm"]
    )
    locality_orders["orders"] = locality_orders["demand_score"]

    return locality_orders


# ------------------------------------------------------------
# LOCALITY ALLOCATION
# 90% OF CITY FORECAST -> TOP 10 HIGH-SCORE LOCALITIES
# REMAINING 10% = GENERAL / UNASSIGNED CITY BUFFER
# ------------------------------------------------------------
def allocate_forecast(city_forecast, locality_orders):
    allocation = (
        locality_orders.copy()
        .sort_values("demand_score", ascending=False)
        .reset_index(drop=True)
    )

    if allocation.empty:
        allocation["forecast_orders"] = pd.Series(dtype=int)
        return allocation

    top_n = min(TOP_LOCALITIES, len(allocation))
    allocation["forecast_orders"] = 0

    active = allocation.index < top_n
    active_demand = round(city_forecast * ACTIVE_DEMAND_SHARE)
    active_total = allocation.loc[active, "demand_score"].sum()

    if active_total > 0 and active_demand > 0:
        raw = (
            allocation.loc[active, "demand_score"]
            / active_total
            * active_demand
        )
        allocation.loc[active, "forecast_orders"] = (
            raw.round().astype(int)
        )

        # Rounding correction stays INSIDE the 90% active pool.
        rounding_difference = (
            active_demand
            - allocation.loc[active, "forecast_orders"].sum()
        )
        if rounding_difference:
            first_active_index = allocation.index[active][0]
            allocation.loc[
                first_active_index, "forecast_orders"
            ] += rounding_difference

    # The remaining 10% is intentionally not assigned to a locality.
    # This preserves the 90% high-demand-zone planning assumption.
    return (
        allocation
        .sort_values("forecast_orders", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# RIDER PLANNING
# ------------------------------------------------------------
def recommend_riders(forecast_orders):
    if forecast_orders <= 0:
        return 0
    return int(np.ceil((forecast_orders / 5) * 1.25))


def recommend_standby_riders(forecast_orders, recommended_riders):
    if forecast_orders <= 0:
        return 0
    if recommended_riders <= 3:
        return 1
    if recommended_riders <= 8:
        return 2
    return int(np.ceil(recommended_riders * 0.2))


def classify_priority(forecast_orders):
    if forecast_orders <= 3:
        return "🟢 Low"
    if forecast_orders <= 6:
        return "🟡 Medium"
    if forecast_orders <= 10:
        return "🟠 High"
    return "🔴 Critical"


def generate_locality_plan(forecast_orders, locality_orders):
    allocation = allocate_forecast(
        forecast_orders,
        locality_orders,
    )

    if allocation.empty:
        return allocation

    allocation["recommended_riders"] = (
        allocation["forecast_orders"].apply(recommend_riders)
    )
    allocation["standby_riders"] = allocation.apply(
        lambda row: recommend_standby_riders(
            row["forecast_orders"],
            row["recommended_riders"],
        ),
        axis=1,
    )
    allocation["priority"] = (
        allocation["forecast_orders"].apply(classify_priority)
    )

    allocation["recommendation"] = allocation.apply(
        lambda row: (
            "No rider deployment required. "
            "Forecast demand is unavailable or negligible for this locality."
            if row["forecast_orders"] == 0
            else (
                f"Deploy {row['recommended_riders']} riders "
                f"with {row['standby_riders']} standby coverage."
            )
        ),
        axis=1,
    )

    return allocation


# ------------------------------------------------------------
# DECISION SUPPORT
# ------------------------------------------------------------
def _confidence_score(confidence):
    return {"High": 90, "Medium": 75, "Low": 55}[confidence]


def _base_response(summary, demand_status, confidence):
    return {
        "forecast_orders": summary["Forecast Orders"],
        "historical_average": summary["Historical Average"],
        "moving_average": summary["7-Day Average"],
        "confidence": confidence,
        "demand_status": demand_status,
    }


def _unavailable_response(
    summary,
    demand_status,
    confidence,
    locality_name,
    message,
    allocation=None,
    historical_records=None,
):
    if historical_records is not None:
        reasons = [
            f"Only {historical_records} historical records are available "
            "for the selected locality.",
            f"At least {MIN_RECORDS} records are required for a reliable "
            "locality-level forecast.",
            "A reliable locality-level forecast cannot be generated.",
        ]
    else:
        reasons = [
            "The selected locality could not be safely supported "
            "by the available data."
        ]

    response = _base_response(
        summary,
        demand_status,
        confidence,
    )
    response.update({
        "data_status": "Insufficient",
        "selected_locality": {
            "name": locality_name,
            "forecast_orders": 0,
            "recommended_riders": 0,
            "standby_riders": 0,
            "priority": "N/A",
            "recommendation": message,
        },
        "operational_status": "NOT READY",
        "risk_level": "High",
        "peak_demand_period": "Not available from daily-level data",
        "decision_confidence": 0,
        "confidence_reasons": reasons,
        "top_priorities": None,
        "allocation": (
            allocation
            if allocation is not None
            else pd.DataFrame()
        ),
        "demand_insight": (
            "Demand insight is unavailable because sufficient locality "
            "data is not available."
        ),
        "locality_insight": (
            "Locality-level demand could not be generated because "
            "sufficient data was not available."
        ),
        "rider_insight": (
            "Rider allocation is unavailable because a reliable "
            "locality-level forecast could not be generated."
        ),
        "risk_insight": (
            "Risk assessment is unavailable because sufficient "
            "locality-level demand data is not available."
        ),
    })
    return response


def run_decision_support(
    city_name,
    locality_name,
    forecast_orders,
    locality_orders,
    city_daily,
):
    summary = forecast_summary(city_daily, forecast_orders)
    confidence = summary["Confidence"]

    if forecast_orders < summary["Historical Average"]:
        demand_status = "LOW"
    elif forecast_orders < summary["Historical Average"] * 1.2:
        demand_status = "MEDIUM"
    else:
        demand_status = "HIGH"

    _, historical_records = validate_locality(
        city_name,
        locality_name,
    )

    if historical_records < MIN_RECORDS:
        return _unavailable_response(
            summary,
            demand_status,
            confidence,
            locality_name,
            (
                f"Forecast unavailable. This locality has only "
                f"{historical_records} historical records, while at least "
                f"{MIN_RECORDS} records are required."
            ),
            historical_records=historical_records,
        )

    if forecast_orders < LOW_DEMAND_THRESHOLD:
        response = _base_response(
            summary,
            demand_status,
            confidence,
        )
        response.update({
            "data_status": "Sufficient",
            "selected_locality": {
                "name": locality_name,
                "forecast_orders": 0,
                "recommended_riders": 0,
                "standby_riders": 0,
                "priority": "Low",
                "recommendation": (
                    "Demand is low. Maintain normal rider coverage "
                    "without additional deployment."
                ),
            },
            "operational_status": "READY",
            "risk_level": "Low",
            "peak_demand_period": "No significant peak identified",
            "decision_confidence": _confidence_score(confidence),
            "confidence_reasons": [
                "Forecast indicates low overall demand.",
                "Additional rider deployment is unlikely to be required.",
                "The recommendation is supported by the available demand pattern.",
            ],
            "top_priorities": None,
            "allocation": None,
            "demand_insight": (
                f"Forecasted demand is {forecast_orders} orders, which is "
                f"below the historical average of "
                f"{round(summary['Historical Average'])} orders. "
                "Demand is expected to remain manageable."
            ),
            "locality_insight": (
                "Locality-level allocation is not required for the "
                "low-demand scenario."
            ),
            "rider_insight": (
                "No additional rider deployment is recommended for "
                "the selected scenario."
            ),
            "risk_insight": (
                "LOW - Overall demand is below the historical average."
            ),
        })
        return response

    allocation = generate_locality_plan(
        forecast_orders,
        locality_orders,
    )

    locality_match = allocation[
        allocation["locality"] == locality_name
    ]

    if locality_match.empty:
        return _unavailable_response(
            summary,
            demand_status,
            confidence,
            locality_name,
            (
                "This locality has sufficient historical data, but it "
                "is outside the top operational demand zones selected "
                "for the current forecast."
            ),
            allocation=allocation,
        )

    locality_info = locality_match.iloc[0]
    locality_forecast = int(locality_info["forecast_orders"])
    recommended_riders = int(locality_info["recommended_riders"])
    standby_riders = int(locality_info["standby_riders"])

    if locality_forecast <= 0:
        locality_demand_status = "No Demand"
    elif locality_forecast <= 5:
        locality_demand_status = "Low"
    elif locality_forecast <= 20:
        locality_demand_status = "Moderate"
    else:
        locality_demand_status = "High"

    if locality_demand_status == "High":
        risk_level = "High"
        recommendation = (
            f"Deploy {recommended_riders} riders and keep "
            f"{standby_riders} standby riders to handle expected "
            "high demand."
        )
    elif locality_demand_status == "Moderate":
        risk_level = "Medium"
        recommendation = (
            f"Deploy {recommended_riders} riders and keep "
            f"{standby_riders} standby riders for demand fluctuations."
        )
    elif locality_demand_status == "Low":
        risk_level = "Low"
        recommendation = (
            f"Deploy {recommended_riders} riders with minimal "
            "standby coverage."
        )
    else:
        risk_level = "Low"
        recommendation = (
            "No rider deployment required. "
            "No predicted demand for this locality."
        )

    top_priorities = (
        allocation
        .sort_values("forecast_orders", ascending=False)
        .head(TOP_LOCALITIES)
    )

    top_localities = (
        allocation
        .sort_values("forecast_orders", ascending=False)
        .head(3)["locality"]
        .tolist()
    )

    if forecast_orders < summary["Historical Average"]:
        demand_insight = (
            f"Forecasted demand is {forecast_orders} orders, which is below "
            f"the historical average of "
            f"{round(summary['Historical Average'])} orders. "
            "Demand is expected to remain manageable."
        )
    else:
        demand_insight = (
            f"Forecasted demand is {forecast_orders} orders, which is above "
            f"the historical average of "
            f"{round(summary['Historical Average'])} orders. "
            "Higher rider availability may be required."
        )

    locality_insight = (
        "Highest demand zones: "
        + ", ".join(top_localities)
        + ". Prioritize rider availability in these areas."
    )

    rider_insight = (
        f"Recommended deployment: {recommended_riders} riders with "
        f"{standby_riders} standby rider(s) to handle demand fluctuations."
    )

    if locality_demand_status == "No Demand":
        risk_insight = (
            "LOW - No predicted demand for this locality. "
            "No additional rider allocation is required."
        )
    elif locality_demand_status == "High":
        risk_insight = (
            "HIGH - Locality demand is elevated. "
            "Additional rider availability may be required."
        )
    elif locality_demand_status == "Moderate":
        risk_insight = (
            "MEDIUM - Locality demand is moderately elevated. "
            "Monitor rider availability."
        )
    else:
        risk_insight = (
            "LOW - Locality demand is within the lower planning range."
        )

    response = _base_response(
        summary,
        demand_status,
        confidence,
    )
    response.update({
        "data_status": "Sufficient",
        "selected_locality": {
            "name": locality_info["locality"],
            "forecast_orders": locality_forecast,
            "recommended_riders": recommended_riders,
            "standby_riders": standby_riders,
            "priority": locality_info["priority"],
            "recommendation": recommendation,
        },
        "operational_status": "READY",
        "risk_level": risk_level,
        "peak_demand_period": "Not available from daily-level data",
        "decision_confidence": _confidence_score(confidence),
        "confidence_reasons": [
            "Forecast is based on historical demand patterns.",
            "Locality-level demand distribution is used for rider allocation.",
            "Recent demand behavior is considered through the forecasting features.",
        ],
        "top_priorities": top_priorities,
        "allocation": allocation,
        "demand_insight": demand_insight,
        "locality_insight": locality_insight,
        "rider_insight": rider_insight,
        "risk_insight": risk_insight,
    })
    return response
