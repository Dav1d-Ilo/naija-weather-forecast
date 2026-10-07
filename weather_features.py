"""
Feature building and prediction helpers for the Nigeria weather dashboard.

IMPORTANT: build_features() is a copy of the function in the notebook. If you change one,
change the other, or the dashboard will feed the models inputs they were not trained on.
"""
import numpy as np
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_DAYS = 70          # enough for the 30-day averages and the 60-day "days since rain" cap


def build_features(g):
    """Same logic as the notebook: lags, rolling averages, rain history, season, and tomorrow's targets."""
    g = g.sort_values("date").copy()
    p, tmax = g["precipitation_sum"], g["temperature_2m_max"]

    for k in [1, 2, 3, 7]:
        g[f"rain_lag{k}"] = p.shift(k)
        g[f"tmax_lag{k}"] = tmax.shift(k)

    for w in [3, 7, 30]:
        g[f"rain_mean{w}"] = p.rolling(w).mean()
        g[f"tmax_mean{w}"] = tmax.rolling(w).mean()

    g["wet_days_7"] = g["wet"].rolling(7).sum()
    last_wet = g["date"].where(g["wet"] == 1).ffill()
    g["days_since_rain"] = (g["date"] - last_wet).dt.days.fillna(365).clip(upper=60)

    if "surface_pressure_mean" in g.columns:
        g["pressure_change"] = g["surface_pressure_mean"].diff()

    doy = g["date"].dt.dayofyear
    g["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    g["rain_tomorrow"] = g["wet"].shift(-1)
    g["tmax_tomorrow"] = tmax.shift(-1)
    return g


def fetch_recent_weather(lat, lon, variables, days=HISTORY_DAYS):
    """Last `days` days of daily weather up to and including today, from the Open-Meteo forecast API."""
    params = dict(latitude=lat, longitude=lon, daily=",".join(variables),
                  past_days=days, forecast_days=1, timezone="Africa/Lagos")
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_feature_row(daily, city, cities, features, wet_day_mm):
    """Turn recent daily weather into the single row of model inputs for 'today' (the last day)."""
    g = daily.copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").drop_duplicates("date").set_index("date").asfreq("D")
    num = g.select_dtypes("number").columns
    g[num] = g[num].interpolate(limit=2)
    g = g.reset_index()

    g["city"] = city
    g["month"] = g["date"].dt.month
    g["wet"] = (g["precipitation_sum"] >= wet_day_mm).astype(float).where(g["precipitation_sum"].notna())
    g = build_features(g)
    for c in cities:
        g[f"city_{c}"] = int(c == city)

    last = g.iloc[[-1]]
    X = last.reindex(columns=features)
    missing = X.columns[X.isna().any()].tolist()
    if missing:
        raise ValueError("Some inputs could not be computed from the recent data: " + ", ".join(missing))
    return X.astype(float), last["date"].iloc[0]


def predict_tomorrow(X, rain_bundle, temp_bundle):
    """Return (calibrated rain probability, expected max temperature in C)."""
    raw = rain_bundle["model"].predict_proba(X)[:, 1]
    prob = float(rain_bundle["calibrator"].predict(raw)[0])
    temp = float(temp_bundle["model"].predict(X)[0])
    return prob, temp


def make_whatif_row(typical, city, month, features, cities, overrides):
    """Start from the typical inputs for this city and month, then apply the user's choices."""
    sel = typical[(typical["location"] == city) & (typical["month"] == month)]
    if sel.empty:
        raise ValueError(f"No typical values for {city}, month {month}")
    X = sel.iloc[[0]].reindex(columns=features).astype(float).reset_index(drop=True)

    for c in cities:
        X[f"city_{c}"] = float(c == city)
    mid_doy = int(round((month - 0.5) * 30.4375))
    X["doy_sin"] = np.sin(2 * np.pi * mid_doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * mid_doy / 365.25)
    X["month"] = float(month)

    for k, v in overrides.items():
        if k in X.columns:
            X[k] = float(v)
    return X
