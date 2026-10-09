import joblib
import pandas as pd
import streamlit as st

from weather_features import (fetch_recent_weather, make_feature_row, make_whatif_row,
                              predict_tomorrow)

st.set_page_config(page_title="Naija Weather Forecast", page_icon="🌦️", layout="centered")

# Edit these if you re-run the notebook and your test results change (they come from the test years).
TEST_RESULTS = pd.DataFrame(
    [["Rain: ranking quality (AUC, higher is better)", "0.910", "0.886"],
     ["Rain: probability error (Brier, lower is better)", "0.121", "0.146"],
     ["Rain: days called correctly", "81.8%", "78.8%"],
     ["Temperature: average miss", "0.91 °C", "1.08 °C"]],
    columns=["Measure (test years: 2022 to June 2026)", "This model", "Simple baseline"])

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 760px;}
    h1 {margin-bottom: 0.2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    rain = joblib.load("rain_model.joblib")
    temp = joblib.load("temp_model.joblib")
    typical = pd.read_csv("typical_values.csv")
    return rain, temp, typical


@st.cache_data(ttl=3600, show_spinner=False)
def get_recent(lat, lon, variables):
    # cached for an hour so repeated clicks do not hammer the free API
    return fetch_recent_weather(lat, lon, list(variables))


try:
    rain_bundle, temp_bundle, typical = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put rain_model.joblib, temp_model.joblib and "
             "typical_values.csv in the same folder as the app (the notebook saves them).")
    st.stop()

FEATURES = rain_bundle["features"]
CITIES = rain_bundle["cities"]            # {"Lagos": (lat, lon), ...}
WET_MM = rain_bundle["wet_day_mm"]
TODAY_VARS = tuple(rain_bundle["today_vars"])


# ----------------------------------------------------------------------------- helpers
def forecast_for(city):
    lat, lon = CITIES[city]
    recent = get_recent(lat, lon, TODAY_VARS)
    X, last_day = make_feature_row(recent, city, list(CITIES), FEATURES, WET_MM)
    prob, temp = predict_tomorrow(X, rain_bundle, temp_bundle)
    return {"prob": prob, "temp": temp, "day": last_day + pd.Timedelta(days=1), "recent": recent}


def condition(p):
    """Emoji, headline, two gradient colours and a one-line tip for a rain probability."""
    if p >= 0.75:
        return "🌧️", "Rain very likely", "#1d4ed8", "#3b82f6", "Take an umbrella."
    if p >= 0.50:
        return "🌦️", "Rain likely", "#2563eb", "#60a5fa", "Pack an umbrella."
    if p >= 0.25:
        return "⛅", "Rain possible", "#475569", "#94a3b8", "Maybe keep an umbrella handy."
    return "☀️", "Probably dry", "#d97706", "#fbbf24", "You can probably leave the umbrella at home."


def hero(subtitle, p, t):
    emoji, label, c1, c2, tip = condition(p)
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{c1},{c2});border-radius:18px;
                    padding:22px 26px;color:white;">
          <div style="opacity:.9;font-size:.95rem;">{subtitle}</div>
          <div style="display:flex;align-items:center;gap:18px;margin-top:6px;">
            <div style="font-size:3.6rem;line-height:1;">{emoji}</div>
            <div>
              <div style="font-size:1.7rem;font-weight:700;">{label}</div>
              <div style="opacity:.95;">{p * 100:.0f}% chance of rain &middot; about {t:.0f} °C</div>
            </div>
          </div>
          <div style="margin-top:12px;opacity:.95;">{tip}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def usual_for(city, month):
    """Typical max temperature and share of rainy days for a city and month (from the training data)."""
    row = typical[(typical["location"] == city) & (typical["month"] == month)]
    if row.empty:
        return None, None
    return float(row["temperature_2m_max"].iloc[0]), float(row["wet_days_7"].iloc[0]) / 7


# ----------------------------------------------------------------------------- header
st.title("🌦️ Naija Weather Forecast")
st.caption("Tomorrow's chance of rain and maximum temperature for four Nigerian cities, "
           "from a machine-learning model trained on daily weather from 2000 to "
           f"{rain_bundle['data_end'][:7]}.")

tab_live, tab_all, tab_whatif, tab_about = st.tabs(
    ["Tomorrow", "Compare cities", "What-if explorer", "About this model"])

# ----------------------------------------------------------------------------- TOMORROW
with tab_live:
    city = st.radio("City", list(CITIES), horizontal=True, key="live_city")
    try:
        with st.spinner("Getting the latest weather..."):
            f = forecast_for(city)
    except Exception as e:
        st.error("Could not get or process the latest weather, so there is no prediction right now. "
                 "Check your internet connection and try again.")
        st.caption(f"Technical details: {e}")
        if st.button("Try again"):
            get_recent.clear()
    else:
        hero(f"{city} · {f['day']:%A %d %B %Y}", f["prob"], f["temp"])
        st.progress(min(max(f["prob"], 0.0), 1.0), text=f"Chance of rain: {f['prob'] * 100:.0f}%")

        usual_t, rainy_share = usual_for(city, f["day"].month)
        if usual_t is not None:
            diff = f["temp"] - usual_t
            word = "above" if diff >= 0 else "below"
            st.caption(f"A typical {f['day']:%B} day in {city} reaches {usual_t:.1f} °C and is rainy on about "
                       f"{rainy_share * 100:.0f}% of days. Tomorrow is expected to be {abs(diff):.1f} °C "
                       f"{word} the usual.")

        st.markdown("**The last two weeks**")
        recent = f["recent"].tail(14).set_index("date")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Rain (mm per day)")
            st.bar_chart(recent["precipitation_sum"])
        with c2:
            st.caption("Max temperature (°C)")
            st.line_chart(recent["temperature_2m_max"])

        with st.expander("The recent weather the model used"):
            st.dataframe(f["recent"].tail(7).set_index("date"))
        st.caption("The last row is today. Values for the rest of today come from the forecast model, "
                   "not from finished observations. A 'rainy day' means at least 1 mm of rain.")
        if st.button("Refresh the weather data"):
            get_recent.clear()

# ----------------------------------------------------------------------------- COMPARE
with tab_all:
    st.subheader("All four cities at a glance")
    results = {}
    with st.spinner("Checking all four cities..."):
        for c in CITIES:
            try:
                results[c] = forecast_for(c)
            except Exception as e:
                st.warning(f"{c}: could not get the latest weather ({e})")
    if results:
        cols = st.columns(len(results))
        for col, (c, f) in zip(cols, results.items()):
            emoji, label, *_ = condition(f["prob"])
            with col:
                with st.container(border=True):
                    st.markdown(f"### {emoji} {c}")
                    st.metric("Chance of rain", f"{f['prob'] * 100:.0f}%")
                    st.metric("Max temp", f"{f['temp']:.0f} °C")
                    st.caption(label)
        table = pd.DataFrame({
            "City": list(results),
            "Chance of rain": [f"{f['prob'] * 100:.0f}%" for f in results.values()],
            "Expected max temp": [f"{f['temp']:.1f} °C" for f in results.values()],
            "Outlook": [condition(f["prob"])[1] for f in results.values()],
        })
        st.dataframe(table, hide_index=True)

# ----------------------------------------------------------------------------- WHAT-IF
with tab_whatif:
    st.subheader("What if the weather looked like this?")
    st.caption("Pick a city and month, choose a starting point, then adjust the sliders. Everything you "
               "don't set is filled in with the typical values for that city and month. This is for "
               "exploring how the model reacts.")

    a, b_ = st.columns(2)
    w_city = a.selectbox("City", list(CITIES), key="w_city")
    w_month = b_.selectbox("Month", list(range(1, 13)), index=5,
                           format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"), key="w_month")
    scenario = st.radio("Start from", ["Typical", "Dry spell", "Wet week"], horizontal=True, key="scenario")

    base = make_whatif_row(typical, w_city, w_month, FEATURES, list(CITIES), {}).iloc[0]

    def b(col, default=0.0):
        return float(base[col]) if col in base.index else default

    rain_def = {"Dry spell": 0.0, "Typical": round(b("precipitation_sum"), 1), "Wet week": 15.0}[scenario]
    rain3_def = {"Dry spell": 0.0, "Typical": round(b("rain_mean3"), 1), "Wet week": 12.0}[scenario]
    wet7_def = {"Dry spell": 0, "Typical": int(round(b("wet_days_7"))), "Wet week": 6}[scenario]
    tmax_def = {"Dry spell": b("temperature_2m_max") + 1, "Typical": b("temperature_2m_max"),
                "Wet week": b("temperature_2m_max") - 2}[scenario]
    hum_def = {"Dry spell": b("relative_humidity_2m_mean") - 10, "Typical": b("relative_humidity_2m_mean"),
               "Wet week": b("relative_humidity_2m_mean") + 8}[scenario]

    key = f"{scenario}_{w_city}_{w_month}"      # changing scenario/city/month resets the sliders
    left, right = st.columns([3, 2])
    with left:
        rain_today = st.slider("Rain today (mm)", 0.0, 80.0, float(min(max(rain_def, 0.0), 80.0)), 0.5,
                               key="r1" + key)
        rain3 = st.slider("Average daily rain, last 3 days (mm)", 0.0, 60.0,
                          float(min(max(rain3_def, 0.0), 60.0)), 0.5, key="r3" + key)
        wet7 = st.slider("Rainy days in the last 7 days", 0, 7, int(min(max(wet7_def, 0), 7)), key="w7" + key)
        tmax = st.slider("Today's max temperature (°C)", 20.0, 42.0,
                         float(min(max(tmax_def, 20.0), 42.0)), 0.5, key="t1" + key)
        tmax3 = st.slider("Average max temperature, last 3 days (°C)", 20.0, 42.0,
                          float(min(max(tmax_def, 20.0), 42.0)), 0.5, key="t3" + key)
        overrides = {"precipitation_sum": rain_today, "rain_mean3": rain3, "wet_days_7": wet7,
                     "temperature_2m_max": tmax, "tmax_mean3": tmax3, "tmax_mean7": tmax3}
        if "relative_humidity_2m_mean" in FEATURES:
            humidity = st.slider("Humidity (%)", 0.0, 100.0, float(min(max(hum_def, 0.0), 100.0)), 1.0,
                                 key="h" + key)
            overrides["relative_humidity_2m_mean"] = humidity

    base_dsr = b("days_since_rain", 10.0)
    overrides["days_since_rain"] = 0.0 if rain_today >= WET_MM else (base_dsr if wet7 == 0 else min(base_dsr, 3.0))

    X = make_whatif_row(typical, w_city, w_month, FEATURES, list(CITIES), overrides)
    prob, temp = predict_tomorrow(X, rain_bundle, temp_bundle)
    with right:
        st.markdown("**Prediction for tomorrow**")
        hero(f"{w_city} · {pd.Timestamp(2000, w_month, 1):%B}", prob, temp)
        st.caption("Some combinations might not make much sense, like heavy rain today even though it hasn't rained all week."
                   
                   "The model wasn't trained on many situations like these, so predictions for unusual combinations may be less reliable.")

# ----------------------------------------------------------------------------- ABOUT
with tab_about:
    st.subheader("How good is it?")
    st.markdown("Scored on years the model never saw while learning (2022 to June 2026), against very simple "
                "baselines. For rain the baseline is the usual rain chance for that city and month. For "
                "temperature it is 'tomorrow will be as hot as today'.")
    st.dataframe(TEST_RESULTS, hide_index=True)
    st.markdown(
        "**In plain words:** it beats the simple baselines, but by a modest margin. The seasons already "
        "explain most of the rain pattern, and temperature barely changes from one day to the next, so "
        "there isn't a lot left to improve. It does best in Port Harcourt and Abuja and gains least in "
        "Kano."
    )
    st.subheader("How it works")
    st.markdown(
        "- **Data:** daily weather from Open-Meteo for Lagos, Port Harcourt, Abuja and Kano, "
        "January 2000 to June 2026. It is modelled (reanalysis) data, not raw station readings.\n"
        "- **Models:** XGBoost for the chance of rain and for tomorrow's maximum temperature.\n"
        "- **Inputs:** today's weather, the last few days, how long since it rained, the time of year "
        "and the city. Recent rain and recent temperatures matter most.\n"
        "- **Live data caveat:** live weather comes from Open-Meteo's forecast service, while the "
        "models were trained on its historical archive. The two are very similar but not identical.\n"
    )

st.divider()
st.caption("Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).")
