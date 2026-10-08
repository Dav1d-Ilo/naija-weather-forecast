# Naija Weather Forecast

Machine learning for predicting **tomorrow's weather in four Nigerian cities**.

This project uses historical daily weather data from Lagos, Port Harcourt, Abuja and Kano to answer two questions:

* 🌧️ **Will it rain tomorrow?**
* 🌡️ **What will tomorrow's maximum temperature be?**

The models are evaluated against simple forecasting baselines and deployed through an interactive **Streamlit dashboard**.

## Try it
**[naija-weather-forecast.streamlit.app](https://naija-weather-forecast.streamlit.app)**

If the page shows a "wake up" button, click it and wait a few seconds. Free apps go to sleep when nobody has used them for a while.

---

## 🌍 Cities

The project currently covers four Nigerian cities:

* **Lagos** — humid coastal climate
* **Port Harcourt** — very wet southern climate
* **Abuja** — central / Middle Belt
* **Kano** — hot and relatively dry northern climate

Using cities from different parts of Nigeria lets the project examine how forecasting performance changes across different weather patterns.

---

## 🎯 What This Project Does

The project treats weather forecasting as two separate machine learning problems.

### Rain Prediction

A classification model predicts whether tomorrow will receive **at least 1 mm of rain**.

### Maximum Temperature Prediction

A regression model predicts tomorrow's **maximum temperature in °C**.

The important part isn't simply getting a high score. The project asks whether machine learning actually improves on simple forecasting rules and which information contributes to the predictions.

---

## 📊 Data

Weather data comes from the **Open-Meteo Historical Weather API**.

The dataset covers **January 2000 through June 2026** for the four cities.

The available variables include:

* Maximum temperature
* Minimum temperature
* Mean temperature
* Precipitation
* Maximum wind speed
* Relative humidity
* Surface pressure
* Cloud cover
* Dew point
* Solar radiation

A day is classified as a **rainy day when precipitation is at least 1 mm**.

---

## 🧹 Data Preparation

Before modelling, the data is checked and prepared by:

* Removing duplicate city/date records
* Checking for missing days
* Checking for invalid weather values
* Creating continuous daily timelines
* Interpolating only small gaps where appropriate
* Keeping larger gaps rather than inventing weather data

In this dataset, the checks found no missing days or invalid values. The downloaded data is also cached so that completed API requests do not need to be downloaded again.

---

## 🧠 Method

### Targets

The models predict:

* **Rain tomorrow:** whether tomorrow receives at least 1 mm of rain
* **Maximum temperature tomorrow:** tomorrow's maximum temperature

### Inputs

The models use information available **by the end of today**, including:

* Today's weather
* Weather from 1, 2, 3 and 7 days ago
* 3, 7 and 30-day averages
* Number of rainy days during the previous week
* Days since the last rain
* Daily pressure change
* Time of year
* City

This prevents future information from leaking into the model inputs.

### Time-based evaluation

Because this is a forecasting problem, the data is split chronologically:

```text
2000–2017       Training
2018–2021       Validation / tuning
2022–June 2026  Test
```

The test period contains years the models never saw during training or tuning.

Every model is also compared with simple baselines.

---

## 🤖 Models

The project compares several approaches.

### Rain

* Logistic Regression
* Random Forest
* XGBoost
* Tuned XGBoost

### Maximum Temperature

* Linear Regression
* Random Forest
* XGBoost
* Tuned XGBoost

The XGBoost models are tuned using **time-aware cross-validation** rather than randomly mixing observations from different points in time.

**"XGBoost (final)"** in the results below is the tuned XGBoost model retrained on the 2000–2021 data (training plus validation years), with its rain probabilities corrected using a calibration step learned on the validation years. It is still scored on the untouched test years.

---

## 🏁 Baselines

A machine learning model is only useful if it can beat simple alternatives.

### Rain

Two baselines are used:

**Usual rain for that city and month**

Uses the historical rain pattern for the same city and month.

**Same as today**

Assumes tomorrow's rain status will be the same as today's.

### Temperature

Two baselines are used:

**Same as today**

Assumes tomorrow's maximum temperature will equal today's.

**Usual temperature for that city and month**

Uses the historical maximum temperature for the same city and month.

---

## 📈 Results

The final results below are from the **unseen test period: 2022 to June 2026**.

### 🌧️ Rain Tomorrow

| Model                                        |     AUC ↑ | Brier Error ↓ | Days Called Correctly |
| -------------------------------------------- | --------: | ------------: | --------------------: |
| Baseline: usual rain for that city and month |     0.886 |         0.146 |                 78.8% |
| Baseline: same as today                      |     0.780 |         0.218 |                 78.2% |
| **XGBoost (final)**                          | **0.910** |     **0.121** |             **81.8%** |

The final XGBoost model improves on both simple baselines, although the improvement is modest because seasonal patterns already explain a large part of the rainfall pattern.

### 🌡️ Maximum Temperature Tomorrow

| Model                                               |       MAE ↓ |      R² ↑ |
| --------------------------------------------------- | ----------: | --------: |
| Baseline: same as today                             |     1.08 °C |     0.798 |
| Baseline: usual temperature for that city and month |     1.51 °C |     0.661 |
| **XGBoost (final)**                                 | **0.91 °C** | **0.865** |

The final model's average error is about **0.91 °C** on the unseen test period.

---

## 🔎 What I Found

* Both final models beat their simple baselines on unseen years, but the gains are **modest**.
* The rain model's improvement over each city's own climatology baseline was largest in **Port Harcourt (~28% lower error)**, followed by **Abuja (~20%)**, **Lagos (~10%)**, and **Kano (~7.5%)**.
* The rain model relies heavily on **recent rainfall and recent weather conditions**.
* The temperature model relies heavily on **recent temperature**, which makes sense given the strong persistence of daily temperatures.
* Logistic Regression came within about **0.007 AUC** of XGBoost for rain prediction, suggesting that the extra complexity of XGBoost provides only a relatively small improvement.
* Calibrating the rain probabilities made almost no difference to the results.
* The results show that building a more complicated model does not automatically produce a dramatically better forecast.

---

## 🔬 Feature Experiments

The project also tests how much different types of information contribute to the forecasts.

The same XGBoost approach is trained with progressively more information:

| Experiment | Information                      | Rain AUC | Temperature MAE |
| ---------- | -------------------------------- | -------: | --------------: |
| **A**      | Today's weather                  |    0.897 |        1.010 °C |
| **B**      | Today's weather + recent history |    0.905 |        0.926 °C |
| **C**      | + time of year                   |    0.907 |        0.925 °C |
| **D**      | + city                           |    0.908 |        0.922 °C |

This helps answer a more interesting question than simply asking which model has the highest score:

> **What information actually makes tomorrow's forecast better?**

* **Temperature:** recent history was the one clear improvement (average error from 1.010 °C to 0.926 °C). Adding the time of year and the city made almost no difference.
* **Rain:** today's weather alone already reached an AUC of 0.897, and everything added on top only lifted it to 0.908. Differences this small are within the noise of a single test period.

---

## 🖥️ Streamlit Dashboard

The trained models are deployed through an interactive **Streamlit dashboard** with four tabs:

* **Tomorrow:** pick a city and see the chance of rain, the expected maximum temperature and the last two weeks of weather
* **Compare cities:** all four cities side by side
* **What-if explorer:** change recent rain, temperature and humidity and see how the predictions react
* **About this model:** how it works and how well it performs

The workflow is:

```text
Select Nigerian city
        ↓
Get recent weather data
        ↓
Build forecasting features
        ↓
Load trained models
        ↓
Predict tomorrow's weather
        ↓
🌧️ Chance of rain
🌡️ Expected maximum temperature
```

The dashboard uses the same feature-building logic used during model development through `weather_features.py`.

The live dashboard uses **Open-Meteo's forecast service**, while the models were trained using data from Open-Meteo's historical archive. The two sources are similar but not identical, so live predictions may differ slightly from what would be produced using historical archive data alone. Today's values are also partly forecast, because the day has not finished.

---

## ⚠️ Limitations

* **Modelled data:** the weather comes from Open-Meteo's reanalysis archive, not direct weather-station measurements.
* **Coverage:** one grid point for each of four cities. Rain can vary a lot within a city, and the results do not automatically apply to the rest of Nigeria.
* **Test period:** about four and a half years, so small differences between models are within noise.
* **Live mode:** the models were trained on the historical archive, while the dashboard uses the forecast service, and today's values are partly forecast. They are similar, not identical.
* **Not an official forecast:** this is a learning project built on simple daily features, not a competitor to operational weather forecasting.

---

## 📁 Project Files

| File                       | Purpose                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `nigeria_weather_ml.ipynb` | Full analysis: data download, cleaning, exploration, feature engineering, modelling and evaluation |
| `app.py`                   | Streamlit dashboard                                                                                |
| `weather_features.py`      | Feature-building code shared by the dashboard                                                      |
| `rain_model.joblib`        | Saved final rain prediction model and configuration                                                |
| `temp_model.joblib`        | Saved final maximum-temperature model and configuration                                            |
| `typical_values.csv`       | Typical weather values used by the dashboard                                                       |
| `requirements.txt`         | Python dependencies and package versions                                                           |
| `weather_*.csv`            | Historical weather data, one file for each city (Lagos, Port Harcourt, Abuja, Kano)                |

---

## 📌 Why I Built It

I wanted to explore a practical question:

> **How much can historical weather data tell us about tomorrow's weather in different parts of Nigeria?**

Instead of stopping at model accuracy, I wanted to test whether machine learning actually adds value over simple baselines, prevent future-data leakage, and investigate which information matters.

The result is both a **machine learning experiment** and a usable forecasting application.

---

## 📚 Data Source

Weather data is provided by [Open-Meteo.com](https://open-meteo.com/) through its historical weather archive and forecast services, under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
