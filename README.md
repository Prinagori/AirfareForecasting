# Airfare Forecast & Anomaly Detection using Time Series Forecasting

This Streamlit project focuses on forecasting flight fares and detecting unusual price patterns. It leverages Python-based time series modeling with **SARIMAX**, applies **z-score-based anomaly detection**, and visualizes results in an interactive web app.

Link to the app - https://airfareforecasting.streamlit.app/

---

## Problem Statement

Can we build an interactive system to:
- Forecast flight ticket prices using historical trends?
- Detect unusual spikes or drops in fare based on statistical anomalies?
- Help users gain insights on when fares deviate from expected behavior?

---

## Dataset

- **Source**: Cleaned dataset containing fields like `Date_of_Journey`, `Source`, `Destination`, `Price`, `Class`, etc.
- Focused only on **Economy class** fares for simplicity and consistency.
- Created a custom `Route` field by combining source and destination.

Sample Preprocessing:
- Parsed `Date_of_Journey` as datetime
- Generated `Route` as `Source ➜ Destination`
- Filtered out rows with missing or non-economy class values

---

## Preprocessing & Modeling Steps

### 1. Preprocessing
- Converted journey date to datetime
- Filtered to include only "Economy" class
- Grouped data by route and date for modeling

### 2. Time Series Forecasting
- Used **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)  
- Captured trends and seasonal patterns in fare data

### 3. Anomaly Detection
- Calculated **residuals** between actual and predicted prices
- Computed **z-scores** of residuals
- Flagged values where `|z| > threshold` as anomalies

---

## Features of the Streamlit App

- Route and date filters in sidebar
- Forecast chart with historical vs predicted prices
- Anomalies clearly highlighted on the timeline
- Interactive, responsive layout using Plotly charts
- Fast data loading with caching

---

## What I Learned

- How to structure time series problems for real-world datasets
- Implementing SARIMAX for multi-seasonality and trend forecasting
- Using z-score thresholds for anomaly detection
- Streamlit best practices for caching, layout, and filters
- Presenting model predictions in a user-friendly dashboard

---

## Future Enhancements

- Add more advanced models (e.g., Prophet, LSTM)
- Include additional features like holidays or booking time window
- Compare multiple routes side-by-side
- Support uploading custom CSVs
- Deploy the app publicly with auto-refresh for new data
- Integrate alert system for anomalies (email, notifications)

---

## Tech Stack

- **Python** (pandas, numpy, statsmodels, plotly, streamlit)
- **Time Series Modeling**: SARIMAX
- **Anomaly Detection**: Z-score based outlier detection
- **App Framework**: Streamlit

---

## How to Run This Project

```bash
# 1. Clone the repo
git clone https://github.com/your-username/airfare-forecast-app.git
cd airfare-forecast-app

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run streamlit_app.py
