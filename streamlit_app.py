import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import filterwarnings

filterwarnings("ignore")

# 1 · PAGE CONFIG & DATA LOADER (cached)

st.set_page_config("Airfare Forecast & Anomaly Detector", layout="wide", page_icon="✈️")

@st.cache_data(show_spinner=False)
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["Date_of_journey"] = pd.to_datetime(df["Date_of_journey"], format="%Y-%m-%d", errors="coerce")
    df["Route"] = df["Source"] + " ➜ " + df["Destination"]
    df = df[df["Class"].str.lower() == "economy"].copy()     # optional filter
    return df

DATA = load_data(Path("data") / "Cleaned_dataset.csv")


# 2 · SIDEBAR CONTROLS

st.sidebar.header("⚙️ Controls")
routes          = sorted(DATA["Route"].unique())
chosen_route    = st.sidebar.selectbox("Route", routes)
forecast_h      = st.sidebar.number_input("Forecast horizon (days)", 7, 90, 30)
anomaly_sigma   = st.sidebar.slider("Anomaly threshold (σ)", 1.0, 4.0, 2.5, 0.1)

# 3 · TIME-SERIES PREP

df_route = (DATA[DATA["Route"] == chosen_route]
            .groupby("Date_of_journey", as_index=False)["Fare"]
            .mean()
            .rename(columns={"Date_of_journey": "ds", "Fare": "y"}))

idx_full  = pd.date_range(df_route["ds"].min(), df_route["ds"].max(), freq="D")
df_route  = (df_route.set_index("ds")
                      .reindex(idx_full)
                      .rename_axis("ds")
                      .reset_index())
df_route["y"] = df_route["y"].interpolate()           # linear gap-fill
df_route.dropna(subset=["y"], inplace=True)

if df_route["y"].nunique() <= 1 or len(df_route) < 20:
    st.warning("Not enough variance or too few points to forecast this route.")
    st.stop()


# 4 · SARIMAX FORECAST

# Simple seasonal (weekly) SARIMAX(1,1,1)(1,1,1,7)
model   = SARIMAX(df_route["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
results = model.fit(disp=False)

pred    = results.get_forecast(steps=int(forecast_h))
conf_i  = pred.conf_int()
future_idx = pd.date_range(df_route["ds"].iloc[-1] + pd.Timedelta(days=1),
                           periods=forecast_h, freq="D")

forecast_df = pd.DataFrame({
    "ds"          : future_idx,
    "yhat"        : pred.predicted_mean.values,
    "yhat_lower"  : conf_i.iloc[:, 0].values,
    "yhat_upper"  : conf_i.iloc[:, 1].values
})

merged = pd.concat([
    df_route[["ds", "y"]].assign(yhat=np.nan, yhat_lower=np.nan, yhat_upper=np.nan),
    forecast_df.assign(y=np.nan)
]).reset_index(drop=True)


# 5 · ANOMALY DETECTION (z-score on residuals of historical part)

# Drop rows with missing actual values
hist = merged.dropna(subset=["y"]).copy()

# Compute residuals between actual values and SARIMAX fitted values
residuals = hist["y"].reset_index(drop=True) - results.fittedvalues.reset_index(drop=True)

# Compute standard deviation of residuals
sigma = residuals.std()

# Compute z-score of residuals
hist["z_score"] = residuals / sigma

# Set number of initial points to skip (to avoid false anomalies)
skip_initial = 7
anomaly_threshold = anomaly_sigma  # keep your existing threshold (e.g., 2)

# Initialize anomaly column as False
hist["anomaly"] = False

# Flag anomalies only after skipping initial rows
hist.loc[skip_initial:, "anomaly"] = hist.loc[skip_initial:, "z_score"].abs() > anomaly_threshold

# Merge the z_score and anomaly columns back into the full dataset
merged = merged.merge(hist[["ds", "z_score", "anomaly"]], on="ds", how="left")



# 6 · PLOT
fig = go.Figure()

fig.add_trace(go.Scatter(x=merged["ds"], y=merged["y"],
                         mode="lines+markers", name="Actual Fare",
                         line=dict(color="#1f77b4")))

fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"],
                         mode="lines", name="Forecast",
                         line=dict(color="#d62728", dash="dash")))

fig.add_traces([
    go.Scatter(x=merged["ds"], y=merged["yhat_upper"], mode="lines",
               line=dict(width=0), showlegend=False),
    go.Scatter(x=merged["ds"], y=merged["yhat_lower"], mode="lines",
               line=dict(width=0), fill="tonexty",
               fillcolor="rgba(214,39,40,0.1)", showlegend=True,
               name="90% CI")
])

anom_df = merged[merged["anomaly"] == True]
fig.add_trace(go.Scatter(x=anom_df["ds"], y=anom_df["y"],
                         mode="markers", marker=dict(size=9, color="red", symbol="x"),
                         name="Anomaly"))

fig.update_layout(title=f"Average Economy Fare · {chosen_route}",
                  xaxis_title="Date", yaxis_title="Fare (₹)",
                  hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02,
                              xanchor="right", x=1))


# 7 · STREAMLIT UI

st.title("✈️ Airfare Price Forecast & Anomaly Detector")

st.markdown(f"""
*Forecast horizon:* **{forecast_h} days**   |  
*Anomaly threshold:* **|z| > {anomaly_sigma}σ**

Use the sidebar to pick routes and tweak parameters.
""")

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Anomaly Details (historical only)"):
    st.dataframe(anom_df[["ds", "y", "z_score"]]
                 .rename(columns={"ds": "Date", "y": "Actual Fare", "z_score": "Z-Score"})
                 .style.format({"Actual Fare": "₹{:.0f}", "Z-Score": "{:.2f}"}))

csv_out = merged[["ds", "y", "yhat", "yhat_lower", "yhat_upper", "anomaly"]]
st.download_button("⬇️ Download Forecast CSV",
                   csv_out.to_csv(index=False),
                   file_name=f"{chosen_route.replace(' ', '_')}_forecast.csv",
                   mime="text/csv")
