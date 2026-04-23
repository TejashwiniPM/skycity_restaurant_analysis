import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_FILE = "SkyCity_Auckland_Restaurants___Bars.csv"

def safe_col(df, col, default=0):
    return df[col] if col in df.columns else default

def safe_log(series):
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    shift = abs(series.min()) + 1 if series.min() < 0 else 0
    return np.log1p(series + shift)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    revenue_cols = [c for c in df.columns if "Revenue" in c]
    profit_cols = [c for c in df.columns if "NetProfit" in c]

    df["TotalRevenue"] = df[revenue_cols].sum(axis=1) if revenue_cols else 0
    df["TotalNetProfit"] = df[profit_cols].sum(axis=1) if profit_cols else 0

    df["NetMargin"] = np.where(df["TotalRevenue"] != 0,
                              df["TotalNetProfit"] / df["TotalRevenue"], 0)

    df["COGSRate"] = safe_col(df, "COGSRate")
    df["OPEXRate"] = safe_col(df, "OPEXRate")
    df["CostPressure"] = df["COGSRate"] + df["OPEXRate"]

    return df

@st.cache_data
def preprocess(df):
    df = df.copy()

    df["MonthlyOrders"] = safe_col(df, "MonthlyOrders")
    df["GrowthFactor"] = safe_col(df, "GrowthFactor", 1)
    df["AOV"] = safe_col(df, "AOV")

    df["UE_share"] = safe_col(df, "UE_share")
    df["DD_share"] = safe_col(df, "DD_share")
    df["SD_share"] = safe_col(df, "SD_share")

    df["Scale"] = df["MonthlyOrders"] * df["GrowthFactor"]
    df["AggregatorDep"] = df["UE_share"] + df["DD_share"]
    df["RevenueQuality"] = df["AOV"] * df["NetMargin"].clip(lower=0)

    for col in ["CuisineType", "Segment", "Subregion"]:
        if col in df.columns:
            df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df[col + "_enc"] = 0

    for col in ["MonthlyOrders", "Scale", "TotalNetProfit"]:
        df[col] = safe_log(df[col])

    features = [
        "GrowthFactor", "AOV", "MonthlyOrders", "Scale",
        "NetMargin", "TotalNetProfit", "CostPressure",
        "AggregatorDep", "RevenueQuality",
        "CuisineType_enc", "Segment_enc", "Subregion_enc"
    ]

    X = df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled

@st.cache_data
def cluster_data(df, X):
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X)

    scores = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        scores.append(silhouette_score(X_pca, labels))

    best_k = np.argmax(scores) + 2

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_pca)

    cluster_means = df.groupby("Cluster")[["GrowthFactor", "NetMargin", "CostPressure"]].mean()

    labels = {}
    for c in cluster_means.index:
        gf = cluster_means.loc[c, "GrowthFactor"]
        nm = cluster_means.loc[c, "NetMargin"]
        cp = cluster_means.loc[c, "CostPressure"]

        if gf > df["GrowthFactor"].mean() and nm > df["NetMargin"].mean():
            labels[c] = "High Growth"
        elif nm > 0:
            labels[c] = "Stable"
        elif cp > df["CostPressure"].mean():
            labels[c] = "High Cost"
        else:
            labels[c] = "Low Performance"

    df["ClusterLabel"] = df["Cluster"].map(labels)

    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X)

    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    return df

@st.cache_data
def calculate_gpi(df):
    df = df.copy()

    df["dim_scale"] = df["MonthlyOrders"] * df["GrowthFactor"]
    df["dim_cost"] = 1 - df["CostPressure"]
    df["dim_quality"] = df["RevenueQuality"]
    df["dim_balance"] = 1 - np.abs(df["AggregatorDep"] - 0.5)

    df["DeliveryRadiusKM"] = safe_col(df, "DeliveryRadiusKM")
    df["dim_logistics"] = df["DeliveryRadiusKM"] * (1 + df["SD_share"])

    dims = ["dim_scale", "dim_cost", "dim_quality", "dim_balance", "dim_logistics"]

    scaler = MinMaxScaler()
    df[dims] = scaler.fit_transform(df[dims].fillna(0))

    df["GPI"] = (
        df["dim_scale"] * 0.3 +
        df["dim_cost"] * 0.25 +
        df["dim_quality"] * 0.2 +
        df["dim_balance"] * 0.15 +
        df["dim_logistics"] * 0.1
    ) * 100

    df["GPI_Tier"] = pd.cut(df["GPI"],
                           bins=[-1, 30, 50, 70, 100],
                           labels=["Risk", "Low", "Moderate", "High"])

    def rec(x):
        if x >= 60:
            return "Scale"
        elif x >= 40:
            return "Improve"
        else:
            return "Fix"

    df["Recommendation"] = df["GPI"].apply(rec)

    return df

def run_dashboard(df):
    st.set_page_config(layout="wide")
    st.title("🍽️ Restaurant Growth Intelligence")

    subregions = st.sidebar.multiselect("Subregion", df.get("Subregion", []), df.get("Subregion", []))
    cuisines = st.sidebar.multiselect("Cuisine", df.get("CuisineType", []), df.get("CuisineType", []))

    fdf = df.copy()
    if "Subregion" in df.columns:
        fdf = fdf[fdf["Subregion"].isin(subregions)]
    if "CuisineType" in df.columns:
        fdf = fdf[fdf["CuisineType"].isin(cuisines)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Restaurants", len(fdf))
    c2.metric("Avg GPI", f"{fdf['GPI'].mean():.1f}")
    c3.metric("Avg Margin", f"{fdf['NetMargin'].mean():.2%}")

    fig1 = px.scatter(fdf, x="PC1", y="PC2", color="ClusterLabel")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(fdf, x="GPI", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(fdf, x="GPI", y="NetMargin", color="ClusterLabel")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(fdf.head(50), use_container_width=True)

    st.download_button("Download CSV", fdf.to_csv(index=False), "output.csv")

def main():
    if not os.path.exists(DATA_FILE):
        st.error("CSV file not found")
        return

    df = load_data()
    df, X = preprocess(df)
    df = cluster_data(df, X)
    df = calculate_gpi(df)
    run_dashboard(df)

if __name__ == "__main__":
    main()
