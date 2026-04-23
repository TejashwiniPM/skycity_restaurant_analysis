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

st.set_page_config(layout="wide", page_title="Growth Intelligence", page_icon="🚀")

st.markdown("""
<style>

/* GLOBAL BACKGROUND */
html, body, .stApp {
    background: linear-gradient(135deg, #000000, #0F1117) !important;
    color: white !important;
}

/* MAIN CONTAINER */
.block-container {
    background-color: #000000 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
}

/* MULTISELECT FIX (THIS WAS YOUR ISSUE) */
div[data-baseweb="select"] > div {
    background-color: #111111 !important;
    border-radius: 10px !important;
    border: 1px solid #333 !important;
}

/* SELECTED TAGS */
div[data-baseweb="tag"] {
    background-color: #222 !important;
    color: white !important;
}

/* INPUT TEXT */
input {
    color: white !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background-color: #111111 !important;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #222222;
}

/* BUTTONS */
.stButton button {
    background-color: #1F1F1F !important;
    color: white !important;
}

/* DATAFRAME */
.stDataFrame {
    background-color: #000000 !important;
}

/* TABS */
.stTabs [data-baseweb="tab"] {
    color: white !important;
}

/* PLOTLY FIX */
.js-plotly-plot, .plotly, .plot-container {
    background-color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    df = pd.read_csv(DATA_FILE)
    df["TotalRevenue"] = df[["InStoreRevenue","UberEatsRevenue","DoorDashRevenue","SelfDeliveryRevenue"]].sum(axis=1)
    df["TotalNetProfit"] = df[["InStoreNetProfit","UberEatsNetProfit","DoorDashNetProfit","SelfDeliveryNetProfit"]].sum(axis=1)
    df["NetMargin"] = np.where(df["TotalRevenue"]>0, df["TotalNetProfit"]/df["TotalRevenue"],0)
    df["CostPressure"] = df["COGSRate"] + df["OPEXRate"]
    df["Scale"] = df["MonthlyOrders"] * df["GrowthFactor"]
    df["AggregatorDep"] = df["UE_share"] + df["DD_share"]
    df["RevenueQuality"] = df["AOV"] * df["NetMargin"].clip(lower=0)
    return df

@st.cache_data
def prep(df):
    for col in ["CuisineType","Segment","Subregion"]:
        df[col+"_enc"] = LabelEncoder().fit_transform(df[col])
    feats = ["GrowthFactor","AOV","MonthlyOrders","Scale","InStoreShare","UE_share","DD_share","SD_share","COGSRate","OPEXRate","CommissionRate","DeliveryRadiusKM","CostPressure","AggregatorDep","TotalNetProfit","RevenueQuality","CuisineType_enc","Segment_enc","Subregion_enc"]
    X = df[feats].fillna(0)
    X = StandardScaler().fit_transform(X)
    return df, X

@st.cache_data
def cluster(df,X):
    pca = PCA(n_components=5)
    Xp = pca.fit_transform(X)
    scores = []
    for k in range(2,7):
        km = KMeans(n_clusters=k,n_init=10,random_state=42)
        labels = km.fit_predict(Xp)
        scores.append(silhouette_score(Xp,labels))
    k = np.argmax(scores)+2
    model = KMeans(n_clusters=k,n_init=10,random_state=42)
    df["Cluster"] = model.fit_predict(Xp)
    names = ["High Growth","Stable","Aggregator Heavy","Premium","Low Performance"]
    df["ClusterLabel"] = df["Cluster"].map(lambda x: names[x%len(names)])
    p2 = PCA(n_components=2)
    coords = p2.fit_transform(X)
    df["PC1"] = coords[:,0]
    df["PC2"] = coords[:,1]
    return df

@st.cache_data
def gpi(df):
    df["dim_scale"] = df["Scale"]
    df["dim_cost"] = 1 - df["CostPressure"]
    df["dim_quality"] = df["RevenueQuality"]
    df["dim_balance"] = 1 - np.abs(df["AggregatorDep"]-0.5)
    df["dim_logistics"] = df["DeliveryRadiusKM"] * (1 + df["SD_share"])
    dims = ["dim_scale","dim_cost","dim_quality","dim_balance","dim_logistics"]
    df[dims] = MinMaxScaler().fit_transform(df[dims])
    df["GPI"] = (df["dim_scale"]*0.3 + df["dim_cost"]*0.25 + df["dim_quality"]*0.2 + df["dim_balance"]*0.15 + df["dim_logistics"]*0.1)*100
    df["Recommendation"] = df["GPI"].apply(lambda x: "🚀 Scale" if x>70 else "⚖️ Improve" if x>40 else "🛑 Fix")
    return df

def app():
    if not os.path.exists(DATA_FILE):
        st.error("CSV missing")
        return

    df = load()
    df,X = prep(df)
    df = cluster(df,X)
    df = gpi(df)

    st.title("🚀 Growth Intelligence Dashboard")

    sub = st.sidebar.multiselect("Subregion", df["Subregion"].unique(), df["Subregion"].unique())
    cui = st.sidebar.multiselect("Cuisine", df["CuisineType"].unique(), df["CuisineType"].unique())

    f = df[df["Subregion"].isin(sub) & df["CuisineType"].isin(cui)]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Restaurants", len(f))
    c2.metric("Avg GPI", f"{f['GPI'].mean():.1f}")
    c3.metric("Margin", f"{f['NetMargin'].mean():.2%}")
    c4.metric("Top Performers", (f["GPI"]>70).sum())

    fig = px.scatter(f,x="PC1",y="PC2",color="ClusterLabel",size="GPI",template="plotly_dark")
    fig.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000",font=dict(color="white"))
    st.plotly_chart(fig,use_container_width=True)

if __name__ == "__main__":
    app()
