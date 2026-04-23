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

.stApp {
    background: radial-gradient(circle at top, #0A0F1C, #000000);
    color: white;
}

[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(10px);
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 15px rgba(0,255,255,0.15);
}

.stTabs [data-baseweb="tab"] {
    color: #00F5FF;
    font-weight: 600;
}

.stButton button {
    background: linear-gradient(135deg, #00F5FF, #7A00FF);
    color: white;
    border-radius: 10px;
    border: none;
}

.js-plotly-plot {
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

def insights(df):
    out = []
    if (df["GPI"]>70).sum()>0:
        out.append(f"🚀 {(df['GPI']>70).sum()} high potential restaurants")
    if (df["GPI"]<40).sum()>0:
        out.append(f"🛑 {(df['GPI']<40).sum()} at risk")
    if df["AggregatorDep"].mean()>0.6:
        out.append("⚠️ Aggregator dependency high")
    if df["NetMargin"].mean()<0.1:
        out.append("📉 Low profitability")
    return out

def forecast(df):
    df["NextMonthOrders"] = df["MonthlyOrders"] * df["GrowthFactor"]
    return df

def app():
    if not os.path.exists(DATA_FILE):
        st.error("CSV file missing")
        return

    df = load()
    df,X = prep(df)
    df = cluster(df,X)
    df = gpi(df)
    df = forecast(df)

    st.title("🚀 Growth Intelligence Dashboard")
    st.caption("AI • ML • Strategy • Forecasting")

    sub = st.sidebar.multiselect("Subregion", df["Subregion"].unique(), df["Subregion"].unique())
    cui = st.sidebar.multiselect("Cuisine", df["CuisineType"].unique(), df["CuisineType"].unique())

    f = df[df["Subregion"].isin(sub) & df["CuisineType"].isin(cui)]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Restaurants", len(f))
    c2.metric("Avg GPI", f"{f['GPI'].mean():.1f}")
    c3.metric("Margin", f"{f['NetMargin'].mean():.2%}")
    c4.metric("Top Performers", (f["GPI"]>70).sum())

    st.markdown("### 🤖 AI Insights")
    for i in insights(f):
        st.success(i)

    tab1,tab2,tab3,tab4 = st.tabs(["Clusters","GPI","Forecast","Data"])

    with tab1:
        fig = px.scatter(f,x="PC1",y="PC2",color="ClusterLabel",size="GPI",hover_name="RestaurantName",template="plotly_dark")
        fig.update_traces(marker=dict(line=dict(width=1,color="cyan")))
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        st.plotly_chart(px.histogram(f,x="GPI",color="ClusterLabel",template="plotly_dark"),use_container_width=True)
        st.plotly_chart(px.box(f,x="ClusterLabel",y="GPI",template="plotly_dark"),use_container_width=True)

    with tab3:
        st.plotly_chart(px.scatter(f,x="MonthlyOrders",y="NextMonthOrders",color="ClusterLabel",template="plotly_dark"),use_container_width=True)

    with tab4:
        st.dataframe(f.sort_values("GPI",ascending=False),use_container_width=True)
        st.download_button("Download", f.to_csv(index=False), "data.csv")

if __name__ == "__main__":
    app()
