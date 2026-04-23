import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_FILE = "SkyCity_Auckland_Restaurants___Bars.csv"

st.set_page_config(layout="wide", page_title="Growth Intelligence", page_icon="🚀")

pio.templates.default = "plotly_dark"
NEON = ["#00F5FF","#7C3AED","#F43F5E","#22C55E","#FACC15","#38BDF8"]

st.markdown("""
<style>
html, body, .stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: #E5E7EB;
    font-family: 'Segoe UI';
}
.block-container {
    padding-top: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #000000);
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}
div[data-baseweb="tag"] {
    background: linear-gradient(135deg,#6366f1,#06b6d4);
    color: white;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 25px rgba(99,102,241,0.2);
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
    names = ["High Growth","Stable","Aggregator","Premium","Low"]
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
        st.error("CSV file not found")
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
    c3.metric("Avg Margin", f"{f['NetMargin'].mean():.2%}")
    c4.metric("Top Performers", (f["GPI"]>70).sum())

    trend = f.sort_values("GPI")
    fig_trend = px.line(trend, y="GPI")
    fig_trend.update_traces(line=dict(width=3,color="#00F5FF"))
    fig_trend.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
    st.plotly_chart(fig_trend, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Clusters","Insights","Performance"])

    with tab1:
        fig = px.scatter(f,x="PC1",y="PC2",color="ClusterLabel",size="GPI",color_discrete_sequence=NEON)
        fig.update_traces(marker=dict(line=dict(width=1,color="white")))
        fig.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        fig1 = px.histogram(f,x="GPI",nbins=30,color_discrete_sequence=["#7C3AED"])
        fig1.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
        st.plotly_chart(fig1,use_container_width=True)

        fig2 = px.box(f,x="ClusterLabel",y="GPI",color="ClusterLabel",color_discrete_sequence=NEON)
        fig2.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
        st.plotly_chart(fig2,use_container_width=True)

        fig3 = px.scatter(f,x="GPI",y="NetMargin",color="ClusterLabel",size="MonthlyOrders",color_discrete_sequence=NEON)
        fig3.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
        st.plotly_chart(fig3,use_container_width=True)

    with tab3:
        d1 = f.groupby("CuisineType")["GPI"].mean().reset_index()
        fig4 = px.bar(d1,x="CuisineType",y="GPI",color="GPI",color_continuous_scale="turbo")
        fig4.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
        st.plotly_chart(fig4,use_container_width=True)

        d2 = f.groupby("Subregion")["GPI"].mean().reset_index()
        fig5 = px.bar(d2,x="Subregion",y="GPI",color="GPI",color_continuous_scale="plasma")
        fig5.update_layout(paper_bgcolor="#000000",plot_bgcolor="#000000")
        st.plotly_chart(fig5,use_container_width=True)

        fig6 = px.pie(f,names="Recommendation",color_discrete_sequence=NEON,hole=0.5)
        fig6.update_layout(paper_bgcolor="#000000")
        st.plotly_chart(fig6,use_container_width=True)

if __name__ == "__main__":
    app()
