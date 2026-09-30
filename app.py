import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

st.set_page_config(
    page_title="Retail Sales Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Trend Analyzer")
st.write("An interactive dashboard for analyzing retail sales trends.")

# Load train data from ZIP file
with zipfile.ZipFile("train.csv.zip", "r") as zip_ref:
    zip_ref.extractall(".")

train = pd.read_csv("train.csv")
store = pd.read_csv("store.csv")

# Merge datasets
df = pd.merge(train, store, on="Store", how="left")

st.success("Data loaded successfully!")

# Basic information
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Stores", df["Store"].nunique())

with col3:
    st.metric("Total Sales", f"{df['Sales'].sum():,.0f}")

# Show data
st.subheader("Sample Data")
st.dataframe(df.head(10))
