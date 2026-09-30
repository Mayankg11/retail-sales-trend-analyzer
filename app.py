import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

st.set_page_config(
    page_title="Retail Sales Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Trend Analyzer")
st.write("An interactive dashboard for analyzing retail sales trends.")

# --------------------------------------------------
# Load Data
# --------------------------------------------------

# Extract train.csv from ZIP if it does not already exist
if not os.path.exists("train.csv"):
    with zipfile.ZipFile("train.csv.zip", "r") as zip_ref:
        zip_ref.extractall(".")

train = pd.read_csv("train.csv")
store = pd.read_csv("store.csv")

# Merge datasets
df = pd.merge(train, store, on="Store", how="left")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Total Stores", df["Store"].nunique())

with col3:
    st.metric("Total Sales", f"{df['Sales'].sum():,.0f}")

with col4:
    st.metric("Total Customers", f"{df['Customers'].sum():,.0f}")

# --------------------------------------------------
# Filters
# --------------------------------------------------

st.sidebar.header("🔎 Filters")

store_types = sorted(df["StoreType"].dropna().unique())

selected_store_type = st.sidebar.multiselect(
    "Select Store Type",
    store_types,
    default=store_types
)

promo_options = ["All", "Promo", "No Promo"]

selected_promo = st.sidebar.selectbox(
    "Promotion",
    promo_options
)

filtered_df = df[df["StoreType"].isin(selected_store_type)]

if selected_promo == "Promo":
    filtered_df = filtered_df[filtered_df["Promo"] == 1]

elif selected_promo == "No Promo":
    filtered_df = filtered_df[filtered_df["Promo"] == 0]

# --------------------------------------------------
# Monthly Sales Trend
# --------------------------------------------------

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .set_index("Date")
    .resample("ME")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(monthly_sales.index, monthly_sales.values)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Sales by Store Type
# --------------------------------------------------

st.subheader("🏪 Sales by Store Type")

store_sales = (
    filtered_df
    .groupby("StoreType")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(store_sales.index, store_sales.values)

ax.set_xlabel("Store Type")
ax.set_ylabel("Total Sales")
ax.set_title("Total Sales by Store Type")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Promo vs Non-Promo Sales
# --------------------------------------------------

st.subheader("🎯 Promo vs Non-Promo Sales")

promo_sales = (
    filtered_df
    .groupby("Promo")["Sales"]
    .sum()
)

promo_labels = ["No Promo", "Promo"]

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    promo_labels,
    [
        promo_sales.get(0, 0),
        promo_sales.get(1, 0)
    ]
)

ax.set_xlabel("Promotion")
ax.set_ylabel("Total Sales")
ax.set_title("Sales Comparison: Promo vs No Promo")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Customers vs Sales
# --------------------------------------------------

st.subheader("👥 Customers vs Sales")

fig, ax = plt.subplots(figsize=(10, 5))

sample_df = filtered_df.sample(
    min(5000, len(filtered_df)),
    random_state=42
)

ax.scatter(
    sample_df["Customers"],
    sample_df["Sales"],
    alpha=0.4
)

ax.set_xlabel("Customers")
ax.set_ylabel("Sales")
ax.set_title("Customers vs Sales")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Year-wise Sales
# --------------------------------------------------

st.subheader("📅 Year-wise Sales")

yearly_sales = (
    filtered_df
    .groupby(filtered_df["Date"].dt.year)["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    yearly_sales.index.astype(str),
    yearly_sales.values
)

ax.set_xlabel("Year")
ax.set_ylabel("Total Sales")
ax.set_title("Year-wise Sales")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Sample Data
# --------------------------------------------------

st.subheader("📋 Sample Data")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

st.success("Dashboard loaded successfully!")
