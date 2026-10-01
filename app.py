import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Trend Analyzer")
st.caption("Interactive retail sales analytics dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not os.path.exists("train.csv"):
    with zipfile.ZipFile("train.csv.zip", "r") as zip_ref:
        zip_ref.extractall(".")

train = pd.read_csv("train.csv")
store = pd.read_csv("store.csv")

# Merge datasets
df = pd.merge(train, store, on="Store", how="left")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔎 Dashboard Filters")

# Store Type
store_types = sorted(df["StoreType"].dropna().unique())

selected_store_types = st.sidebar.multiselect(
    "🏪 Store Type",
    store_types,
    default=store_types
)

# Store selection
store_list = sorted(df["Store"].unique())

selected_stores = st.sidebar.multiselect(
    "🏬 Select Stores",
    store_list,
    default=store_list
)

# Promotion
promo_option = st.sidebar.selectbox(
    "🎯 Promotion",
    ["All", "Promo Only", "No Promo"]
)

# State Holiday
holiday_option = st.sidebar.selectbox(
    "🏖️ State Holiday",
    ["All", "Holiday", "No Holiday"]
)

# School Holiday
school_option = st.sidebar.selectbox(
    "🎒 School Holiday",
    ["All", "Holiday", "No Holiday"]
)

# Date Range
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df["StoreType"].isin(selected_store_types)
]

filtered_df = filtered_df[
    filtered_df["Store"].isin(selected_stores)
]

if promo_option == "Promo Only":
    filtered_df = filtered_df[filtered_df["Promo"] == 1]

elif promo_option == "No Promo":
    filtered_df = filtered_df[filtered_df["Promo"] == 0]

if holiday_option == "Holiday":
    filtered_df = filtered_df[filtered_df["StateHoliday"] != "0"]

elif holiday_option == "No Holiday":
    filtered_df = filtered_df[filtered_df["StateHoliday"] == "0"]

if school_option == "Holiday":
    filtered_df = filtered_df[filtered_df["SchoolHoliday"] == 1]

elif school_option == "No Holiday":
    filtered_df = filtered_df[filtered_df["SchoolHoliday"] == 0]

# Date filter
if isinstance(date_range, tuple) and len(date_range) == 2:

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    filtered_df = filtered_df[
        (filtered_df["Date"] >= start_date) &
        (filtered_df["Date"] <= end_date)
    ]

# --------------------------------------------------
# CHECK DATA
# --------------------------------------------------

if filtered_df.empty:

    st.warning("⚠️ No data available for the selected filters.")

    st.stop()

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

st.subheader("📌 Key Performance Indicators")

total_sales = filtered_df["Sales"].sum()
total_customers = filtered_df["Customers"].sum()
average_sales = filtered_df["Sales"].mean()
total_stores = filtered_df["Store"].nunique()

promo_sales = filtered_df.loc[
    filtered_df["Promo"] == 1,
    "Sales"
].sum()

total_sales_for_promo = filtered_df["Sales"].sum()

if total_sales_for_promo > 0:
    promo_percentage = (promo_sales / total_sales_for_promo) * 100
else:
    promo_percentage = 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "💰 Total Sales",
        f"{total_sales:,.0f}"
    )

with col2:
    st.metric(
        "🏪 Stores",
        f"{total_stores:,}"
    )

with col3:
    st.metric(
        "👥 Customers",
        f"{total_customers:,.0f}"
    )

with col4:
    st.metric(
        "📊 Avg. Sales",
        f"{average_sales:,.0f}"
    )

with col5:
    st.metric(
        "🎯 Promo Sales %",
        f"{promo_percentage:.1f}%"
    )

# --------------------------------------------------
# MONTHLY SALES TREND
# --------------------------------------------------

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .set_index("Date")
    .resample("ME")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales Trend")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# TWO COLUMN SECTION
# --------------------------------------------------

col1, col2 = st.columns(2)

# --------------------------------------------------
# SALES BY STORE TYPE
# --------------------------------------------------

with col1:

    st.subheader("🏪 Sales by Store Type")

    store_type_sales = (
        filtered_df
        .groupby("StoreType")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        store_type_sales.index,
        store_type_sales.values
    )

    ax.set_xlabel("Store Type")
    ax.set_ylabel("Total Sales")
    ax.set_title("Sales by Store Type")

    plt.tight_layout()

    st.pyplot(fig)

# --------------------------------------------------
# PROMO ANALYSIS
# --------------------------------------------------

with col2:

    st.subheader("🎯 Promo vs No Promo")

    promo_sales_data = (
        filtered_df
        .groupby("Promo")["Sales"]
        .sum()
    )

    no_promo = promo_sales_data.get(0, 0)
    promo = promo_sales_data.get(1, 0)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        ["No Promo", "Promo"],
        [no_promo, promo]
    )

    ax.set_ylabel("Total Sales")
    ax.set_title("Promotion Impact on Sales")

    plt.tight_layout()

    st.pyplot(fig)

# --------------------------------------------------
# TOP 10 STORES
# --------------------------------------------------

st.subheader("🏆 Top 10 Stores by Sales")

top_stores = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    top_stores.index.astype(str),
    top_stores.values
)

ax.set_xlabel("Store")
ax.set_ylabel("Total Sales")
ax.set_title("Top 10 Performing Stores")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# BOTTOM 10 STORES
# --------------------------------------------------

st.subheader("📉 Bottom 10 Stores by Sales")

bottom_stores = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .sort_values()
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    bottom_stores.index.astype(str),
    bottom_stores.values
)

ax.set_xlabel("Store")
ax.set_ylabel("Total Sales")
ax.set_title("Bottom 10 Stores")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# CUSTOMER VS SALES
# --------------------------------------------------

st.subheader("👥 Customer Count vs Sales")

sample_size = min(5000, len(filtered_df))

sample_df = filtered_df.sample(
    sample_size,
    random_state=42
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.scatter(
    sample_df["Customers"],
    sample_df["Sales"],
    alpha=0.4
)

ax.set_xlabel("Customers")
ax.set_ylabel("Sales")
ax.set_title("Relationship Between Customers and Sales")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# HOLIDAY ANALYSIS
# --------------------------------------------------

st.subheader("🏖️ Holiday vs Non-Holiday Sales")

holiday_sales = (
    filtered_df
    .groupby("SchoolHoliday")["Sales"]
    .mean()
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    ["No School Holiday", "School Holiday"],
    [
        holiday_sales.get(0, 0),
        holiday_sales.get(1, 0)
    ]
)

ax.set_ylabel("Average Sales")
ax.set_title("Average Sales During School Holidays")

plt.xticks(rotation=10)
plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# DAY OF WEEK ANALYSIS
# --------------------------------------------------

st.subheader("📅 Average Sales by Day of Week")

day_sales = (
    filtered_df
    .groupby("DayOfWeek")["Sales"]
    .mean()
)

day_names = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}

day_sales.index = day_sales.index.map(day_names)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    day_sales.index,
    day_sales.values
)

ax.set_xlabel("Day")
ax.set_ylabel("Average Sales")
ax.set_title("Average Sales by Day of Week")

plt.xticks(rotation=30)
plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# YEAR-WISE SALES
# --------------------------------------------------

st.subheader("📅 Year-wise Sales")

year_sales = (
    filtered_df
    .groupby(filtered_df["Date"].dt.year)["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    year_sales.index.astype(str),
    year_sales.values
)

ax.set_xlabel("Year")
ax.set_ylabel("Total Sales")
ax.set_title("Year-wise Sales")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# STORE PERFORMANCE
# --------------------------------------------------

st.subheader("🔍 Individual Store Performance")

selected_store = st.selectbox(
    "Select a Store",
    sorted(filtered_df["Store"].unique())
)

store_data = filtered_df[
    filtered_df["Store"] == selected_store
]

store_total_sales = store_data["Sales"].sum()
store_avg_sales = store_data["Sales"].mean()
store_customers = store_data["Customers"].sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Store Total Sales",
        f"{store_total_sales:,.0f}"
    )

with col2:
    st.metric(
        "Average Daily Sales",
        f"{store_avg_sales:,.0f}"
    )

with col3:
    st.metric(
        "Total Customers",
        f"{store_customers:,.0f}"
    )

store_monthly_sales = (
    store_data
    .set_index("Date")
    .resample("ME")["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    store_monthly_sales.index,
    store_monthly_sales.values,
    marker="o"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title(f"Monthly Sales - Store {selected_store}")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------

st.subheader("💡 Business Insights")

best_store = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .idxmax()
)

best_store_sales = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .max()
)

best_month = (
    filtered_df
    .groupby(filtered_df["Date"].dt.to_period("M"))["Sales"]
    .sum()
    .idxmax()
)

promo_average = filtered_df[
    filtered_df["Promo"] == 1
]["Sales"].mean()

non_promo_average = filtered_df[
    filtered_df["Promo"] == 0
]["Sales"].mean()

st.write(
    f"🏆 **Best Performing Store:** Store {best_store} "
    f"with sales of {best_store_sales:,.0f}"
)

st.write(
    f"📅 **Highest Sales Month:** {best_month}"
)

st.write(
    f"🎯 **Average Sales with Promotion:** "
    f"{promo_average:,.0f}"
)

st.write(
    f"🛒 **Average Sales without Promotion:** "
    f"{non_promo_average:,.0f}"
)

# --------------------------------------------------
# DOWNLOAD FILTERED DATA
# --------------------------------------------------

st.subheader("📥 Download Filtered Data")

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="filtered_retail_sales.csv",
    mime="text/csv"
)

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------

st.subheader("📋 Filtered Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

st.success("✅ Dashboard loaded successfully!")
