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

st.title("Retail Sales Trend Analyzer")
st.caption("Interactive analysis of Rossmann retail sales data")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    if not os.path.exists("train.csv"):
        with zipfile.ZipFile("train.csv.zip", "r") as zip_ref:
            zip_ref.extractall(".")

    train = pd.read_csv("train.csv")
    store = pd.read_csv("store.csv")

    df = pd.merge(
        train,
        store,
        on="Store",
        how="left"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df


df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Filters")

# Store Type
store_types = sorted(df["StoreType"].dropna().unique())

selected_store_types = st.sidebar.multiselect(
    "Store Type",
    store_types,
    default=store_types
)

# Store
store_list = sorted(df["Store"].unique())

selected_stores = st.sidebar.multiselect(
    "Stores",
    store_list,
    default=store_list
)

# Promotion
promo_option = st.sidebar.selectbox(
    "Promotion",
    ["All", "Promo Only", "No Promo"]
)

# State Holiday
holiday_option = st.sidebar.selectbox(
    "State Holiday",
    ["All", "Holiday", "No Holiday"]
)

# School Holiday
school_option = st.sidebar.selectbox(
    "School Holiday",
    ["All", "Holiday", "No Holiday"]
)

# Date
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
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
    filtered_df = filtered_df[
        filtered_df["Promo"] == 1
    ]

elif promo_option == "No Promo":
    filtered_df = filtered_df[
        filtered_df["Promo"] == 0
    ]

if holiday_option == "Holiday":
    filtered_df = filtered_df[
        filtered_df["StateHoliday"] != "0"
    ]

elif holiday_option == "No Holiday":
    filtered_df = filtered_df[
        filtered_df["StateHoliday"] == "0"
    ]

if school_option == "Holiday":
    filtered_df = filtered_df[
        filtered_df["SchoolHoliday"] == 1
    ]

elif school_option == "No Holiday":
    filtered_df = filtered_df[
        filtered_df["SchoolHoliday"] == 0
    ]

if len(date_range) == 2:

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    filtered_df = filtered_df[
        (filtered_df["Date"] >= start_date) &
        (filtered_df["Date"] <= end_date)
    ]

# --------------------------------------------------
# EMPTY DATA CHECK
# --------------------------------------------------

if filtered_df.empty:

    st.warning("No data is available for the selected filters.")
    st.stop()

# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------

st.subheader("Sales Overview")

total_sales = filtered_df["Sales"].sum()
total_customers = filtered_df["Customers"].sum()
average_sales = filtered_df["Sales"].mean()
total_stores = filtered_df["Store"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

with col2:
    st.metric(
        "Customers",
        f"{total_customers:,.0f}"
    )

with col3:
    st.metric(
        "Average Daily Sales",
        f"{average_sales:,.0f}"
    )

with col4:
    st.metric(
        "Stores",
        f"{total_stores:,}"
    )

# --------------------------------------------------
# MONTHLY SALES TREND
# --------------------------------------------------

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(
        filtered_df["Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

monthly_sales.index = monthly_sales.index.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 4))

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
# STORE TYPE ANALYSIS
# --------------------------------------------------

st.subheader("Store Analysis")

col1, col2 = st.columns(2)

# Sales by Store Type
with col1:

    store_type_sales = (
        filtered_df
        .groupby("StoreType")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        store_type_sales.index,
        store_type_sales.values
    )

    ax.set_xlabel("Store Type")
    ax.set_ylabel("Total Sales")
    ax.set_title("Sales by Store Type")

    plt.tight_layout()

    st.pyplot(fig)

# Pie Chart
with col2:

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.pie(
        store_type_sales.values,
        labels=store_type_sales.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Sales Share by Store Type")

    plt.tight_layout()

    st.pyplot(fig)

# --------------------------------------------------
# PROMOTION ANALYSIS
# --------------------------------------------------

st.subheader("Promotion Analysis")

promo_data = (
    filtered_df
    .groupby("Promo")["Sales"]
    .mean()
)

no_promo_average = promo_data.get(0, 0)
promo_average = promo_data.get(1, 0)

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["No Promotion", "Promotion"],
        [no_promo_average, promo_average]
    )

    ax.set_ylabel("Average Daily Sales")
    ax.set_title("Average Sales: Promotion vs No Promotion")

    plt.tight_layout()

    st.pyplot(fig)

with col2:

    promo_sales = (
        filtered_df
        .groupby("Promo")["Sales"]
        .sum()
    )

    labels = []
    values = []

    if 0 in promo_sales.index:
        labels.append("No Promotion")
        values.append(promo_sales[0])

    if 1 in promo_sales.index:
        labels.append("Promotion")
        values.append(promo_sales[1])

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Sales Share: Promotion vs No Promotion")

    plt.tight_layout()

    st.pyplot(fig)

# --------------------------------------------------
# TOP STORES
# --------------------------------------------------

st.subheader("Store Performance")

top_stores = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.bar(
    top_stores.index.astype(str),
    top_stores.values
)

ax.set_xlabel("Store")
ax.set_ylabel("Total Sales")
ax.set_title("Top 10 Stores by Sales")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# BOTTOM STORES
# --------------------------------------------------

bottom_stores = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
    .sort_values()
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.bar(
    bottom_stores.index.astype(str),
    bottom_stores.values
)

ax.set_xlabel("Store")
ax.set_ylabel("Total Sales")
ax.set_title("Bottom 10 Stores by Sales")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# CUSTOMER ANALYSIS
# --------------------------------------------------

st.subheader("Customer Analysis")

sample_size = min(5000, len(filtered_df))

sample_df = filtered_df.sample(
    sample_size,
    random_state=42
)

fig, ax = plt.subplots(figsize=(12, 4))

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
# SCHOOL HOLIDAY ANALYSIS
# --------------------------------------------------

st.subheader("Holiday Analysis")

school_holiday_sales = (
    filtered_df
    .groupby("SchoolHoliday")["Sales"]
    .mean()
)

holiday_values = [
    school_holiday_sales.get(0, 0),
    school_holiday_sales.get(1, 0)
]

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    ["Normal Day", "School Holiday"],
    holiday_values
)

ax.set_ylabel("Average Daily Sales")
ax.set_title("Average Sales During School Holidays")

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# DAY OF WEEK
# --------------------------------------------------

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

day_sales = day_sales.reindex(
    [1, 2, 3, 4, 5, 6, 7]
)

day_sales.index = [
    day_names[x]
    for x in day_sales.index
]

fig, ax = plt.subplots(figsize=(12, 4))

ax.bar(
    day_sales.index,
    day_sales.values
)

ax.set_xlabel("Day")
ax.set_ylabel("Average Daily Sales")
ax.set_title("Average Sales by Day of Week")

plt.xticks(rotation=25)
plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# YEAR-WISE ANALYSIS
# --------------------------------------------------

st.subheader("Year-wise Sales Performance")

# Calculate monthly sales first
monthly_year_data = (
    filtered_df
    .assign(
        Year=filtered_df["Date"].dt.year,
        Month=filtered_df["Date"].dt.month
    )
    .groupby(["Year", "Month"])["Sales"]
    .sum()
    .reset_index()
)

# Average monthly sales for each year
year_sales = (
    monthly_year_data
    .groupby("Year")["Sales"]
    .mean()
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    year_sales.index.astype(str),
    year_sales.values
)

ax.set_xlabel("Year")
ax.set_ylabel("Average Monthly Sales")
ax.set_title("Average Monthly Sales by Year")

plt.tight_layout()

st.pyplot(fig)

st.caption(
    "2015 contains only part of the year in the Rossmann dataset, "
    "so average monthly sales are used for a fairer year-to-year comparison."
)

# --------------------------------------------------
# INDIVIDUAL STORE
# --------------------------------------------------

st.subheader("Individual Store Performance")

selected_store = st.selectbox(
    "Select a Store",
    sorted(filtered_df["Store"].unique())
)

store_data = filtered_df[
    filtered_df["Store"] == selected_store
]

store_total_sales = store_data["Sales"].sum()
store_average_sales = store_data["Sales"].mean()
store_total_customers = store_data["Customers"].sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Store Sales",
        f"{store_total_sales:,.0f}"
    )

with col2:
    st.metric(
        "Average Daily Sales",
        f"{store_average_sales:,.0f}"
    )

with col3:
    st.metric(
        "Customers",
        f"{store_total_customers:,.0f}"
    )

store_monthly = (
    store_data
    .groupby(
        store_data["Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

store_monthly.index = store_monthly.index.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    store_monthly.index,
    store_monthly.values,
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

st.subheader("Business Insights")

store_totals = (
    filtered_df
    .groupby("Store")["Sales"]
    .sum()
)

best_store = store_totals.idxmax()
best_store_sales = store_totals.max()

monthly_totals = (
    filtered_df
    .groupby(
        filtered_df["Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

best_month = monthly_totals.idxmax()

st.write(
    f"**Highest-performing store:** Store {best_store} "
    f"with total sales of {best_store_sales:,.0f}."
)

st.write(
    f"**Highest-sales month:** {best_month}."
)

if 1 in promo_data.index and 0 in promo_data.index:

    if promo_average > no_promo_average:

        st.write(
            f"**Promotion effect:** Average daily sales were higher "
            f"during promotion periods ({promo_average:,.0f}) "
            f"than during non-promotion periods ({no_promo_average:,.0f})."
        )

    else:

        st.write(
            f"**Promotion effect:** Average daily sales were lower "
            f"during promotion periods ({promo_average:,.0f}) "
            f"than during non-promotion periods ({no_promo_average:,.0f})."
        )

# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

st.subheader("Download Filtered Data")

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="filtered_retail_sales.csv",
    mime="text/csv"
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

st.subheader("Filtered Data Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)
