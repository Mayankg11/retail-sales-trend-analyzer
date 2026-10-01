import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Sales Trend Analyzer")
st.caption("Rossmann Store Sales | EDA + Business Insights")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    if os.path.exists("train.csv"):
        train = pd.read_csv("train.csv")

    elif os.path.exists("train.csv.zip"):
        with zipfile.ZipFile("train.csv.zip", "r") as z:

            file_name = [
                x for x in z.namelist()
                if x.endswith("train.csv")
            ][0]

            with z.open(file_name) as f:
                train = pd.read_csv(f)

    else:
        raise FileNotFoundError("train.csv or train.csv.zip not found")

    store = pd.read_csv("store.csv")

    # Same cleaning used in Kaggle analysis
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()
    )

    store = store.drop(columns=[
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear"
    ])

    train["Date"] = pd.to_datetime(train["Date"])

    df = train.merge(
        store,
        on="Store",
        how="left"
    )

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M")
    df["Day"] = df["Date"].dt.day
    df["Week"] = df["Date"].dt.isocalendar().week

    df["StateHoliday"] = df["StateHoliday"].astype(str)

    return df


try:
    df = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------
# BASIC CALCULATIONS
# ---------------------------------------------------------

avg_sales = df["Sales"].mean()
avg_customers = df["Customers"].mean()

promo_sales = df.groupby("Promo")["Sales"].mean()

customer_sales_corr = df[
    ["Customers", "Sales"]
].corr().iloc[0, 1]

monthly_sales = df.groupby("YearMonth")["Sales"].mean()

year_sales = df.groupby("Year")["Sales"].mean()

month_sales = df.groupby("Month")["Sales"].mean()

store_type_avg = df.groupby("StoreType")["Sales"].mean()

store_type_total = (
    df.groupby("StoreType")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

store_sales = df.groupby("Store")["Sales"].mean()

top_stores = store_sales.sort_values(
    ascending=False
).head(10)

bottom_stores = store_sales.sort_values(
    ascending=True
).head(10)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Project Sections")

page = st.sidebar.radio(
    "Select",
    [
        "Overview",
        "Sales Trends",
        "Sales Drivers",
        "Store Performance",
        "Findings"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records",
        f"{len(df):,}"
    )

    c2.metric(
        "Average Sales",
        f"{avg_sales:,.0f}"
    )

    c3.metric(
        "Avg Customers",
        f"{avg_customers:,.0f}"
    )

    c4.metric(
        "Customer ↔ Sales",
        f"{customer_sales_corr:.3f}"
    )

    st.write("")

    st.subheader("Quick look")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Average sales by store type**")

        temp = (
            store_type_avg
            .sort_values(ascending=False)
            .reset_index()
        )

        temp.columns = [
            "Store Type",
            "Average Sales"
        ]

        st.dataframe(
            temp.round(2),
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.write("**Total sales by store type**")

        temp2 = (
            store_type_total
            .reset_index()
        )

        temp2.columns = [
            "Store Type",
            "Total Sales"
        ]

        st.dataframe(
            temp2,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("What stands out?")

    st.markdown(
        f"""
        - **Customers and sales:** strong positive relationship ({customer_sales_corr:.3f})
        - **Promotion:** average sales are higher when Promo = 1
        - **Highest average-sales store type:** {store_type_avg.idxmax().upper()}
        - **Highest total-sales store type:** {store_type_total.idxmax().upper()}
        - **Highest average-sales month:** Month {month_sales.idxmax()}
        """
    )


# =========================================================
# SALES TRENDS
# =========================================================

elif page == "Sales Trends":

    st.header("Sales Trends")

    # ---------------- Monthly ----------------

    st.subheader("Monthly Average Sales")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        linewidth=2
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        "Sales move up and down across the year instead of staying constant."
    )

    # ---------------- Year ----------------

    st.subheader("Average Sales by Year")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.bar(
        year_sales.index.astype(str),
        year_sales.values
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Sales")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        f"{year_sales.idxmax()} has the highest average sales among the available years."
    )

    # ---------------- Month ----------------

    st.subheader("Average Sales by Month")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(
        month_sales.index,
        month_sales.values
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        f"Month {month_sales.idxmax()} has the highest average sales, "
        f"while Month {month_sales.idxmin()} has the lowest."
    )


# =========================================================
# SALES DRIVERS
# =========================================================

elif page == "Sales Drivers":

    st.header("Sales Drivers")

    # ---------------- Customers ----------------

    st.subheader("Customers vs Sales")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.scatter(
        df["Customers"],
        df["Sales"],
        alpha=0.2
    )

    ax.set_xlabel("Customers")
    ax.set_ylabel("Sales")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.markdown(
        f"**Correlation: {customer_sales_corr:.3f}** — more customers are generally linked with higher sales."
    )

    # ---------------- Promo ----------------

    st.subheader("Promotion vs Sales")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    sns.boxplot(
        x="Promo",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Promo (0 = No, 1 = Yes)")
    ax.set_ylabel("Sales")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.markdown(
        f"""
        **Without Promo:** {promo_sales[0]:,.0f} average sales  
        **With Promo:** {promo_sales[1]:,.0f} average sales

        → Promotion periods show noticeably higher sales.
        """
    )

    # ---------------- Day + Promo ----------------

    st.subheader("Day of Week + Promotion")

    day_promo = (
        df.groupby(
            ["DayOfWeek", "Promo"]
        )["Sales"]
        .mean()
        .reset_index()
    )

    pivot = day_promo.pivot(
        index="DayOfWeek",
        columns="Promo",
        values="Sales"
    )

    pivot.columns = [
        "No Promo",
        "Promo"
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    pivot.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Sales")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        "The promotion difference changes depending on the day."
    )

    # ---------------- Heatmap ----------------

    st.subheader("Correlation Heatmap")

    columns = [
        "Sales",
        "Customers",
        "Promo",
        "Open",
        "SchoolHoliday",
        "CompetitionDistance"
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    sns.heatmap(
        df[columns].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        "Customers show the strongest positive relationship with Sales among the variables shown."
    )


# =========================================================
# STORE PERFORMANCE
# =========================================================

elif page == "Store Performance":

    st.header("Store Performance")

    # ---------------- Store type ----------------

    st.subheader("Sales by Store Type")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    sns.boxplot(
        x="StoreType",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Store Type")
    ax.set_ylabel("Sales")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        f"Store Type {store_type_avg.idxmax().upper()} has the highest average sales per record."
    )

    # ---------------- Revenue Pie ----------------

    st.subheader("Where does the total sales come from?")

    col1, col2 = st.columns([1, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        ax.pie(
            store_type_total.values,
            labels=store_type_total.index,
            autopct="%1.1f%%"
        )

        ax.set_title("Total Sales Contribution")

        st.pyplot(fig, use_container_width=False)

    with col2:

        revenue_table = pd.DataFrame({
            "Store Type": store_type_total.index,
            "Total Sales": store_type_total.values
        })

        st.write("**Total sales ranking**")

        st.dataframe(
            revenue_table,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Type **{store_type_total.index[0].upper()}** "
            f"contributes the most total sales."
        )

    # ---------------- Top stores ----------------

    st.subheader("Top 10 Stores")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    top_stores.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        f"Store {top_stores.index[0]} has the highest average sales in the dataset."
    )

    # ---------------- Bottom stores ----------------

    st.subheader("Bottom 10 Stores")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bottom_stores.sort_values().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    st.caption(
        f"Store {bottom_stores.index[0]} has the lowest average sales among the stores."
    )


# =========================================================
# FINDINGS
# =========================================================

elif page == "Findings":

    st.header("What We Found")

    st.subheader("1. Customers matter")

    st.write(
        f"Customer count has a strong positive relationship with sales "
        f"(correlation = {customer_sales_corr:.3f})."
    )

    st.info(
        "Keep an eye on customer traffic when monitoring store performance."
    )

    st.subheader("2. Promotions are useful")

    st.write(
        f"Average sales with promotion are {promo_sales[1]:,.0f}, "
        f"compared with {promo_sales[0]:,.0f} without promotion."
    )

    st.info(
        "Promotions can be planned around periods where customer demand is stronger."
    )

    st.subheader("3. Sales change with time")

    st.write(
        f"Month {month_sales.idxmax()} has the highest average sales, "
        f"while Month {month_sales.idxmin()} has the lowest."
    )

    st.info(
        "Historical monthly patterns can help with inventory and staff planning."
    )

    st.subheader("4. Store types behave differently")

    st.write(
        f"Type {store_type_avg.idxmax().upper()} has the highest average sales, "
        f"but Type {store_type_total.idxmax().upper()} contributes the most total sales."
    )

    st.info(
        "Use both average sales and total sales when comparing store types."
    )

    st.subheader("5. Some stores need closer attention")

    st.write(
        f"Store {top_stores.index[0]} is the highest performer by average sales, "
        f"while Store {bottom_stores.index[0]} is among the lowest."
    )

    st.info(
        "Low-performing stores can be investigated using customers, promotions, "
        "competition and store characteristics."
    )

    st.divider()

    st.subheader("Final takeaway")

    st.write(
        "Sales are not controlled by one variable. "
        "Customer traffic, promotions, time patterns and store-level "
        "differences all show useful relationships in the data."
    )

    st.caption(
        "Note: These are patterns observed in the dataset, not proof of causation."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Retail Sales Trend Analyzer • Built using Python, Pandas, Matplotlib, Seaborn & Streamlit"
)
