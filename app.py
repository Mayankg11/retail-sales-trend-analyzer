import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os


st.set_page_config(
    page_title="Retail Sales Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Sales Trend Analyzer")
st.caption("Rossmann Store Sales | EDA + Interactive Business Insights")

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
        raise FileNotFoundError(
            "train.csv or train.csv.zip not found"
        )

    store = pd.read_csv("store.csv")

    # Same cleaning used in Kaggle analysis
    store["CompetitionDistance"] = store[
        "CompetitionDistance"
    ].fillna(
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

st.sidebar.title("Dashboard Controls")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Sales Trends",
        "Sales Drivers",
        "Store Performance",
        "Findings"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Filters")

# Year filter
years = sorted(df["Year"].unique())

selected_years = st.sidebar.multiselect(
    "Year",
    years,
    default=years
)

# Store type filter
store_types = sorted(df["StoreType"].unique())

selected_store_types = st.sidebar.multiselect(
    "Store Type",
    store_types,
    default=store_types
)

# Promo filter
promo_options = st.sidebar.multiselect(
    "Promotion",
    [0, 1],
    default=[0, 1],
    format_func=lambda x: "No Promo" if x == 0 else "Promo"
)

# Store filter
store_list = sorted(df["Store"].unique())

selected_stores = st.sidebar.multiselect(
    "Store",
    store_list,
    default=[]
)

filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["StoreType"].isin(selected_store_types)) &
    (df["Promo"].isin(promo_options))
].copy()

# Store filter only applied if user selects stores
if selected_stores:
    filtered_df = filtered_df[
        filtered_df["Store"].isin(selected_stores)
    ]

st.sidebar.divider()

if st.sidebar.button("Reset Filters"):
    st.rerun()

st.sidebar.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} records"
)

if filtered_df.empty:

    st.warning(
        "No data available for the selected filters. "
        "Please change the filter values."
    )

    st.stop()

csv_data = filtered_df.to_csv(index=False)

st.sidebar.download_button(
    label="Download Filtered Data",
    data=csv_data,
    file_name="filtered_retail_sales.csv",
    mime="text/csv"
)

avg_sales = filtered_df["Sales"].mean()

avg_customers = filtered_df["Customers"].mean()

total_sales = filtered_df["Sales"].sum()

customer_sales_corr = filtered_df[
    ["Customers", "Sales"]
].corr().iloc[0, 1]

promo_sales = filtered_df.groupby(
    "Promo"
)["Sales"].mean()

monthly_sales = filtered_df.groupby(
    "YearMonth"
)["Sales"].mean()

year_sales = filtered_df.groupby(
    "Year"
)["Sales"].mean()

month_sales = filtered_df.groupby(
    "Month"
)["Sales"].mean()

store_type_avg = filtered_df.groupby(
    "StoreType"
)["Sales"].mean()

store_type_total = (
    filtered_df.groupby("StoreType")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

store_sales = filtered_df.groupby(
    "Store"
)["Sales"].mean()

top_stores = store_sales.sort_values(
    ascending=False
).head(10)

bottom_stores = store_sales.sort_values(
    ascending=True
).head(10)


if page == "Overview":

    st.header("Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Average Sales",
        f"{avg_sales:,.0f}"
    )

    c3.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    c4.metric(
        "Customer ↔ Sales",
        f"{customer_sales_corr:.3f}"
    )

    st.divider()

    st.subheader("Quick Insights")

    q1, q2, q3 = st.columns(3)

    if not store_type_avg.empty:

        q1.metric(
            "Top Store Type",
            store_type_avg.idxmax().upper(),
            f"{store_type_avg.max():,.0f} avg sales"
        )

    if not month_sales.empty:

        q2.metric(
            "Best Month",
            f"Month {month_sales.idxmax()}",
            f"{month_sales.max():,.0f} avg sales"
        )

    if not promo_sales.empty and len(promo_sales) > 1:

        promo_difference = (
            promo_sales[1] - promo_sales[0]
        )

        q3.metric(
            "Promo Difference",
            f"{promo_difference:,.0f}",
            "Average sales difference"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Average Sales by Store Type**")

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

        st.write("**Total Sales by Store Type**")

        temp2 = store_type_total.reset_index()

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

    st.subheader("Current Filter Summary")

    st.write(
        f"""
        **Years:** {", ".join(map(str, selected_years))}  
        **Store Types:** {", ".join(selected_store_types)}  
        **Promotion:** {", ".join(
            ["No Promo" if x == 0 else "Promo"
             for x in promo_options]
        )}
        """
    )


elif page == "Sales Trends":

    st.header("Sales Trends")

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

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    if not monthly_sales.empty:

        st.caption(
            f"Highest monthly average: "
            f"{monthly_sales.idxmax()} "
            f"({monthly_sales.max():,.0f})."
        )

    st.subheader("Average Sales by Year")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.bar(
        year_sales.index.astype(str),
        year_sales.values
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Sales")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    if not year_sales.empty:

        st.caption(
            f"{year_sales.idxmax()} has the highest "
            f"average sales in the selected data."
        )
    st.subheader("Average Sales by Month")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(
        month_sales.index,
        month_sales.values
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    if not month_sales.empty:

        st.caption(
            f"Month {month_sales.idxmax()} has the highest "
            f"average sales."
        )


elif page == "Sales Drivers":

    st.header("Sales Drivers")
    st.subheader("Customers vs Sales")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.scatter(
        filtered_df["Customers"],
        filtered_df["Sales"],
        alpha=0.2
    )

    ax.set_xlabel("Customers")
    ax.set_ylabel("Sales")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    st.markdown(
        f"**Correlation: {customer_sales_corr:.3f}** — "
        "higher customer counts are generally associated "
        "with higher sales."
    )

    st.subheader("Promotion vs Sales")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    sns.boxplot(
        x="Promo",
        y="Sales",
        data=filtered_df,
        ax=ax
    )

    ax.set_xlabel("Promo (0 = No, 1 = Yes)")
    ax.set_ylabel("Sales")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    if 0 in promo_sales.index:

        st.write(
            f"**Without Promo:** "
            f"{promo_sales[0]:,.0f} average sales"
        )

    if 1 in promo_sales.index:

        st.write(
            f"**With Promo:** "
            f"{promo_sales[1]:,.0f} average sales"
        )
    st.subheader("Day of Week + Promotion")

    day_promo = (
        filtered_df.groupby(
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

    pivot = pivot.rename(
        columns={
            0: "No Promo",
            1: "Promo"
        }
    )

    fig, ax = plt.subplots(figsize=(7, 3.5))

    pivot.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Sales")
    ax.tick_params(
        axis="x",
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    st.caption(
        "The difference between promotion and non-promotion "
        "sales varies across the week."
    )


    st.subheader("Correlation Heatmap")

    columns = [
        "Sales",
        "Customers",
        "Promo",
        "Open",
        "SchoolHoliday",
        "CompetitionDistance"
    ]

    fig, ax = plt.subplots(
        figsize=(6.5, 4.5)
    )

    sns.heatmap(
        filtered_df[columns].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    st.caption(
        "Customers show the strongest positive relationship "
        "with Sales among the selected variables."
    )
elif page == "Store Performance":

    st.header("Store Performance")
    available_stores = sorted(
        filtered_df["Store"].unique()
    )

    selected_store = st.selectbox(
        "Select a Store to Explore",
        available_stores
    )

    selected_store_data = filtered_df[
        filtered_df["Store"] == selected_store
    ]

    store_avg = selected_store_data["Sales"].mean()

    store_customers = selected_store_data[
        "Customers"
    ].mean()

    store_promo = selected_store_data[
        "Promo"
    ].mean()

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Selected Store",
        selected_store
    )

    s2.metric(
        "Average Sales",
        f"{store_avg:,.0f}"
    )

    s3.metric(
        "Average Customers",
        f"{store_customers:,.0f}"
    )

    st.divider()

    st.subheader("Sales by Store Type")

    fig, ax = plt.subplots(
        figsize=(6, 3.5)
    )

    sns.boxplot(
        x="StoreType",
        y="Sales",
        data=filtered_df,
        ax=ax
    )

    ax.set_xlabel("Store Type")
    ax.set_ylabel("Sales")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    if not store_type_avg.empty:

        st.caption(
            f"Store Type "
            f"{store_type_avg.idxmax().upper()} "
            f"has the highest average sales."
        )
    st.subheader(
        "Where does the Total Sales Come From?"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(4.5, 4.5)
        )

        ax.pie(
            store_type_total.values,
            labels=store_type_total.index,
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Total Sales Contribution"
        )

        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)

    with col2:

        revenue_table = pd.DataFrame({
            "Store Type":
                store_type_total.index,
            "Total Sales":
                store_type_total.values
        })

        st.write(
            "**Total Sales by Store Type**"
        )

        st.dataframe(
            revenue_table,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Top 10 Stores")

    fig, ax = plt.subplots(
        figsize=(7, 3.5)
    )

    top_stores.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")
    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)
    st.subheader("Bottom 10 Stores")

    fig, ax = plt.subplots(
        figsize=(7, 3.5)
    )

    bottom_stores.sort_values().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")
    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)
elif page == "Findings":

    st.header("What We Found")

    st.subheader("1. Customers matter")

    st.write(
        f"Customer count has a strong positive relationship "
        f"with sales (correlation = "
        f"{customer_sales_corr:.3f})."
    )

    st.info(
        "Customer traffic is an important variable when "
        "examining sales performance."
    )

    st.subheader("2. Promotions show higher sales")

    if 0 in promo_sales.index and 1 in promo_sales.index:

        st.write(
            f"Average sales with promotion are "
            f"{promo_sales[1]:,.0f}, compared with "
            f"{promo_sales[0]:,.0f} without promotion."
        )

    st.info(
        "Promotion periods show a noticeable difference "
        "in average sales."
    )

    st.subheader("3. Sales change with time")

    if not month_sales.empty:

        st.write(
            f"Month {month_sales.idxmax()} has the highest "
            f"average sales, while Month "
            f"{month_sales.idxmin()} has the lowest."
        )

    st.info(
        "Historical time patterns can help understand "
        "changes in sales."
    )

    st.subheader("4. Store types behave differently")

    if not store_type_avg.empty:

        st.write(
            f"Type {store_type_avg.idxmax().upper()} "
            f"has the highest average sales."
        )

    if not store_type_total.empty:

        st.write(
            f"Type {store_type_total.idxmax().upper()} "
            f"contributes the most total sales."
        )

    st.info(
        "Average sales and total sales provide different "
        "views of store performance."
    )

    st.subheader("5. Store performance varies")

    if not top_stores.empty and not bottom_stores.empty:

        st.write(
            f"Store {top_stores.index[0]} has the highest "
            f"average sales in the selected data, while "
            f"Store {bottom_stores.index[0]} is among the lowest."
        )

    st.info(
        "Store-level differences can be explored using "
        "customers, promotions, competition and store type."
    )

    st.divider()

    st.subheader("Final Takeaway")

    st.write(
        "Sales are related to multiple factors including "
        "customer traffic, promotions, time patterns and "
        "store-level characteristics."
    )

    st.caption(
        "Note: These are patterns observed in the dataset "
        "and do not prove causation."
    )

st.divider()

st.caption(
    "Retail Sales Trend Analyzer • "
    "Python • Pandas • Matplotlib • Seaborn • Streamlit"
)
