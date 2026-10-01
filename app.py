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
st.caption("Rossmann Store Sales | Interactive EDA + Business Insights")


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

    store["CompetitionDistance"] = store[
        "CompetitionDistance"
    ].fillna(
        store["CompetitionDistance"].median()
    )

    store = store.drop(columns=[
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear"
    ])

    train["Date"] = pd.to_datetime(
        train["Date"]
    )

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

    df["StateHoliday"] = df[
        "StateHoliday"
    ].astype(str)

    return df


try:
    df = load_data()

except Exception as e:
    st.error(str(e))
    st.stop()


st.sidebar.title("🔎 Filters")

years = sorted(
    df["Year"].unique()
)

selected_years = st.sidebar.multiselect(
    "Year",
    years,
    default=years
)

store_types = sorted(
    df["StoreType"].dropna().unique()
)

selected_store_types = st.sidebar.multiselect(
    "Store Type",
    store_types,
    default=store_types
)

selected_promo = st.sidebar.multiselect(
    "Promotion",
    [0, 1],
    default=[0, 1],
    format_func=lambda x:
        "No Promo" if x == 0 else "Promo"
)

stores = sorted(
    df["Store"].unique()
)

selected_stores = st.sidebar.multiselect(
    "Store",
    stores,
    default=stores
)

st.sidebar.divider()

st.sidebar.title("Project Sections")

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

if st.sidebar.button("🔄 Reset Filters"):

    st.session_state.clear()
    st.rerun()


filtered_df = df[
    df["Year"].isin(selected_years)
    & df["StoreType"].isin(selected_store_types)
    & df["Promo"].isin(selected_promo)
    & df["Store"].isin(selected_stores)
].copy()


if filtered_df.empty:

    st.warning(
        "No data is available for the selected filters. "
        "Please select different filter values."
    )

    st.stop()


st.sidebar.metric(
    "Filtered Records",
    f"{len(filtered_df):,}"
)


avg_sales = filtered_df[
    "Sales"
].mean()

avg_customers = filtered_df[
    "Customers"
].mean()

promo_sales = filtered_df.groupby(
    "Promo"
)["Sales"].mean()

customer_sales_corr = filtered_df[
    ["Customers", "Sales"]
].corr().iloc[0, 1]

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
        "Avg Customers",
        f"{avg_customers:,.0f}"
    )

    c4.metric(
        "Customer ↔ Sales",
        f"{customer_sales_corr:.3f}"
    )

    st.divider()

    st.subheader("Quick Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Average Sales by Store Type**"
        )

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

        st.write(
            "**Total Sales by Store Type**"
        )

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
        - **Customers and sales:** correlation = {customer_sales_corr:.3f}
        - **Promotion:** average sales are higher when promotion is active
        - **Highest average-sales store type:** {store_type_avg.idxmax().upper()}
        - **Highest total-sales store type:** {store_type_total.idxmax().upper()}
        - **Highest average-sales month:** Month {month_sales.idxmax()}
        """
    )


elif page == "Sales Trends":

    st.header("📈 Sales Trends")

    st.subheader("Monthly Average Sales")

    fig, ax = plt.subplots(
        figsize=(7, 2.5)
    )

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        linewidth=2
    )

    ax.set_xlabel("Month")
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

    st.caption(
        "Average sales movement across the selected period."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Average Sales by Year")

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
        )

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

    with col2:

        st.subheader("Average Sales by Month")

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
        )

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

    st.caption(
        f"Highest average sales month: Month {month_sales.idxmax()}."
    )


elif page == "Sales Drivers":

    st.header("📊 Sales Drivers")

    st.subheader("Customers vs Sales")

    fig, ax = plt.subplots(
        figsize=(6, 2.5)
    )

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

    st.markdown(
        f"""
        **Correlation: {customer_sales_corr:.3f}**

        Customer count generally increases with sales.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Promotion vs Sales")

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
        )

        sns.boxplot(
            x="Promo",
            y="Sales",
            data=filtered_df,
            ax=ax
        )

        ax.set_xlabel(
            "Promo (0 = No, 1 = Yes)"
        )

        ax.set_ylabel("Sales")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

    with col2:

        st.subheader(
            "Day of Week + Promotion"
        )

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

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
        )

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

    no_promo = promo_sales.get(
        0,
        0
    )

    promo = promo_sales.get(
        1,
        0
    )

    st.info(
        f"Average Sales without Promo: {no_promo:,.0f} | "
        f"With Promo: {promo:,.0f}"
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
        figsize=(6, 3)
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


elif page == "Store Performance":

    st.header("🏪 Store Performance")

    st.subheader("Sales by Store Type")

    fig, ax = plt.subplots(
        figsize=(6, 2.5)
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

    st.subheader(
        "Total Sales Contribution by Store Type"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(3.5, 3.5)
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

    with col2:

        revenue_table = pd.DataFrame({
            "Store Type": store_type_total.index,
            "Total Sales": store_type_total.values
        })

        st.dataframe(
            revenue_table,
            use_container_width=True,
            hide_index=True
        )

    st.subheader(
        "🔍 Individual Store Analysis"
    )

    available_stores = sorted(
        filtered_df["Store"].unique()
    )

    if available_stores:

        selected_store = st.selectbox(
            "Select a store",
            available_stores
        )

        selected_store_data = filtered_df[
            filtered_df["Store"] == selected_store
        ]

        store_avg = selected_store_data[
            "Sales"
        ].mean()

        store_customers = selected_store_data[
            "Customers"
        ].mean()

        a, b, c = st.columns(3)

        a.metric(
            "Average Sales",
            f"{store_avg:,.0f}"
        )

        b.metric(
            "Average Customers",
            f"{store_customers:,.0f}"
        )

        c.metric(
            "Records",
            f"{len(selected_store_data):,}"
        )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top 10 Stores")

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
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

    with col2:

        st.subheader("Bottom 10 Stores")

        fig, ax = plt.subplots(
            figsize=(5, 2.5)
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


elif page == "Findings":

    st.header("💡 What We Found")

    st.subheader("1. Customers matter")

    st.write(
        f"Customer count has a strong positive relationship "
        f"with sales (correlation = {customer_sales_corr:.3f})."
    )

    st.info(
        "Customer traffic is an important indicator when "
        "monitoring store performance."
    )

    st.subheader("2. Promotions are useful")

    st.write(
        f"Average sales with promotion are "
        f"{promo:,.0f}, compared with "
        f"{no_promo:,.0f} without promotion."
    )

    st.info(
        "Promotion periods show higher average sales "
        "in the selected data."
    )

    st.subheader("3. Sales change with time")

    st.write(
        f"Month {month_sales.idxmax()} has the highest "
        f"average sales, while Month "
        f"{month_sales.idxmin()} has the lowest."
    )

    st.info(
        "Historical monthly patterns can help with "
        "inventory and staff planning."
    )

    st.subheader(
        "4. Store types behave differently"
    )

    st.write(
        f"Type {store_type_avg.idxmax().upper()} has the "
        f"highest average sales, while Type "
        f"{store_type_total.idxmax().upper()} contributes "
        "the most total sales."
    )

    st.info(
        "Both average sales and total sales are useful "
        "when comparing store types."
    )

    st.subheader(
        "5. Store performance varies"
    )

    st.write(
        f"Store {top_stores.index[0]} has the highest "
        f"average sales among the selected stores, while "
        f"Store {bottom_stores.index[0]} is among the lowest."
    )

    st.info(
        "Store performance can be explored using customers, "
        "promotions, competition and store characteristics."
    )

    st.divider()

    st.subheader("Final Takeaway")

    st.write(
        "Sales are influenced by multiple factors. "
        "Customer traffic, promotions, time patterns and "
        "store-level differences all show useful relationships "
        "in the dataset."
    )

    st.caption(
        "Note: These are patterns observed in the dataset, "
        "not proof of causation."
    )


st.divider()

st.subheader("📥 Export Filtered Data")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered CSV",
    data=csv_data,
    file_name="filtered_retail_sales.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Retail Sales Trend Analyzer • "
    "Python • Pandas • Matplotlib • Seaborn • Streamlit"
)
