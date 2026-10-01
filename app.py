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
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 5px;
}

.subtitle {
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 650;
    color: #111827;
    margin-top: 10px;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

.stButton > button {
    border-radius: 8px;
    border: none;
}

.stDownloadButton > button {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():

    if os.path.exists("train.csv"):

        train = pd.read_csv("train.csv")

    elif os.path.exists("train.csv.zip"):

        with zipfile.ZipFile(
            "train.csv.zip",
            "r"
        ) as z:

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

    store["CompetitionDistance"] = (
        store["CompetitionDistance"]
        .fillna(
            store["CompetitionDistance"].median()
        )
    )

    store = store.drop(
        columns=[
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear"
        ]
    )

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

    df["StateHoliday"] = (
        df["StateHoliday"].astype(str)
    )

    return df


try:

    df = load_data()

except Exception as e:

    st.error(str(e))
    st.stop()


st.sidebar.markdown(
    "## 📊 Retail Sales"
)

st.sidebar.caption(
    "Trend Analyzer Dashboard"
)

st.sidebar.divider()

st.sidebar.markdown(
    "### 🔎 Filters"
)


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


st.sidebar.divider()

st.sidebar.markdown(
    "### 📑 Navigation"
)


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


if st.sidebar.button(
    "🔄 Reset Filters",
    use_container_width=True
):

    st.session_state.clear()
    st.rerun()


filtered_df = df[
    df["Year"].isin(selected_years)
    & df["StoreType"].isin(
        selected_store_types
    )
    & df["Promo"].isin(
        selected_promo
    )
].copy()


if filtered_df.empty:

    st.warning(
        "No data is available for the selected filters."
    )

    st.stop()


avg_sales = filtered_df[
    "Sales"
].mean()


avg_customers = filtered_df[
    "Customers"
].mean()


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
    filtered_df.groupby(
        "StoreType"
    )["Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
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


st.markdown(
    '<div class="main-title">Retail Sales Trend Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive analysis of retail sales, customer behavior, '
    'promotions and store performance'
    '</div>',
    unsafe_allow_html=True
)


if page == "Overview":

    st.markdown(
        '<div class="section-title">📌 Overview</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Average Sales",
        f"{avg_sales:,.0f}"
    )

    c3.metric(
        "Average Customers",
        f"{avg_customers:,.0f}"
    )

    c4.metric(
        "Customer-Sales Correlation",
        f"{customer_sales_corr:.3f}"
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 📊 Average Sales by Store Type"
        )

        temp = (
            store_type_avg
            .sort_values(
                ascending=False
            )
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

        st.markdown(
            "### 💰 Total Sales by Store Type"
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

    st.write("")

    st.markdown(
        "### 🔍 Key Highlights"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            f"Highest average-sales store type: "
            f"**{store_type_avg.idxmax().upper()}**"
        )

    with col2:

        st.info(
            f"Highest total-sales store type: "
            f"**{store_type_total.idxmax().upper()}**"
        )

    with col3:

        st.info(
            f"Highest average-sales month: "
            f"**Month {month_sales.idxmax()}**"
        )


elif page == "Sales Trends":

    st.markdown(
        '<div class="section-title">📈 Sales Trends</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        "### Monthly Average Sales"
    )

    fig, ax = plt.subplots(
        figsize=(8, 2.8)
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

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Average Sales by Year"
        )

        fig, ax = plt.subplots(
            figsize=(5, 2.7)
        )

        ax.bar(
            year_sales.index.astype(str),
            year_sales.values
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Average Sales")

        ax.grid(
            axis="y",
            alpha=0.2
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

    with col2:

        st.markdown(
            "### Average Sales by Month"
        )

        fig, ax = plt.subplots(
            figsize=(5, 2.7)
        )

        ax.bar(
            month_sales.index,
            month_sales.values
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Average Sales")

        ax.grid(
            axis="y",
            alpha=0.2
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )


elif page == "Sales Drivers":

    st.markdown(
        '<div class="section-title">📊 Sales Drivers</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        "### 👥 Customers vs Sales"
    )

    fig, ax = plt.subplots(
        figsize=(7, 2.8)
    )

    ax.scatter(
        filtered_df["Customers"],
        filtered_df["Sales"],
        alpha=0.2
    )

    ax.set_xlabel("Customers")
    ax.set_ylabel("Sales")

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.info(
        f"Customer-Sales correlation: "
        f"**{customer_sales_corr:.3f}**"
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🎯 Promotion vs Sales"
        )

        fig, ax = plt.subplots(
            figsize=(5, 2.7)
        )

        sns.boxplot(
            x="Promo",
            y="Sales",
            data=filtered_df,
            ax=ax
        )

        ax.set_xlabel(
            "Promotion"
        )

        ax.set_ylabel(
            "Sales"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

    with col2:

        st.markdown(
            "### 📅 Day of Week + Promotion"
        )

        day_promo = (
            filtered_df.groupby(
                [
                    "DayOfWeek",
                    "Promo"
                ]
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
            figsize=(5, 2.7)
        )

        pivot.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Day of Week"
        )

        ax.set_ylabel(
            "Average Sales"
        )

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
        f"Average Sales — No Promo: **{no_promo:,.0f}** "
        f"| Promo: **{promo:,.0f}**"
    )

    st.write("")

    st.markdown(
        "### 🔥 Correlation Heatmap"
    )

    columns = [
        "Sales",
        "Customers",
        "Promo",
        "Open",
        "SchoolHoliday",
        "CompetitionDistance"
    ]

    fig, ax = plt.subplots(
        figsize=(7, 3.2)
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

    st.markdown(
        '<div class="section-title">🏪 Store Performance</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        "### Store Type Sales Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(7, 2.8)
    )

    sns.boxplot(
        x="StoreType",
        y="Sales",
        data=filtered_df,
        ax=ax
    )

    ax.set_xlabel(
        "Store Type"
    )

    ax.set_ylabel(
        "Sales"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🥧 Total Sales Contribution"
        )

        fig, ax = plt.subplots(
            figsize=(3.8, 3.2)
        )

        ax.pie(
            store_type_total.values,
            labels=store_type_total.index,
            autopct="%1.1f%%"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

    with col2:

        st.markdown(
            "### 💰 Sales by Store Type"
        )

        revenue_table = pd.DataFrame({
            "Store Type":
                store_type_total.index,
            "Total Sales":
                store_type_total.values
        })

        st.dataframe(
            revenue_table,
            use_container_width=True,
            hide_index=True
        )

    st.write("")

    st.markdown(
        "### 🔍 Individual Store Analysis"
    )

    available_stores = sorted(
        filtered_df["Store"].unique()
    )

    selected_store = st.selectbox(
        "Select a Store",
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

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Sales",
        f"{store_avg:,.0f}"
    )

    col2.metric(
        "Average Customers",
        f"{store_customers:,.0f}"
    )

    col3.metric(
        "Records",
        f"{len(selected_store_data):,}"
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🏆 Top 10 Stores"
        )

        fig, ax = plt.subplots(
            figsize=(5, 2.8)
        )

        top_stores.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Store"
        )

        ax.set_ylabel(
            "Average Sales"
        )

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

        st.markdown(
            "### 📉 Bottom 10 Stores"
        )

        fig, ax = plt.subplots(
            figsize=(5, 2.8)
        )

        bottom_stores.sort_values().plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Store"
        )

        ax.set_ylabel(
            "Average Sales"
        )

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

    st.markdown(
        '<div class="section-title">💡 Key Findings</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 👥 Customer Behavior"
        )

        st.write(
            f"Customers and sales show a strong positive "
            f"relationship with a correlation of "
            f"**{customer_sales_corr:.3f}**."
        )

    with col2:

        st.markdown(
            "### 🎯 Promotion Impact"
        )

        st.write(
            f"Average sales with promotion are "
            f"**{promo:,.0f}**, compared with "
            f"**{no_promo:,.0f}** without promotion."
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 📅 Time Patterns"
        )

        st.write(
            f"Month **{month_sales.idxmax()}** has the "
            f"highest average sales in the selected data."
        )

    with col2:

        st.markdown(
            "### 🏪 Store Types"
        )

        st.write(
            f"Store Type **{store_type_avg.idxmax().upper()}** "
            f"has the highest average sales."
        )

    st.write("")

    st.markdown(
        "### 📌 Overall Insight"
    )

    st.write(
        "The analysis shows that retail sales are influenced "
        "by several factors including customer traffic, "
        "promotional activity, time-based patterns and "
        "differences between stores."
    )

    st.info(
        "These findings represent patterns observed in the "
        "historical dataset and should not be interpreted as "
        "proof of direct causation."
    )


st.divider()

st.markdown(
    "### 📥 Export Filtered Data"
)

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
