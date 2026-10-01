import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Retail Sales Trend Analyzer",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Retail Sales Trend Analyzer")

st.markdown(
    """
    ### From Sales Data to Business Insights

    This project analyzes the Rossmann Store Sales dataset to understand:

    - How sales change over time
    - How promotions are associated with sales
    - How customer traffic relates to sales
    - How sales differ across store types
    - Which stores perform strongly or weakly
    - Which factors are associated with higher or lower sales
    - What business actions can be suggested from the analysis
    """
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    # -----------------------------
    # Load train.csv
    # -----------------------------

    if os.path.exists("train.csv"):
        train = pd.read_csv("train.csv")

    elif os.path.exists("train.csv.zip"):
        with zipfile.ZipFile("train.csv.zip", "r") as z:
            train_file = [
                name for name in z.namelist()
                if name.endswith("train.csv")
            ][0]

            with z.open(train_file) as f:
                train = pd.read_csv(f)

    else:
        raise FileNotFoundError(
            "train.csv or train.csv.zip was not found."
        )

    # -----------------------------
    # Load store.csv
    # -----------------------------

    if not os.path.exists("store.csv"):
        raise FileNotFoundError(
            "store.csv was not found."
        )

    store = pd.read_csv("store.csv")

    # =====================================================
    # SAME CLEANING AS KAGGLE ANALYSIS
    # =====================================================

    # Fill CompetitionDistance missing values
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()
    )

    # Drop columns that had large missing values
    store = store.drop(columns=[
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear"
    ])

    # Convert Date
    train["Date"] = pd.to_datetime(train["Date"])

    # Merge datasets
    df = train.merge(
        store,
        on="Store",
        how="left"
    )

    # Date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M")
    df["Day"] = df["Date"].dt.day
    df["Week"] = df["Date"].dt.isocalendar().week

    # Convert StateHoliday to string
    df["StateHoliday"] = df["StateHoliday"].astype(str)

    return df


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = load_data()

except Exception as e:

    st.error("Data loading error:")
    st.code(str(e))

    st.info(
        """
        Make sure these files are present in the GitHub repository:

        - train.csv.zip
        - store.csv
        """
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Sales Trends",
        "What Affects Sales?",
        "Store Analysis",
        "Business Insights & Solutions",
        "Original Kaggle Visualizations"
    ]
)


# =========================================================
# COMMON CALCULATIONS
# =========================================================

average_sales = df["Sales"].mean()
average_customers = df["Customers"].mean()

promo_sales = df.groupby("Promo")["Sales"].mean()

store_type_average = df.groupby("StoreType")["Sales"].mean()

store_type_total = (
    df.groupby("StoreType")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

customer_sales_corr = df[["Customers", "Sales"]].corr().iloc[0, 1]

year_sales = df.groupby("Year")["Sales"].mean()

month_sales = df.groupby("Month")["Sales"].mean()

monthly_sales = (
    df.groupby("YearMonth")["Sales"]
    .mean()
)

store_sales = df.groupby("Store")["Sales"].mean()

top_stores = (
    store_sales
    .sort_values(ascending=False)
    .head(10)
)

bottom_stores = (
    store_sales
    .sort_values(ascending=True)
    .head(10)
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if section == "Executive Overview":

    st.header("Executive Overview")

    st.markdown(
        """
        This section gives a quick summary of the complete analysis.
        The objective is not only to display numbers, but to identify
        patterns that can help explain sales performance.
        """
    )

    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Average Sales",
            f"{average_sales:,.0f}"
        )

    with col3:
        st.metric(
            "Average Customers",
            f"{average_customers:,.0f}"
        )

    with col4:
        st.metric(
            "Customer-Sales Correlation",
            f"{customer_sales_corr:.3f}"
        )

    st.divider()

    # -----------------------------------------------------
    # Key findings
    # -----------------------------------------------------

    st.subheader("Key Findings")

    # Promo difference
    promo_difference = (
        promo_sales.get(1, np.nan)
        - promo_sales.get(0, np.nan)
    )

    # Store type with highest total sales
    highest_revenue_type = store_type_total.index[0]

    # Store type with highest average sales
    highest_average_type = store_type_average.idxmax()

    # Best month
    best_month = month_sales.idxmax()

    # Best year
    best_year = year_sales.idxmax()

    findings = [
        f"Average sales during promotion records were "
        f"{promo_sales.get(1, 0):,.0f}, compared with "
        f"{promo_sales.get(0, 0):,.0f} without promotion.",

        f"Customer count has a strong positive observed relationship "
        f"with sales, with a correlation of {customer_sales_corr:.3f}.",

        f"Store Type {highest_revenue_type.upper()} contributes the "
        f"highest total sales across the complete dataset.",

        f"Store Type {highest_average_type.upper()} has the highest "
        f"average sales per record.",

        f"Month {best_month} has the highest average sales in the "
        f"month-wise analysis.",

        f"{best_year} has the highest average sales among the available years."
    ]

    for finding in findings:
        st.markdown(f"- {finding}")

    st.divider()

    # -----------------------------------------------------
    # Important distinction
    # -----------------------------------------------------

    st.subheader("Average Sales vs Total Sales")

    st.warning(
        """
        Important distinction:

        **Average sales** tells us how strongly a store type performs
        per observation.

        **Total sales** tells us how much revenue that store type
        contributes across all observations.

        Therefore, the store type with the highest average sales
        does not necessarily have the highest total revenue.
        """
    )

    comparison = pd.DataFrame({
        "Store Type": store_type_average.index,
        "Average Sales": store_type_average.values,
        "Total Sales": [
            store_type_total.get(x, 0)
            for x in store_type_average.index
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SALES TRENDS
# =========================================================

elif section == "Sales Trends":

    st.header("📈 Sales Trends")

    st.markdown(
        """
        The time-based analysis helps us understand whether sales
        remain stable or show seasonal/periodic changes.
        """
    )

    # -----------------------------------------------------
    # Monthly Average Sales
    # -----------------------------------------------------

    st.subheader("Monthly Average Sales Trend")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values
    )

    ax.set_title("Monthly Average Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown(
        """
        **What does this show?**

        The line represents the average sales recorded for each
        year-month period.

        **Why does this graph change?**

        The data shows that sales are not constant throughout the year.
        Different months have different average sales levels.

        **Business interpretation:**

        This indicates that seasonal or time-related factors may be
        associated with sales variation.

        **Possible solution:**

        Businesses can use historical monthly patterns for:

        - inventory planning
        - staff planning
        - promotional planning
        - demand forecasting

        This analysis identifies a pattern; it does not by itself prove
        that the month causes the change.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Year-wise
    # -----------------------------------------------------

    st.subheader("Year-wise Average Sales")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        year_sales.index.astype(str),
        year_sales.values
    )

    ax.set_title("Average Sales by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Sales")

    st.pyplot(fig)

    best_year_value = year_sales.max()

    st.info(
        f"""
        **Observation:** {int(year_sales.idxmax())} has the highest
        average sales among the available years at approximately
        {best_year_value:,.0f} per record.

        **Interpretation:** Average sales show an upward movement
        across the available yearly observations, although the
        difference between years is relatively moderate.

        **Action:** Historical yearly trends can be used as a
        reference for future sales planning.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Month-wise
    # -----------------------------------------------------

    st.subheader("Month-wise Average Sales")

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    month_display = month_sales.rename(index=month_names)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        month_display.index,
        month_display.values
    )

    ax.set_title("Month-wise Average Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown(
        f"""
        **Highest average-sales month:** {month_names[month_sales.idxmax()]}

        **Lowest average-sales month:** {month_names[month_sales.idxmin()]}

        This pattern can help businesses identify periods where
        demand historically tends to be higher or lower.
        """
    )


# =========================================================
# WHAT AFFECTS SALES
# =========================================================

elif section == "What Affects Sales?":

    st.header("🔍 What Affects Sales?")

    st.markdown(
        """
        This section moves beyond simply showing charts.

        We examine relationships in the data to understand which
        variables are associated with sales and how those relationships
        may explain the observed patterns.
        """
    )

    # -----------------------------------------------------
    # Customers vs Sales
    # -----------------------------------------------------

    st.subheader("1. Customers vs Sales")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Customers"],
        df["Sales"],
        alpha=0.2
    )

    ax.set_title("Customers vs Sales")
    ax.set_xlabel("Customers")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    st.metric(
        "Customer-Sales Correlation",
        f"{customer_sales_corr:.3f}"
    )

    st.markdown(
        f"""
        **What do we observe?**

        The correlation between customers and sales is
        **{customer_sales_corr:.3f}**, which indicates a strong
        positive linear relationship in this dataset.

        In simple terms:

        **More customers are generally associated with higher sales.**

        **What may explain this?**

        A larger number of customers means more transactions and/or
        purchasing activity, which naturally tends to increase sales.

        **Business solution:**

        Stores can monitor customer traffic as an important sales
        indicator. If customer traffic falls, businesses can investigate
        reasons such as weak promotions, low demand, or operational issues.

        ⚠️ Correlation does not prove that customer count alone causes
        every change in sales.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Promo
    # -----------------------------------------------------

    st.subheader("2. Promotion and Sales")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        x="Promo",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_title("Sales Distribution by Promotion")
    ax.set_xlabel("Promo (0 = No, 1 = Yes)")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    promo_no = promo_sales.get(0, 0)
    promo_yes = promo_sales.get(1, 0)

    st.markdown(
        f"""
        **Observed average sales:**

        - No promotion: **{promo_no:,.0f}**
        - Promotion: **{promo_yes:,.0f}**

        Promotion records therefore have substantially higher average
        sales in this dataset.

        **What does this indicate?**

        Promotions are strongly associated with higher sales.

        **Possible explanation:**

        Promotions may encourage customers to visit stores or purchase
        more products.

        **Business solution:**

        Businesses can compare promotion performance and identify which
        promotional periods generate strong sales.

        However, this analysis alone does not establish that promotion
        is the only reason for the difference.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Day + Promo
    # -----------------------------------------------------

    st.subheader("3. Sales by Day of Week and Promotion")

    day_promo = (
        df.groupby(
            ["DayOfWeek", "Promo"]
        )["Sales"]
        .mean()
        .reset_index()
    )

    pivot_day_promo = day_promo.pivot(
        index="DayOfWeek",
        columns="Promo",
        values="Sales"
    )

    pivot_day_promo.columns = [
        "No Promo",
        "Promo"
    ]

    st.dataframe(
        pivot_day_promo.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    pivot_day_promo.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Average Sales by Day of Week and Promotion")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Sales")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    st.markdown(
        """
        **What does this tell us?**

        The effect of promotion is not identical across every day.

        This means businesses should not look only at the overall
        promotion average. Day-level behavior can also be important.

        **Business solution:**

        Promotional campaigns can be evaluated separately by day of week
        to identify combinations where promotional activity and customer
        demand are strongest.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Correlation Heatmap
    # -----------------------------------------------------

    st.subheader("4. Correlation Heatmap")

    correlation_columns = [
        "Sales",
        "Customers",
        "Promo",
        "Open",
        "SchoolHoliday",
        "CompetitionDistance"
    ]

    corr_matrix = df[correlation_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

    st.markdown(
        """
        **How to read this heatmap:**

        - Values close to **+1** indicate a strong positive relationship.
        - Values close to **-1** indicate a strong negative relationship.
        - Values close to **0** indicate a weak linear relationship.

        The strongest relationship with Sales in this analysis is
        observed with Customers.

        **Business implication:**

        Customer traffic is an important variable to monitor when
        understanding sales performance.

        Correlation should be interpreted as an association, not proof
        of direct causation.
        """
    )


# =========================================================
# STORE ANALYSIS
# =========================================================

elif section == "Store Analysis":

    st.header("🏪 Store Performance Analysis")

    st.markdown(
        """
        Store-level analysis helps identify differences between
        store types and individual stores.
        """
    )

    # -----------------------------------------------------
    # Store Type Boxplot
    # -----------------------------------------------------

    st.subheader("Sales Distribution by Store Type")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        x="StoreType",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_title("Sales Distribution by Store Type")
    ax.set_xlabel("Store Type")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    st.markdown(
        """
        **Observation:**

        Store types have different sales distributions.

        Store Type B has the highest average sales per record, while
        the other store types have lower average values.

        **Important:**

        This does not mean Store Type B generates the highest total
        revenue. Total revenue also depends on how many observations
        and stores belong to each type.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Store Type Average Table
    # -----------------------------------------------------

    st.subheader("Average Sales by Store Type")

    average_type_table = (
        store_type_average
        .sort_values(ascending=False)
        .reset_index()
    )

    average_type_table.columns = [
        "Store Type",
        "Average Sales"
    ]

    st.dataframe(
        average_type_table.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # Total Revenue Contribution
    # -----------------------------------------------------

    st.subheader("Total Sales Contribution by Store Type")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        store_type_total.values,
        labels=store_type_total.index,
        autopct="%1.1f%%"
    )

    ax.set_title("Total Sales Contribution by Store Type")

    st.pyplot(fig)

    revenue_percent = (
        store_type_total /
        store_type_total.sum() *
        100
    )

    revenue_table = pd.DataFrame({
        "Store Type": store_type_total.index,
        "Total Sales": store_type_total.values,
        "Contribution (%)": revenue_percent.values
    })

    st.dataframe(
        revenue_table.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        f"""
        **Revenue insight:**

        Store Type **{store_type_total.index[0].upper()}** contributes
        the largest total sales amount in the complete dataset.

        This is different from average sales per record.

        **Business implication:**

        When making business decisions, both metrics should be considered:

        - Average sales → individual performance
        - Total sales → overall revenue contribution
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Top 10
    # -----------------------------------------------------

    st.subheader("Top 10 Stores by Average Sales")

    fig, ax = plt.subplots(figsize=(10, 5))

    top_stores.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Top 10 Stores by Average Sales")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")

    st.pyplot(fig)

    st.markdown(
        """
        **What does this tell us?**

        These stores have the highest average sales across their
        observations.

        **Possible use:**

        Their patterns can be studied to understand what operational,
        customer, promotional, or location-related characteristics
        may be associated with stronger performance.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Bottom 10
    # -----------------------------------------------------

    st.subheader("Bottom 10 Stores by Average Sales")

    fig, ax = plt.subplots(figsize=(10, 5))

    bottom_stores.sort_values().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Bottom 10 Stores by Average Sales")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")

    st.pyplot(fig)

    st.markdown(
        """
        **What does this tell us?**

        These stores have the lowest average sales.

        **Possible reasons to investigate:**

        The dataset contains variables such as:

        - customer count
        - promotion
        - store type
        - competition distance
        - holidays
        - opening status

        These variables can be investigated to understand why
        individual stores perform differently.

        **Business solution:**

        Low-performing stores should be investigated individually
        instead of applying the same solution to every store.
        """
    )


# =========================================================
# BUSINESS INSIGHTS & SOLUTIONS
# =========================================================

elif section == "Business Insights & Solutions":

    st.header("💡 Business Insights & Solutions")

    st.markdown(
        """
        This is the main conclusion of the project.

        The objective of EDA is not simply to produce graphs.
        The graphs are used to identify patterns and convert those
        patterns into practical business insights.
        """
    )

    # -----------------------------------------------------
    # Insight 1
    # -----------------------------------------------------

    st.subheader("Insight 1 — Customer Traffic is Strongly Associated with Sales")

    st.markdown(
        f"""
        **Evidence**

        Customer-Sales correlation = **{customer_sales_corr:.3f}**

        **Observation**

        Stores with more customers generally record higher sales.

        **What may explain it?**

        Higher customer traffic creates more opportunities for purchases.

        **Recommended action**

        Monitor customer traffic alongside sales rather than looking
        at sales alone.

        A sudden fall in customer traffic can be treated as an early
        signal for investigation.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Insight 2
    # -----------------------------------------------------

    st.subheader("Insight 2 — Promotions are Associated with Higher Sales")

    st.markdown(
        f"""
        **Evidence**

        Average sales without promotion:

        **{promo_sales.get(0, 0):,.0f}**

        Average sales with promotion:

        **{promo_sales.get(1, 0):,.0f}**

        **Observation**

        Promotion records have considerably higher average sales.

        **Possible explanation**

        Promotions may increase customer visits and/or purchasing
        activity.

        **Recommended action**

        Promotions should be evaluated using sales results and
        customer response, rather than applying the same promotion
        everywhere.

        **Important limitation**

        This analysis shows an association. It does not prove that
        promotion alone caused the increase.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Insight 3
    # -----------------------------------------------------

    st.subheader("Insight 3 — Sales Vary Across Time")

    st.markdown(
        f"""
        **Observation**

        Different months and years have different average sales levels.

        The highest month-wise average occurs in:

        **{month_names[month_sales.idxmax()]}**

        The highest yearly average occurs in:

        **{int(year_sales.idxmax())}**

        **Business meaning**

        Demand is not completely uniform throughout the year.

        **Recommended action**

        Historical sales trends can be used for:

        - inventory planning
        - workforce planning
        - promotional scheduling
        - demand forecasting
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Insight 4
    # -----------------------------------------------------

    st.subheader("Insight 4 — Store Type Performance is Different")

    avg_best = store_type_average.idxmax()
    total_best = store_type_total.idxmax()

    st.markdown(
        f"""
        **Average-sales perspective**

        Store Type **{avg_best.upper()}** has the highest average sales
        per observation.

        **Total-revenue perspective**

        Store Type **{total_best.upper()}** contributes the highest
        total sales across the dataset.

        **Why are these different?**

        Average performance and total revenue answer two different
        business questions.

        A store type can have strong sales per observation but still
        contribute less total revenue if it has fewer observations.

        **Recommended action**

        Business decisions should consider both average performance
        and total revenue instead of relying on only one metric.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Insight 5
    # -----------------------------------------------------

    st.subheader("Insight 5 — Individual Stores Need Different Attention")

    st.markdown(
        f"""
        The analysis identifies clear differences between individual
        stores.

        Highest average-sales store:

        **Store {int(top_stores.index[0])}**

        Average sales:

        **{top_stores.iloc[0]:,.0f}**

        Lowest average-sales store:

        **Store {int(bottom_stores.index[0])}**

        Average sales:

        **{bottom_stores.iloc[0]:,.0f}**

        **Recommended action**

        Instead of applying one strategy to every store:

        1. Study high-performing stores to identify useful patterns.
        2. Investigate low-performing stores separately.
        3. Compare customer traffic and promotion activity.
        4. Examine competition and store characteristics.
        5. Use these findings for targeted interventions.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Final Solution
    # -----------------------------------------------------

    st.subheader("🎯 Overall Business Solution")

    st.success(
        """
        The analysis suggests that sales performance should be managed
        using multiple signals rather than one metric.

        A practical monitoring approach would combine:

        **Customer Traffic**
        ↓
        **Promotion Activity**
        ↓
        **Store Performance**
        ↓
        **Time / Seasonal Pattern**
        ↓
        **Sales**

        This can help management identify unusual performance,
        evaluate promotional periods, and focus attention on
        stores that need further investigation.
        """
    )

    st.caption(
        "Note: These recommendations are based on observed patterns "
        "in the dataset. Correlation and descriptive analysis do not "
        "establish causation."
    )


# =========================================================
# ORIGINAL KAGGLE VISUALIZATIONS
# =========================================================

elif section == "Original Kaggle Visualizations":

    st.header("📊 Original Kaggle Analysis")

    st.markdown(
        """
        This section reproduces the main visualizations from the
        original Kaggle EDA so that the Streamlit application remains
        consistent with the project analysis.
        """
    )

    # -----------------------------------------------------
    # 1 Monthly Trend
    # -----------------------------------------------------

    st.subheader("1. Monthly Average Sales Trend")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values
    )

    ax.set_title("Monthly Average Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Sales")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -----------------------------------------------------
    # 2 Sales Distribution
    # -----------------------------------------------------

    st.subheader("2. Sales Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["Sales"],
        bins=50
    )

    ax.set_title("Sales Distribution")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 3 Customers vs Sales
    # -----------------------------------------------------

    st.subheader("3. Customers vs Sales")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["Customers"],
        df["Sales"],
        alpha=0.2
    )

    ax.set_title("Customers vs Sales")
    ax.set_xlabel("Customers")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 4 Store Type
    # -----------------------------------------------------

    st.subheader("4. Sales Distribution by Store Type")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        x="StoreType",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_title("Sales Distribution by Store Type")
    ax.set_xlabel("Store Type")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 5 Day of Week
    # -----------------------------------------------------

    st.subheader("5. Sales Distribution by Day of Week")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        x="DayOfWeek",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_title("Sales Distribution by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 6 Promo
    # -----------------------------------------------------

    st.subheader("6. Sales Distribution by Promotion")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        x="Promo",
        y="Sales",
        data=df,
        ax=ax
    )

    ax.set_title("Sales Distribution by Promotion")
    ax.set_xlabel("Promo (0 = No, 1 = Yes)")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 7 Correlation Heatmap
    # -----------------------------------------------------

    st.subheader("7. Correlation Heatmap")

    correlation_columns = [
        "Sales",
        "Customers",
        "Promo",
        "Open",
        "SchoolHoliday",
        "CompetitionDistance"
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df[correlation_columns].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 8 Top Stores
    # -----------------------------------------------------

    st.subheader("8. Top 10 Stores by Average Sales")

    fig, ax = plt.subplots(figsize=(10, 5))

    top_stores.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Top 10 Stores by Average Sales")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 9 Bottom Stores
    # -----------------------------------------------------

    st.subheader("9. Bottom 10 Stores by Average Sales")

    fig, ax = plt.subplots(figsize=(10, 5))

    bottom_stores.sort_values().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Bottom 10 Stores by Average Sales")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average Sales")

    st.pyplot(fig)

    # -----------------------------------------------------
    # 10 Store Type Revenue Pie
    # -----------------------------------------------------

    st.subheader("10. Total Sales Contribution by Store Type")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        store_type_total.values,
        labels=store_type_total.index,
        autopct="%1.1f%%"
    )

    ax.set_title("Total Sales Contribution by Store Type")

    st.pyplot(fig)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Retail Sales Trend Analyzer | Data analysis based on the Rossmann Store Sales dataset"
)
