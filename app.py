import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import joblib


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


@st.cache_resource
def load_sales_model():
    """Load the Random Forest model trained in the Kaggle notebook."""
    model_path = "tiny_random_forest.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "tiny_random_forest.pkl was not found. Upload it to the "
            "same GitHub repository folder as app.py."
        )
    return joblib.load(model_path)


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
        "Sales Prediction",
        "Model Performance",
        "Findings"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Filters")

years = sorted(
    df["Year"].unique()
)

selected_years = st.sidebar.multiselect(
    "Year",
    years,
    default=years
)

store_types = sorted(
    df["StoreType"].unique()
)

selected_store_types = st.sidebar.multiselect(
    "Store Type",
    store_types,
    default=store_types
)

promo_options = st.sidebar.multiselect(
    "Promotion",
    [0, 1],
    default=[0, 1],
    format_func=lambda x:
        "No Promo" if x == 0 else "Promo"
)


filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["StoreType"].isin(selected_store_types)) &
    (df["Promo"].isin(promo_options))
].copy()


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

    plt.close(fig)

    if not monthly_sales.empty:

        st.caption(
            f"Highest monthly average: "
            f"{monthly_sales.idxmax()} "
            f"({monthly_sales.max():,.0f})."
        )

    st.subheader("Average Sales by Year")

    fig, ax = plt.subplots(
        figsize=(6, 2.5)
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

    plt.close(fig)

    if not year_sales.empty:

        st.caption(
            f"{year_sales.idxmax()} has the highest "
            f"average sales in the selected data."
        )

    st.subheader("Average Sales by Month")

    fig, ax = plt.subplots(
        figsize=(7, 2.5)
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

    plt.close(fig)

    if not month_sales.empty:

        st.caption(
            f"Month {month_sales.idxmax()} has the highest "
            f"average sales."
        )


elif page == "Sales Drivers":

    st.header("Sales Drivers")

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

    plt.close(fig)

    st.markdown(
        f"**Correlation: {customer_sales_corr:.3f}** — "
        "higher customer counts are generally associated "
        "with higher sales."
    )

    st.subheader("Promotion vs Sales")

    fig, ax = plt.subplots(
        figsize=(5.5, 2.5)
    )

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

    fig, ax = plt.subplots(
        figsize=(7, 2.5)
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
        figsize=(6.5, 3.5)
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

    store_avg = selected_store_data[
        "Sales"
    ].mean()

    store_customers = selected_store_data[
        "Customers"
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
            figsize=(4, 3)
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
        figsize=(7, 2.5)
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
        figsize=(7, 2.5)
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



elif page == "Sales Prediction":

    st.header("Sales Prediction")
    st.write(
        "Enter store and trading-day details to estimate sales "
        "using the trained Random Forest model."
    )

    try:
        sales_model = load_sales_model()
    except Exception as e:
        st.error(f"Could not load the sales prediction model: {e}")
        st.stop()

    st.subheader("Prediction Inputs")

    with st.form("sales_prediction_form"):
        input_col1, input_col2, input_col3 = st.columns(3)

        with input_col1:
            prediction_store = st.number_input(
                "Store ID", min_value=1, max_value=1115,
                value=1, step=1
            )
            prediction_day = st.selectbox(
                "Day of Week", options=[1, 2, 3, 4, 5, 6, 7],
                format_func=lambda x: {
                    1: "1 - Monday", 2: "2 - Tuesday",
                    3: "3 - Wednesday", 4: "4 - Thursday",
                    5: "5 - Friday", 6: "6 - Saturday",
                    7: "7 - Sunday"
                }[x]
            )
            prediction_customers = st.number_input(
                "Expected Customers", min_value=0, max_value=10000,
                value=500, step=10
            )

        with input_col2:
            prediction_promo = st.selectbox(
                "Promotion", options=[0, 1],
                format_func=lambda x: "No Promo" if x == 0 else "Promo"
            )
            prediction_distance = st.number_input(
                "Competition Distance", min_value=0.0,
                max_value=1000000.0, value=2325.0, step=100.0
            )
            prediction_month = st.selectbox(
                "Month", options=list(range(1, 13)),
                format_func=lambda x: f"{x} - {pd.Timestamp(2000, x, 1).strftime('%B')}"
            )

        with input_col3:
            prediction_year = st.selectbox(
                "Year", options=[2013, 2014, 2015], index=2
            )
            prediction_store_type = st.selectbox(
                "Store Type", options=["a", "b", "c", "d"],
                format_func=lambda x: f"Type {x.upper()}"
            )

        predict_clicked = st.form_submit_button(
            "Predict Sales", type="primary", use_container_width=True
        )

    if predict_clicked:
        # These column names and their order must match model training.
        model_input = pd.DataFrame([{
            "Store": int(prediction_store),
            "DayOfWeek": int(prediction_day),
            "Customers": int(prediction_customers),
            "Promo": int(prediction_promo),
            "CompetitionDistance": float(prediction_distance),
            "Month": int(prediction_month),
            "Year": int(prediction_year),
            "StoreType_b": int(prediction_store_type == "b"),
            "StoreType_c": int(prediction_store_type == "c"),
            "StoreType_d": int(prediction_store_type == "d")
        }], columns=[
            "Store", "DayOfWeek", "Customers", "Promo",
            "CompetitionDistance", "Month", "Year",
            "StoreType_b", "StoreType_c", "StoreType_d"
        ])

        try:
            predicted_sales = float(sales_model.predict(model_input)[0])
            st.divider()
            st.subheader("Estimated Sales")
            st.metric("Predicted Sales", f"{max(0, predicted_sales):,.0f}")
            st.caption(
                "This is a model-based estimate from historical Rossmann data, "
                "not a guaranteed actual sales figure. Accuracy can vary with "
                "the store and input values."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# 7. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")
    st.write(
        "This page explains how the trained Random Forest model performed "
        "on the held-out test set in the Kaggle notebook."
    )

    try:
        sales_model = load_sales_model()
    except Exception as e:
        st.error(f"Could not load the sales prediction model: {e}")
        st.stop()

    # These are the test-set metrics recorded when this small model was
    # evaluated in the Kaggle notebook. They are not recalculated on this page.
    mae = 646.96
    rmse = 968.07
    r2 = 0.93663

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:,.2f}")
    m2.metric("RMSE", f"{rmse:,.2f}")
    m3.metric("R² Score", f"{r2:.4f}")

    st.markdown(
        "- **MAE:** average absolute difference between predicted and actual sales.\n"
        "- **RMSE:** prediction error metric that penalizes larger errors more strongly.\n"
        "- **R²:** proportion of variation explained by the model on the test set; closer to 1 is better."
    )

    st.info(
        "These values are the previously recorded Kaggle test-set results for "
        "the 20-tree Random Forest model. They are shown as saved evaluation "
        "results, not recomputed from the full dataset inside this dashboard."
    )

    st.subheader("Random Forest Feature Importance")

    if hasattr(sales_model, "feature_importances_"):
        feature_names = getattr(
            sales_model,
            "feature_names_in_",
            np.array([
                "Store", "DayOfWeek", "Customers", "Promo",
                "CompetitionDistance", "Month", "Year",
                "StoreType_b", "StoreType_c", "StoreType_d"
            ])
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": sales_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Input Feature")
        ax.set_title("Which inputs influenced the model most?")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Feature importance describes how much the model used each feature "
            "to make decisions. It does not prove that a feature causes sales to change."
        )
    else:
        st.warning(
            "Feature importance is unavailable for the loaded model type."
        )


st.divider()

st.caption(
    "Retail Sales Trend Analyzer • "
    "Python • Pandas • Matplotlib • Seaborn • Streamlit"
)
