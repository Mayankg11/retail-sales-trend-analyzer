elif page == "Sales Prediction":
    st.header("Sales Prediction")

    store = st.number_input("Store", min_value=1, max_value=1115, value=1)
    day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
    customers = st.number_input("Customers", min_value=0, value=500)
    promo = st.selectbox("Promotion", [0, 1])
    competition_distance = st.number_input(
        "Competition Distance",
        min_value=0.0,
        value=2325.0
    )
    month = st.selectbox("Month", list(range(1, 13)))
    year = st.selectbox("Year", [2013, 2014, 2015])
    store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])

    if st.button("Predict Sales"):
        store_type_b = 1 if store_type == "b" else 0
        store_type_c = 1 if store_type == "c" else 0
        store_type_d = 1 if store_type == "d" else 0

        input_data = pd.DataFrame([[
            store,
            day_of_week,
            customers,
            promo,
            competition_distance,
            month,
            year,
            store_type_b,
            store_type_c,
            store_type_d
        ]], columns=[
            "Store",
            "DayOfWeek",
            "Customers",
            "Promo",
            "CompetitionDistance",
            "Month",
            "Year",
            "StoreType_b",
            "StoreType_c",
            "StoreType_d"
        ])

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Sales: {prediction:.0f}")
