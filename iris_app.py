

import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris

# 1. Load the saved model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

# 2. Load flower names and feature names
iris = load_iris()
flower_names = iris.target_names
feature_names = iris.feature_names

# 3. Design the app
st.title("🌸 Iris Flower Predictor")
st.write("Enter the measurements to predict the flower species.")

st.divider()

# 4. Get measurements from the user
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal length (cm)", 0.0, 10.0, 5.1
    )
    sepal_width = st.number_input(
        "Sepal width (cm)", 0.0, 10.0, 3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal length (cm)", 0.0, 10.0, 1.4
    )
    petal_width = st.number_input(
        "Petal width (cm)", 0.0, 10.0, 0.2
    )

# 5. Predict the flower
if st.button("Predict Flower", type="primary"):
    measurements = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    scaled_measurements = scaler.transform(measurements)

    prediction = model.predict(scaled_measurements)[0]
    probabilities = model.predict_proba(scaled_measurements)[0]

    predicted_name = flower_names[prediction]

    # 6. Display prediction
    st.success(f"🌼 Predicted flower: {predicted_name.title()}")

    # 7. Display model probabilities
    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Flower": flower_names,
        "Probability (%)": probabilities * 100
    })

    st.bar_chart(
        probability_df.set_index("Flower")
    )

    st.dataframe(
        probability_df.style.format({
            "Probability (%)": "{:.2f}%"
        }),
        hide_index=True
    )

    # 8. Display entered measurements
    st.subheader("Your Flower Measurements")

    measurement_df = pd.DataFrame({
        "Measurement": feature_names,
        "Value (cm)": measurements[0]
    })

    st.bar_chart(
        measurement_df.set_index("Measurement")
    )

    st.caption(
        "Probabilities are model estimates, not guarantees "
        "of prediction accuracy."
    )
