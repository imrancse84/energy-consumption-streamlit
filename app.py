import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("⚡ Energy Consumption Prediction")

st.write("**Student ID: 223002020**")

# Sample Data
data = {
    "Temperature": [20, 22, 25, 27, 30, 32, 35, 18, 24, 28],
    "Humidity": [30, 35, 40, 45, 50, 55, 60, 25, 38, 48],
    "Energy": [200, 220, 260, 300, 350, 400, 450, 180, 240, 310]
}

df = pd.DataFrame(data)

X = df[["Temperature", "Humidity"]]
y = df["Energy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.sidebar.header("Input Values")

temp = st.sidebar.slider("Temperature", 15, 40, 25)
humidity = st.sidebar.slider("Humidity", 20, 70, 40)

if st.button("Predict"):
    prediction = model.predict([[temp, humidity]])
    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f}")

    y_pred = model.predict(X_test)
    st.write("### Model Performance")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))