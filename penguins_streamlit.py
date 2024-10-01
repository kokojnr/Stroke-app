import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Stroke Prediction Web App")

st.write(
    """This app uses 6 inputs to predict
     the probability of stroke"""
)




penguin_file = st.file_uploader("Upload your own Stroke data")

if penguin_file is None:
    rf_pickle = open("random_forest_penguin.pickle", "rb")
    map_pickle = open("output_penguin.pickle", "rb")
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
    penguin_df = pd.read_csv("penguins.csv")
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df["stroke"]
    features = penguin_df[
        [
           "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "avg_glucose_level",
        "work_type",
        ]
    ]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write(
        f"""We trained a Random Forest model on these data,
        it has a score of {score}! Use the
        inputs below to try out the model"""
    )

with st.form("user_inputs"):
    hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", options=["Yes", "No"])
    work_type = st.selectbox("Work Type", options=['Private', 'Self-employed', 'Govt_job', 'children','Never_worked'])
    avg_glucose = st.number_input("Average Glucose Level", min_value=0)
    age = st.number_input("Age", min_value=0)
    bmi= st.number_input("Body Mass Index", min_value=0)
    st.form_submit_button()
le = LabelEncoder()
hypertension = le.fit_transform([hypertension])[0]
heart_disease = le.fit_transform([heart_disease])[0]
work_type = le.fit_transform([work_type])[0]

new_prediction = rfc.predict(
    [
        [
            avg_glucose,
            age,
            bmi,
            hypertension,
            heart_disease,
            work_type
        ]
    ]
)
st.subheader("Predicting stroke:")
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"Stroke Risk {prediction_species}")

# Create buttons to initiate record
tricep_button = st.button("Record Tricep")
bicep_button = st.button("Record Bicep")

# Create a text area to display the sampling status
sampling_status = st.text_area("Record Status:", height=10)

# Define a function to send a command to the ESP32 board
#def send_command(command):
    # Replace with your ESP32 board serial port
#   serial_port = pyserial.Serial('COM4', 9600, timeout=1)
#    serial_port.write(command.encode() + b'\n')
#   serial_port.close()

# Handle button clicks
if tricep_button:
    send_command("sample_tricep")
    sampling_status.text = "Sampling Tricep..."
if bicep_button:
    send_command("sample_bicep")
    sampling_status.text = "Sampling Bicep..."


