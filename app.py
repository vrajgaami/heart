import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title(" Heart Diseases")

# Input fields for each parameter
age = st.number_input("age", min_value=0.0, max_value=100.0, value=1.0)
sex = st.number_input("sex",min_value=0.0, max_value=1.0, value=1.0)
cp = st.number_input("cp", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
chol = st.number_input("chol", min_value=0, max_value=1000, value=1.0)
fbs = st.number_input("fbs", min_value=0.0, max_value=100.0, value=.10)
restecg = st.number_input("restecg", min_value=0.0, max_value=500.0, value=7.25, step=0.1)
thalach = st.number_input("thalach",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
exang= st.number_input("exang",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
oldpeak = st.number_input("exang",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
slope = st.number_input("slope",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
ca = st.number_input("ca",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
thal = st.number_input("thal",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
target = st.number_input("target",min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Create the input dictionary for prediction
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'fbs': fbs,
    'restecg': restecg,
    'exang': exang,
    'oldpeak':oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': target
}



# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_target(input_data)

        if pred == 1:
            # heart attack y
            st.success(f"Prediction: heart attack  is possible with probability {prob:.2f}")
        else:
            # heart attack n
            st.error(f"Prediction: heart attack  is not possible with probability {prob:.2f}")
