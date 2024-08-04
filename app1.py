import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained SVM model and label encoder
model_path = 'svc_model.joblib'
classes_path = 'label_encoder_classes.npy'

# Load the SVM model
svc_model = joblib.load(model_path)

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(classes_path, allow_pickle=True)

# Define a function to make predictions
def make_prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Reshape for the model
    prediction = svc_model.predict(input_data)
    predicted_class = prediction[0]
    return label_encoder.inverse_transform([predicted_class])[0]

# Streamlit app title
st.title('🌸 Iris Flower species Classification')

# Input fields for features
st.write("Enter the feature values:")
feature1 = st.number_input('Sepal Length')
feature2 = st.number_input('Sepal Width')
feature3 = st.number_input('Petal Length')
feature4 = st.number_input('Petal Width')

# Button to make predictions
if st.button('Predict'):
    input_data = [feature1, feature2, feature3, feature4]
    try:
        result = make_prediction(input_data)
        st.header(f'{result}')
    except Exception as e:
        st.write(f"Error making prediction: {e}")
