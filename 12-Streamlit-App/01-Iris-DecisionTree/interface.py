import pickle 
import streamlit as st

st.title("IRIS Flower Species Predictor")

st.write("""
This application predicts the species of an IRIS flower based on its features.
Please input the following parameters:
""")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]   

with open("decision_tree_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

if st.button("Predict Species"):
    with st.spinner("Predicting..."):
        prediction = model.predict(input_data)
        species = label_encoder.inverse_transform(prediction)
        st.success(f"The predicted species is: {species[0]}")