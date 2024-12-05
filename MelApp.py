import streamlit as st
import pandas as pd
import pickle
import numpy as np




# Load the pre-trained model from a pickle file
@st.cache_resource
def load_model_one():
    with open('Mel_pca.pkl', 'rb') as file:
        model_one = pickle.load(file)
    return model_one

def load_model_two():
    with open('Mel_lda.pkl', 'rb') as file:
        model_two = pickle.load(file)
    return model_two

# Function to make predictions
def transform_1(model_one, data_1):
    values_1 = model_one.transform(data_1)
    return values_1

def transform_2(model_two, data_2):
    values_2 = model_two.predict(data_2)
    return values_2

# Streamlit app
def main():
    st.title("Melanoma Prediction App")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


    

    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, header=None)
        #w = df[0]
        spec = df[1].values.reshape((1, 6639)) 

              
                
        # Transform using the PCA
        model_1 = load_model_one()
        value_1 = transform_1(model_1, spec)
        
        # Make predictions with the LDA
        model_2 = load_model_two()
        value_2 = transform_2(model_2, value_1)

                       
        # Make prediction
        
        if value_2 == 0:
            st.write("Prediction: Control")
        elif value_2 == 1:
            st.write("Prediction: Disease")
        else: 
            st.write("Error")


        
        

if __name__ == "__main__":
    main()
