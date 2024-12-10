import streamlit as st
import pandas as pd
import pickle
import numpy as np




# Load the pre-trained model from a pickle file
@st.cache_resource
def load_model_one():
    with open('Mye_pca.pkl', 'rb') as file:
        model_one = pickle.load(file)
    return model_one

def load_model_two():
    with open('Mye_lda.pkl', 'rb') as file:
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
    st.title("Myeloma Prediction App")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


    

    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, header=None)
        #w = df[0]

        if df.ndim == 1:
            specs = df[1].values[1:].reshape((1, 6639))
            names = pd.DataFrame(df.values[:1,1:].T, columns=['names']) 
        else: 
            specs = df.values[1:,1:].T
            names = pd.DataFrame(df.values[:1,1:].T, columns=['names'])
       

              
                
            # Transform using the PCA
            model_1 = load_model_one()
            value_1 = transform_1(model_1, specs)
        
            # Make predictions with the LDA
            model_2 = load_model_two()
            value_2 = transform_2(model_2, value_1)

            converted = pd.DataFrame(np.where(value_2 == 1, "Disease", "Control"), columns=['Predictions'])
            labels = pd.DataFrame(value_2, columns=['Pred Label'])
        
        
            
            pred_df = pd.concat([names, converted, labels],axis=1)

        st.write(pred_df)    

                       
        # Make prediction
        
        #if value_2 == 0:
        #    st.write("Prediction: Control")
        #elif value_2 == 1:
        #    st.write("Prediction: Disease")
        #else: 
        #    st.write("Error")


        
        

if __name__ == "__main__":
    main()
