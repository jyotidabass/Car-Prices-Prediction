# Importing Libraries

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Loading the model

data = joblib.load('data.joblib')
pipe = data['pipe']
makers_models = data['makers_&_models']
makers = makers_models.keys()

# Create the Page


# creating two cols
col1, col2 = st.columns([1, 3], gap='small', border=False, vertical_alignment='top')

# Insert an Image Under col1
with col1:
    st.image("Car Images /logo.png")

# Insert a Title Under Col2

with col2:
    st.title(':car: :green[Car Price Predictor]', )

# Creating an Expander

with st.expander(':violet[***ABOUT THE PROJECT***]'):

    # Writing About the Project
    st.write('**:car: :green[Car Price Predictor]** is a **Supervised Machine Learning Model** which predicts the **Price of Cars** based on three features namely **CAR MAKER, MODEL AND MILEAGE**.')
    st.write('**:blue[Linear Regression]** algorithm is working behind this model.')

# Show the Model

# create a header for the model
st.success(f'**MODEL**', icon=':material/model_training:')

# create a container for input and output
with st.container():

    col1, col2 = st.columns([2.5,1], border=True,vertical_alignment='top')

    # use col1 for input
    with col1:
        st.info('**INPUT**', icon=':material/input:')

        inp1, inp2= st.columns(2, border=True)
        inp3, inp4 = st.columns(2, border=True)

        with inp1:
            maker = st.selectbox(':blue[CAR MAKER]', options=makers)

        with inp2:
            if maker:
                model_name = makers_models[maker]
            else:
                model_name = None
                
            model_name = st.selectbox(':blue[CAR MODEL]', options=model_name)
            model_name = str(model_name)

        with inp3:
            mini = 5
            maxi = 2856196
            mileage = st.text_input(':blue[CAR MILEAGE]', placeholder=f'{mini} - {maxi}')
            if mileage:
                mileage = np.int64(mileage)

        with inp4:
            st.write('**CLICK ON :red[PREDICT PRICE]**')
            st.button('PREDICT PRICE', type='primary', use_container_width=True, on_click=None)
            if maker and model_name and mileage:
                input_data = pd.DataFrame(data=[[mileage, maker, model_name]], columns=['Mileage', 'Make', 'Model'])
                prediction = pipe.predict(input_data)

    # Use col2 for output
    with col2:
        st.success('**OUTPUT**', icon=':material/output:')

        with st.container():
            if mileage:
                price = np.round(prediction[0], 2)
                st.info(f'##### **:material/currency_rupee: {price}**')
                

# container for like, share, video, repository, connection, hf space
with st.container(border=False):
    col1, col2 = st.columns([.85,1], border=False, vertical_alignment='top')

    with col1:
        st.button('**LIKE**', icon=':material/favorite:')

    with col2:
        st.button('**MADEE**', icon=':material/flight:', disabled=True)
