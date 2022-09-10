import os
os.system('pip install streamlit xgboost pandas joblib')
import streamlit as st
import pandas as pd
import webbrowser

# some bullons and texts
st.title("Wildfire Prediction App")
st.text("This app predicts the likelihood of a wildfire in a given area")

dataset_url1 = 'https://www.sciencedirect.com/science/article/abs/pii/S0379711218303941'
notebook_url = 'https://github.com/tikendraw/Wildfire_prediction/blob/main/Wildfire.ipynb'





col1, col2, col3 = st.columns([1,1,1])

with col1:
    if st.button('Dataset'):
        webbrowser.open_new_tab(dataset_url1)
with col2:
    if st.button('Code/Notebook'):
        webbrowser.open_new_tab(notebook_url)
with col3:
    st.button('Reload')


# parameters
ndvi = st.sidebar.number_input("Enter the NDVI Index Value:", value=0.2, step=0.01, min_value = 0.0, max_value = 0.8)

lst = st.sidebar.number_input( "Enter the LST: Land Surface Temperature (Kelvin): ", value=14662.50, step=0.01, min_value = 0.00, max_value = 20000.00)

burned_area = st.sidebar.number_input( "Enter the Burned Area : ", value=2.0, step=0.01, min_value = 0.00, max_value = 10.0)


# parameter info
ndvi_info = '''**Normalized Difference Vegetation Index**
The most common measurement is called the Normalized Difference Vegetation Index (NDVI).
Very low values of NDVI (0.1 and below) correspond to barren areas of rock, sand, or snow. 
Moderate values represent shrub and grassland (0.2 to 0.3), 
while high values indicate temperate and tropical rainforests (0.6 to 0.8).'''

st.write(ndvi_info)

lst_info = '''**Land surface temperature (LST)** is the radiative skin temperature of the land derived from solar radiation.
LST measures the emission of thermal radiance from the land surface where the incoming solar energy interacts
with and heats the ground, or the surface of the canvaluesopy in vegetated areas. LST is a mixture of vegetation and 
bare soil temperatures. This quality makes LST a good indicator of energy partitioning at the land 
surface–atmosphere boundary and sensitive to changing surface conditions, common range 13200-15600'''

st.write(lst_info)

burned_area_info = '''**Burned Area** is the area of the forest that has been burned in Hectares.'''

st.write(burned_area_info)

# printing Inputs
st.write(ndvi,lst, burned_area)


# Prediction
modelname = st.sidebar.selectbox("Prediction Model:", ['Random Forest', 'XG Boost'])

predict_button = st.sidebar.button("Predict")

import joblib
if modelname == 'Random Forest':
    model = joblib.load('./RandomForestClassifier.joblib')
else:
    model = joblib.load('./XGBClassifier.joblib')

input_data = pd.DataFrame([[ndvi,lst,burned_area]])
if predict_button:

    result = model.predict(input_data)
    result_proba = model.predict_proba(input_data)

    confidance = float(result_proba[0][result])
    st.write('0: Fire, 1: No Fire')
    st.markdown(result)
    st.write(result_proba)
    st.markdown(f'Confidance: {round(confidance,2)} %')
    if result == 1:
        good = "The area is not likely to have a wildfire"
        st.markdown(f'<p style="color:#2AC153;font-size:24px;border-radius:2%;"><b>{good}</b></p>', unsafe_allow_html=True)
        
    elif result == 0:
        bad = "Alert!!! The area is likely to have a wildfire 🔥"
        st.sidebar.header('🔥')
        st.markdown(f'<p style="color:#E44D32;font-size:24px;border-radius:2%;"><b>{bad}</b></p>', unsafe_allow_html=True)




dataset_info = '''The study area is composed of multiple zones located in the center of Canada. 
The surface of this area is approximately 2 million hectares. These zones dirand_tuneder in their size,
burn period, date of burn and extent. We have chosen to apply the experiment in a big region of Canada's 
forests because it is known for its high rate of wildfires and also for the availability of fire
information (start and end fire date, cause of fire and the surface of the burned area in hectares),
these information were acquired from The Canadian Wild-land Fire Information System (CWFIS)which creates 
daily fire weather and fire behavior maps year-round and hot spot maps throughout the forest fire season'''

st.subheader('About the Dataset')
st.text(dataset_info)

data_repo = 'https://github.com/ouladsayadyounes/WildFires'
st.write(f'Download the Dataset: {data_repo}')
