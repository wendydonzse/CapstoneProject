import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import joblib



st.set_page_config(layout='wide')

st.header('Forecaster')


data_file = st.file_uploader('Upload file containing timestamps',type='csv')

df = pd.DataFrame()

if data_file:
    df = pd.read_csv(data_file)


col1,col2 =  st.columns(2) 


# st.write('Your data: ',df.head(5))
col1.write("Your data:")
# col1.write(df.head(5))
col1.dataframe(df,height=250,width=600)





# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Feature Engineering
# Extract date and time components
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date
# Convert 'Timestamp' to datetime
#df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = pd.to_datetime(df['Date'])

## Removing outliers







# ML Model
#
#


model_selection = st.selectbox('Choose a model:',options=['Random Forest','Gradient Boosting Regressor'], index=None)
if model_selection == 'Random Forest':

    rf_model = joblib.load('rf_model.pkl')


    # data_cleaned = pd.read_csv('NO2_07112024_NoOutliers.csv')

    data = df[['Timestamp']]
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])



    # Check for missing values
    data = data.dropna()

    # Extract features and target variable
    X = data.index.astype(int).values.reshape(-1, 1)  # Use timestamp as a feature

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)




    y_pred = rf_model.predict(X_scaled)

    data['Value'] = y_pred


    st.write(data)