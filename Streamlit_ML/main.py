import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
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






if len(df)>0 and 'Value' in df.columns:

    # Convert 'Value' to numeric, forcing errors to NaN
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Value'] = df['Value'].fillna(df['Value'].mean())

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

    # Method 2: Z-Score
    df['Z-Score'] = zscore(df['Value'])
    outliers_z = df[np.abs(df['Z-Score']) > 3]
    # print("Outliers detected using Z-Score method:")
    # print(outliers_z)

    # Define a threshold for outlier detection (e.g., z-score > 3)
    z_threshold = 3

    # Calculate z-scores for each numerical column
    z_scores = stats.zscore(df.select_dtypes(include=['number']))

    # Identify outliers by z-score
    outliers = (abs(z_scores) > z_threshold).any(axis=1)

    # print("Outliers detected using z-score method:")
    # print(outliers)

    # Remove outliers from the DataFrame
    df_no_outliers = df[~outliers]

    df1 = df_no_outliers.copy()
    # df1.shape

    # Drop the 'Z-Score' column
    df_cleaned = df1.drop(columns=['Z-Score'])
    # print(df_cleaned.head())

    df_cleaned.to_csv(r'NO2_07112024_NoOutliers.csv', index=False)


    df_WithOutliers = df.drop(columns=['Z-Score'])


    df_WithOutliers.to_csv(r'NO2_07112024_WithOutliers.csv', index=False)

    data_cleaned = pd.read_csv('NO2_07112024_NoOutliers.csv')

    # data = pd.read_csv('NO2_07112024_WithOutliers.csv')
    # 



    # st.write('Preprocessed data: ',df_cleaned.head(5))
    col2.write("Preprocessed data:")
    # col2.write(df_cleaned.head(5))
    col2.dataframe(df_cleaned,height=250,width=600)




    with col1.container(height=500):

        st.markdown('')

        plt.figure(figsize=(10, 6))
        plt.boxplot(df['Value'], vert=False)
        plt.title('Box Plot of NO2 Values')
        plt.xlabel('NO2 Value')

        # Show the plot in the Streamlit app
        # st.pyplot(plt,use_container_width=True)
        st.pyplot(plt,use_container_width=True)



    with col2.container(height=500):

        st.markdown('')
        # Convert 'Timestamp' to datetime
        data_cleaned['Timestamp'] = pd.to_datetime(data_cleaned['Timestamp'])
        data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'])

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=data_cleaned, x='Timestamp', y='Value')
        plt.title('NO2 Levels Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('NO2 Value')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot in the Streamlit app
        st.pyplot(plt)

    with col1.container(height=500):

        st.markdown('')
        # Distribution Plot: Check the distribution of NO2 values
        # Histogram of NO2 values
        plt.figure(figsize=(12, 5))
        sns.histplot(data_cleaned['Value'], bins=30, kde=True)
        plt.title('Distribution of NO2 Values')
        plt.xlabel('NO2 Value')
        plt.ylabel('Frequency')
        plt.show()


        st.pyplot(plt)

    with col2.container(height=500):

        st.markdown('')
        # Feature Enginering : Extract Weekday Information: Add a column to indicate whether the day is a weekday or weekend.
        # We want to see the No2 values over the Weekdays vs Weekends patterns 

        # Add a column for the day of the week
        data_cleaned['DayOfWeek'] = data_cleaned['Timestamp'].dt.dayofweek

        # Add a column to indicate if it's a weekend (Saturday=5, Sunday=6)
        data_cleaned['IsWeekend'] = data_cleaned['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        # Display the transformed DataFrame
        print(data_cleaned.head())


        sns.set(style="whitegrid")

        # Plot the data
        plt.figure(figsize=(16, 4))
        sns.lineplot(data=data_cleaned, x='Timestamp', y='Value', hue='IsWeekend', palette='coolwarm', ci=None)
        plt.title('NO2 Levels: Weekdays vs. Weekends')
        plt.xlabel('Timestamp')
        plt.ylabel('NO2 Value')
        plt.xticks(rotation=45)
        plt.legend(title='Is Weekend', labels=['Weekday', 'Weekend'])
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)



    with col1.container(height=500):

        st.markdown('')




        # Set the style
        sns.set(style="whitegrid")

        # Plot the data
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=data_cleaned, x='IsWeekend', y='Value', palette='coolwarm')
        plt.title('NO2 Levels: Weekdays vs. Weekends')
        plt.xlabel('IsWeekend')
        plt.ylabel('NO2 Value')
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)

    mean_values = data_cleaned.groupby('IsWeekend')['Value'].mean()
    # Define Peak and Off-Peak Hours
    # Assume peak hours are from 7 AM to 9 AM and 4 PM to 6 PM, and off-peak hours are the rest.
    # Define peak and off-peak hours
    def categorize_time(hour):
        if 7 <= hour <= 9 or 16 <= hour <= 18:
            return 'Peak'
        else:
            return 'Off-Peak'

    # Apply the function to create a new column
    data_cleaned['Period'] = data_cleaned['Timestamp'].dt.hour.apply(categorize_time)
    # Display the transformed DataFrame
    # print(data_cleaned.head())
    # Display data types
    # print(data_cleaned.dtypes)

    with col2.container(height=500):

        st.markdown('')

        sns.set(style="whitegrid")

        # Plot the data
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=data_cleaned, x='Period', y='Value', palette='coolwarm')
        plt.title('NO2 Levels: Peak vs. Off-Peak')
        plt.xlabel('Period')
        plt.ylabel('NO2 Value')
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)



    # Calculate Average NO2 Levels for Peak and Off-Peak:

    mean_values = data_cleaned.groupby('Period')['Value'].mean()

    # Define Seasons : Assume winter months are June, July, and August, and summer months are December, January, and February.
    # Define a function to categorize seasons
    def categorize_season(month):
        if month in [6, 7, 8]:
            return 'Winter'
        elif month in [12, 1, 2]:
            return 'Summer'
        else:
            return 'Other'

    # Apply the function to create a new column
    data_cleaned['Season'] = data_cleaned['Timestamp'].dt.month.apply(categorize_season)
    # Display the transformed DataFrame
    print(data_cleaned.head())
    # Display data types
    print(data_cleaned.dtypes)



    with col1.container(height=500):

        st.markdown('')
        sns.set(style="whitegrid")

        # Filter data for Winter and Summer
        seasonal_data = data_cleaned[data_cleaned['Season'].isin(['Winter', 'Summer'])]

        # Plot the data
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=seasonal_data, x='Season', y='Value', palette='coolwarm')
        plt.title('NO2 Levels: Winter vs. Summer')
        plt.xlabel('Season')
        plt.ylabel('NO2 Value')
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)



    mean_values = data_cleaned.groupby('Season')['Value'].mean()
    data_selected = data_cleaned[['Timestamp', 'Value']]


    with col2.container(height=500):

        st.markdown('')

        data_selected.boxplot()

        # Set title and labels
        plt.title('Boxplot of NO2 Variables')
        plt.xlabel('NO2')
        plt.ylabel('Values')

        # Show the plot
        plt.show()


        st.pyplot(plt)


    # Convert 'Timestamp' to datetime
    data_selected['Timestamp'] = pd.to_datetime(data_selected['Timestamp'])

    # Format the 'Timestamp' to 'Date' format
    data_selected['Date'] = data_selected['Timestamp'].dt.strftime('%-d/%m/%Y %H:%M')
    # Extract date and time components
    data_selected['Hour'] = data_selected['Timestamp'].dt.hour
    data_selected['Date'] = data_selected['Timestamp'].dt.date 
    data_selected['DateHour'] = data_selected['Timestamp'].dt.strftime('%d/%m/%Y %H:%M') 
    data_arima = data_selected[['DateHour', 'Value']]
    data_arima.to_csv(r'No2_data_arima.csv', index=False)

    data_arima = pd.read_csv('No2_data_arima.csv')
    # Convert the "DateHour" column to datetime format
    data_arima['DateHour'] = pd.to_datetime(data_arima['DateHour'], dayfirst = True)

    # Sort the DataFrame by the "Date" column to ensure that the timestamps are in ascending order.
    data_arima = data_arima.sort_values(by='DateHour')

    # Calculate the time differences between consecutive timestamps
    time_diffs = data_arima['DateHour'].diff()




    with col1.container(height=500):

        st.markdown('')

        # Plot ACF
        plot_acf(data_arima['Value'], lags=48)  # Change lags value according to your data
        plt.title('ACF Plot')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()

        st.pyplot(plt)



    with col2.container(height=500):

        st.markdown('')

        # Plot PACF
        plot_pacf(data_arima['Value'], lags=48)  # Change lags value according to your data
        plt.title('PACF Plot')
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')
        plt.show()



        st.pyplot(plt)





    with col1.container(height=500):

        st.markdown('')

        # Convert index to DatetimeIndex if not already done
        data_arima.index = pd.to_datetime(data_arima.index)

        # Set the start date explicitly if needed
        start_date = '2006-01-01 00:00'
        data_arima.index = pd.date_range(start=start_date, periods=len(data_arima), freq='H')  # Assuming hourly frequency

        # Perform seasonal decomposition
        result = seasonal_decompose(data_arima['Value'], model='additive', period=24)  # Set period to 24 hours in a day 

        # Plot the decomposition
        plt.figure(figsize=(12, 8))

        plt.subplot(411)
        plt.plot(data_arima.index, data_arima['Value'], label='Original')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(412)
        plt.plot(data_arima.index, result.trend, label='Trend')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(413)
        plt.plot(data_arima.index, result.seasonal, label='Seasonal')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(414)
        plt.plot(data_arima.index, result.resid, label='Residual')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.tight_layout()
        plt.show()

        st.pyplot(plt)


    with col2.container(height=500):

        st.markdown('')


        # Convert index to DatetimeIndex if not already done
        data_arima.index = pd.to_datetime(data_arima.index)

        # Set the start date explicitly if needed
        start_date = '2024-11-02 16:00:00'
        data_arima.index = pd.date_range(start=start_date, periods=len(data_arima), freq='H')  # Assuming hourly frequency

        # Perform seasonal decomposition
        result = seasonal_decompose(data_arima['Value'], model='additive', period=24)  # Set period to 24 hours in a day 

        # Plot the decomposition
        plt.figure(figsize=(12, 8))

        plt.subplot(411)
        plt.plot(data_arima.index, data_arima['Value'], label='Original')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(412)
        plt.plot(data_arima.index, result.trend, label='Trend')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(413)
        plt.plot(data_arima.index, result.seasonal, label='Seasonal')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(414)
        plt.plot(data_arima.index, result.resid, label='Residual')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.tight_layout()
        plt.show()

        st.pyplot(plt)

    # Load the data
    data_arima = pd.read_csv('No2_data_arima.csv', parse_dates=['DateHour'], index_col='DateHour', dayfirst=True)
    # Convert index to DatetimeIndex if not already done
    data_arima.index = pd.to_datetime(data_arima.index)


    with col1.container(height=500):

        st.markdown('')

        # Perform seasonal decomposition
        result = seasonal_decompose(data_arima['Value'], model='additive', period=24)  # Assuming hourly data with daily seasonality

        # Plot the decomposition
        plt.figure(figsize=(12, 8))

        plt.subplot(411)
        plt.plot(data_arima.index, data_arima['Value'], label='Original')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(412)
        plt.plot(data_arima.index, result.trend, label='Trend')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(413)
        plt.plot(data_arima.index, result.seasonal, label='Seasonal')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.subplot(414)
        plt.plot(data_arima.index, result.resid, label='Residual')
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

        plt.tight_layout()
        plt.show()


        st.pyplot(plt)





if len(df)>0:

    # ML Model
    #
    #


    model_selection = st.selectbox('Choose a model for forecasting:',options=['Random Forest','Gradient Boost Regressor'], index=None)


    if model_selection :

        data = df[['Timestamp']]
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])

        # Check for missing values
        data = data.dropna()
            
        # Extract features and target variable
        X = data.index.astype(int).values.reshape(-1, 1)  # Use timestamp as a feature

        if model_selection == 'Random Forest':

            model = joblib.load('rf_model.pkl')

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = X_scaled


        if model_selection == 'Gradient Boost Regressor':

            model = joblib.load('gb_model.pkl')



        y_pred = model.predict(X)

        data['Value'] = y_pred
        st.write(data)







