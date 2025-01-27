import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense

import joblib
from pmdarima import auto_arima

import gc




st.set_page_config(layout='wide')

st.header('NO2 level Analysing & Forecasting')


data_file = st.file_uploader('Upload file containing Timestamp and Value',type='csv')

df = pd.DataFrame()

if data_file:
    df = pd.read_csv(data_file)


col1,col2 =  st.columns(2) 


# st.write('Your data: ',df.head(5))
col1.write("Your data:")
# col1.write(df.head(5))
col1.dataframe(df,height=250,use_container_width=True)






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

    del df_no_outliers # As it is no longer required
    gc.collect()

    # df1.shape

    # Drop the 'Z-Score' column
    df_cleaned = df1.drop(columns=['Z-Score'])

    del df1 # As it is no longer required
    gc.collect()
    # print(df_cleaned.head())

    df_cleaned.to_csv(r'NO2_07112024_NoOutliers.csv', index=False)


    df_WithOutliers = df.drop(columns=['Z-Score'])


    df_WithOutliers.to_csv(r'NO2_07112024_WithOutliers.csv', index=False)
    

    del df_WithOutliers # As it is no longer required
    gc.collect()

    data_cleaned = pd.read_csv('NO2_07112024_NoOutliers.csv')

    # data = pd.read_csv('NO2_07112024_WithOutliers.csv')
    # 



    # st.write('Preprocessed data: ',df_cleaned.head(5))


    # with col2.container():
    col2.write("Preprocessed data:")
    # col2.write(df_cleaned.head(5))
    col2.dataframe(df_cleaned,height=250,use_container_width=True)

    del df_cleaned # As it is no longer required
    gc.collect()




    with col1.container(height=600):

        # st.markdown('')
        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('Box Plot of NO2 Values')

        plt.figure(figsize=(10, 6))
        plt.boxplot(df['Value'], vert=False)
        plt.title('Box Plot of NO2 Values')
        plt.xlabel('NO2 Value')

        # Show the plot in the Streamlit app
        # st.pyplot(plt,use_container_width=True)
        st.pyplot(plt,use_container_width=True)

    # As df is no longer required
    del df
    gc.collect()



    with col2.container(height=600):

        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('NO2 Levels Over Time')
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

    with col1.container(height=600):

        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('Distribution of NO2 Values')
        # Distribution Plot: Check the distribution of NO2 values
        # Histogram of NO2 values
        plt.figure(figsize=(12, 5))
        sns.histplot(data_cleaned['Value'], bins=30, kde=True)
        plt.title('Distribution of NO2 Values')
        plt.xlabel('NO2 Value')
        plt.ylabel('Frequency')
        plt.show()


        st.pyplot(plt)

    with col2.container(height=600):
        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('NO2 Levels: Weekdays vs. Weekends')
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



    with col1.container(height=600):

        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('NO2 Levels: Weekdays vs. Weekends')


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

    with col2.container(height=600):

        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('NO2 Levels: Peak vs. Off-Peak')

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



    with col1.container(height=600):

        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('NO2 Levels: Winter vs. Summer')
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


    with col2.container(height=600):

        c1,c2,c3 = st.columns([2,3,1])
        c2.subheader('Boxplot of NO2 Variables')
        plt.figure(figsize=(14, 6))

        # Create the boxplot
        data_selected.boxplot(figsize=(14, 6))

        # Set title and labels
        plt.title('Boxplot of NO2 Variables')
        plt.xlabel('NO2')
        plt.ylabel('Values')

        # Render the plot in Streamlit
        st.pyplot(plt)


    # Convert 'Timestamp' to datetime
    data_selected['Timestamp'] = pd.to_datetime(data_selected['Timestamp'])

    # Format the 'Timestamp' to 'Date' format
    data_selected['Date'] = data_selected['Timestamp'].dt.strftime('%-d/%m/%Y %H:%M')
    # Extract date and time components
    data_selected['Hour'] = data_selected['Timestamp'].dt.hour
    data_selected['Date'] = data_selected['Timestamp'].dt.date 
    data_selected['DateHour'] = data_selected['Timestamp'].dt.strftime('%d/%m/%Y %H:%M') 
    # data_arima = data_selected[['DateHour', 'Value']]


    # del data_selected # As it is no longer required
    # gc.collect()

    # data_arima.to_csv(r'No2_data_arima.csv', index=False)

    # data_arima = pd.read_csv('No2_data_arima.csv')
    # # Convert the "DateHour" column to datetime format
    # data_arima['DateHour'] = pd.to_datetime(data_arima['DateHour'], dayfirst = True)

    # # Sort the DataFrame by the "Date" column to ensure that the timestamps are in ascending order.
    # data_arima = data_arima.sort_values(by='DateHour')

    # # Calculate the time differences between consecutive timestamps
    # time_diffs = data_arima['DateHour'].diff()




    # with col1.container(height=600):

    #     c1,c2,c3 = st.columns([2,3,1])
    #     c2.subheader('ACF Plot')
    #     # Plot ACF
    #     plot_acf(data_arima['Value'], lags=48)  # Change lags value according to your data
    #     plt.title('ACF Plot')
    #     plt.xlabel('Lag')
    #     plt.ylabel('Autocorrelation')
    #     plt.show()

    #     st.pyplot(plt)



    # with col2.container(height=600):

    #     c1,c2,c3 = st.columns([2,3,1])
    #     c2.subheader('PACF Plot')
    #     # Plot PACF
    #     plot_pacf(data_arima['Value'], lags=48)  # Change lags value according to your data
    #     plt.title('PACF Plot')
    #     plt.xlabel('Lag')
    #     plt.ylabel('Partial Autocorrelation')
    #     plt.show()



    #     st.pyplot(plt)







    # # filtered_data = data_arima[(data_arima['Timestamp'].dt.year == 2024) & (data_arima['Timestamp'].dt.month.isin([10, 11]))]
    # # filtered_data.to_csv(r'No2_filtered_data.csv', index=False)
    # # Load the data
    # filtered_data = pd.read_csv('No2_data_arima.csv', parse_dates=['DateHour'], index_col='DateHour', dayfirst=True)
    # # Convert index to DatetimeIndex if not already done
    # filtered_data.index = pd.to_datetime(filtered_data.index)


    # result = seasonal_decompose(filtered_data['Value'], model='additive', period=24)  # Assuming hourly data withValue daily seasonality

    # # Plot the decomposition
    # plt.figure(figsize=(12, 8))

    # plt.subplot(411)
    # plt.plot(filtered_data.index, data_arima['Value'], label='Original')
    # plt.legend(loc='upper left')
    # plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

    # plt.subplot(412)
    # plt.plot(filtered_data.index, result.trend, label='Trend')
    # plt.legend(loc='upper left')
    # plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

    # plt.subplot(413)
    # plt.plot(filtered_data.index, result.seasonal, label='Seasonal')
    # plt.legend(loc='upper left')
    # plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

    # plt.subplot(414)
    # plt.plot(filtered_data.index, result.resid, label='Residual')
    # plt.legend(loc='upper left')
    # plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

    # plt.tight_layout()
    # plt.show()

    # with col1.container(height=600):

    #     st.pyplot(plt)


    st.markdown('') # For adding vertical line space
    st.markdown('')
    st.markdown('') 


    st.subheader('Training results of ML Models: ', help='Models to be trained are: 1) Random Forest Regression (RFR) 2) Gradient Boosting Regression (GBR) 3) Support Vector Regression (SVR) 4) Artificial Neural Networks (ANN)')
    



    # ML Model

    # model_selection = st.selectbox('Choose a model for forecasting (Random Forest seems to provide better results):',options=['Random Forest','Gradient Boost Regressor','SVR','ANN'], index=None)

    st.session_state.models_run = False
    st.session_state.arima_run = False



    @st.fragment
    def train_models():
        train_model_button = st.button('Start training ML Models')
        if train_model_button:
            with st.container(border=2):

                model_scores = {}


                data_cleaned = pd.read_csv('NO2_07112024_NoOutliers.csv')
                data = data_cleaned[['Timestamp', 'Value']]

                data['Timestamp'] = pd.to_datetime(data['Timestamp'])

                # Check for missing values
                data = data.dropna()
                    
                # Extract features and target variable
                X = data.index.astype(int).values.reshape(-1, 1)  # Use timestamp as a feature
                y = data['Value'].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




                # if model_selection == 'Random Forest':
                try:


                    # Scale the features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X = X_scaled

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


                    # Initialize the Random Forest Regressor
                    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

                    # Train the model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Evaluate the model
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    model_scores['Random Forest Regression (RFR)'] = {'mse':mse,'r2':r2}

                    print(f"Mean Squared Error: {mse}")
                    print(f"R^2 Score: {r2}")


                    plt.figure(figsize=(10, 5))
                    plt.scatter(X_test, y_test, color='blue', label='Actual')
                    plt.scatter(X_test, y_pred, color='red', label='Predicted')
                    plt.xlabel('Timestamp')
                    plt.ylabel('NO2 Value')
                    plt.title('Random Forest Regression on NO2 Data')
                    plt.legend()
                    plt.show()

                    st.pyplot(plt)




                    # model = joblib.load('rf_model.pkl')

                except Exception as e:
                    st.write('Some error occured while training RandomForest',e)



                # if model_selection == 'Gradient Boost Regressor':
                try:
                    # model = joblib.load('gb_model.pkl')

                    # Initialize the Gradient Boosting Regressor
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

                    # Train the model
                    model.fit(X_train, y_train)



                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Evaluate the model
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_scores['Gradient Boosting Regression (GBR)'] = {'mse':mse,'r2':r2}


                    print(f"Mean Squared Error: {mse}")
                    print(f"R^2 Score: {r2}")

                    # Plot the results

                    plt.figure(figsize=(10, 5))
                    plt.scatter(X_test, y_test, color='blue', label='Actual')
                    plt.scatter(X_test, y_pred, color='red', label='Predicted')
                    plt.xlabel('Timestamp')
                    plt.ylabel('NO2 Value')
                    plt.title('Gradient Boosting Regression on NO2 Data')
                    plt.legend()
                    plt.show()

                    st.pyplot(plt)
                except Exception as e:
                    st.write('Some error occured while training Gradient Boosting Regressor',e)


                # if model_selection == 'SVR':
                try:


                    # Initialize and train the SVR model
                    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                    model.fit(X_train, y_train)


                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Evaluate the model
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_scores['Support Vector Regression (SVR)'] = {'mse':mse,'r2':r2}

                    print(f"Mean Squared Error: {mse}")
                    print(f"R^2 Score: {r2}")


                    plt.figure(figsize=(10, 5))
                    plt.scatter(X_test, y_test, color='blue', label='Actual')
                    plt.scatter(X_test, y_pred, color='red', label='Predicted')
                    plt.xlabel('Timestamp')
                    plt.ylabel('NO2 Value')
                    plt.title('Support Vector Regression on NO2 Data')
                    plt.legend()
                    plt.show()

                    st.pyplot(plt)
                    
                except Exception as e:
                    st.write('Some error occured while training Support Vector Regressor',e)

                # if model_selection=='ANN':
                try:



                    # Initialize the ANN
                    model = Sequential()

                    # Add input layer and first hidden layer
                    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

                    # Add second hidden layer
                    model.add(Dense(units=32, activation='relu'))

                    # Add output layer
                    model.add(Dense(units=1))

                    # Compile the ANN
                    model.compile(optimizer='adam', loss='mean_squared_error')


                    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

                    # Evaluate the model on the test set
                    y_pred = model.predict(X_test)

                    # Calculate performance metrics

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    model_scores['Artificial Neural Networks (ANN)'] = {'mse':mse,'r2':r2}

                    print(f"Mean Squared Error: {mse}")
                    print(f"R^2 Score: {r2}")

                    # Plot the training and validation loss

                    plt.figure(figsize=(10, 5))
                    plt.plot(history.history['loss'], label='Training Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')
                    plt.legend()
                    plt.show()

                    # Plot the results
                    plt.figure(figsize=(10, 5))
                    plt.scatter(X_test, y_test, color='blue', label='Actual')
                    plt.scatter(X_test, y_pred, color='red', label='Predicted')
                    plt.xlabel('Timestamp')
                    plt.ylabel('NO2 Value')
                    plt.title('ANN Regression on NO2 Data')
                    plt.legend()
                    plt.show()

                    st.pyplot(plt)


                except Exception as e:
                    st.write('Some error occured while training ANN',e)


                st.subheader('Current Training Results: ')
                # st.write(model_scores)

                # Convert dictionary to a DataFrame for tabular display
                results_df = pd.DataFrame.from_dict(model_scores, orient='index').reset_index()
                results_df.columns = ['Model', 'Mean Squared Error (MSE)', 'R² Score']

                # Display the table in Streamlit
                # st.title("Model Results Table")
                # st.dataframe(results_df, use_container_width=True)
                # st.table(results_df)
                # Adjust DataFrame to remove index during display
                results_df = results_df.set_index('Model')

                # Display in Streamlit
                st.dataframe(results_df, use_container_width=True)


            st.markdown('')
            with st.container(border=2):
                st.subheader('On Average Training Results looks like: ')
                data = {
                    "Models": [
                        "Support Vector Regression (SVR)",
                        "Gradient Boosting Regression (GBR)",
                        "Random Forest Regression (RFR)",
                        "Artificial Neural Networks (ANN)",
                    ],
                    "Results": [
                        "Mean Squared Error: 87.58514344123985\nR² Score: -0.011607182429077727",
                        "Mean Squared Error: 0.09722533529960477\nR² Score: 0.16039535114709946",
                        "Mean Squared Error: 0.1689098881399603\nR² Score: 0.797047990230989",
                        "Mean Squared Error: 0.10782718665217803\nR² Score: 0.06884139913821286",
                    ],
                    "Analysis": [
                        "The SVR model has the highest MSE and a negative R² score, indicating poor performance and that the model does not fit the data well. Perhaps sensitive to scale,noise,outliers. Need further feature engineer. ",
            		"GBR performed better than SVR, with a significantly lower MSE. However, the R² score is still low, indicating that the model explains only a small portion of the variance in the data.",
                        "RFR performed well, with a low MSE and a highest R² score. It explains a significant portion of the variance in the data, making it one of the best-performing models in this analysis. The model can deal with nonlinear relationship, is robust to overfitting, and flexible enough to capture feature interactions, less impact by outliers and noise",
                         "ANN has a low MSE, but its R² score is very low, indicating that it does not explain much of the variance in the data. This suggests that the model may not be well-suited for this problem, perhaps sensitive to noise/outliers or requires more complex tuning",

                    ],
                }

                # Create a DataFrame
                df = pd.DataFrame(data)

                # Streamlit layout
                # st.title("Machine Learning Model Performance")

                # Display the table
                st.table(df)

            st.session_state.models_run = True

            show_conclusion()



    train_models()






    st.markdown('') # For adding vertical line space
    st.markdown('')
    st.markdown('') 


    st.subheader('Forecast results by arima: ')
    
    @st.fragment
    def arima_forecast():


        short_data = st.file_uploader('Upload file containing less number of Timestamp and Value',type='csv', key='arima')
        short_df = pd.DataFrame()

        start_arima = st.button('Start Arima Forecasting')

        if short_data and start_arima:
            short_df = pd.read_csv(short_data)


            # Convert 'Value' to numeric, forcing errors to NaN
            short_df['Value'] = pd.to_numeric(short_df['Value'], errors='coerce')
            # Convert 'Timestamp' to datetime
            short_df['Timestamp'] = pd.to_datetime(short_df['Timestamp'])
            short_df['Value'] = short_df['Value'].fillna(short_df['Value'].mean())

            # Feature Engineering
            # Extract date and time components
            short_df['Year'] = short_df['Timestamp'].dt.year
            short_df['Month'] = short_df['Timestamp'].dt.month
            short_df['Day'] = short_df['Timestamp'].dt.day
            short_df['Hour'] = short_df['Timestamp'].dt.hour
            short_df['Date'] = short_df['Timestamp'].dt.date
            # Convert 'Timestamp' to datetime
            #df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            short_df['Date'] = pd.to_datetime(short_df['Date'])

            ## Removing outliers

            # Method 2: Z-Score
            short_df['Z-Score'] = zscore(short_df['Value'])
            outliers_z = short_df[np.abs(short_df['Z-Score']) > 3]

            # Define a threshold for outlier detection (e.g., z-score > 3)
            z_threshold = 3

            # Calculate z-scores for each numerical column
            z_scores = stats.zscore(short_df.select_dtypes(include=['number']))

            # Identify outliers by z-score
            outliers = (abs(z_scores) > z_threshold).any(axis=1)

            # Remove outliers from the DataFrame
            short_df_no_outliers = short_df[~outliers]


            # Drop the 'Z-Score' column
            short_df_cleaned = short_df_no_outliers.drop(columns=['Z-Score'])

            short_df_cleaned.to_csv('short_No2_data_arima.csv')

            short_data_selected = short_df_cleaned[['Timestamp', 'Value']]
            # Convert 'Timestamp' to datetime
            short_data_selected['Timestamp'] = pd.to_datetime(short_data_selected['Timestamp'])

            # Format the 'Timestamp' to 'Date' format
            short_data_selected['Date'] = short_data_selected['Timestamp'].dt.strftime('%-d/%m/%Y %H:%M')
            # Extract date and time components
            short_data_selected['Hour'] = short_data_selected['Timestamp'].dt.hour
            short_data_selected['Date'] = short_data_selected['Timestamp'].dt.date 
            short_data_selected['DateHour'] = short_data_selected['Timestamp'].dt.strftime('%d/%m/%Y %H:%M') 
            short_data_arima = short_data_selected[['DateHour', 'Value']]


            short_data_arima.to_csv(r'short_No2_data_arima.csv', index=False)

            # Convert the "DateHour" column to datetime format
            short_data_arima['DateHour'] = pd.to_datetime(short_data_arima['DateHour'], dayfirst = True)

            # Sort the DataFrame by the "Date" column to ensure that the timestamps are in ascending order.
            short_data_arima = short_data_arima.sort_values(by='DateHour')

            # Calculate the time differences between consecutive timestamps
            # time_diffs = data_arima['DateHour'].diff()


            col1,col2 =  st.columns(2) 

            with col1.container(height=600):

                c1,c2,c3 = st.columns([2,3,1])
                c2.subheader('ACF Plot')
                # Plot ACF
                plot_acf(short_data_arima['Value'], lags=48)  # Change lags value according to your data
                plt.title('ACF Plot')
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.show()

                st.pyplot(plt)



            with col2.container(height=600):

                c1,c2,c3 = st.columns([2,3,1])
                c2.subheader('PACF Plot')
                # Plot PACF
                plot_pacf(short_data_arima['Value'], lags=48)  # Change lags value according to your data
                plt.title('PACF Plot')
                plt.xlabel('Lag')
                plt.ylabel('Partial Autocorrelation')
                plt.show()



                st.pyplot(plt)
    



            short_filtered_data = pd.read_csv('short_No2_data_arima.csv', parse_dates=['DateHour'], index_col='DateHour', dayfirst=True)
            # Convert index to DatetimeIndex if not already done
            short_filtered_data.index = pd.to_datetime(short_filtered_data.index)


            short_result = seasonal_decompose(short_filtered_data['Value'], model='additive', period=24)  # Assuming hourly data withValue daily seasonality
            # Plot the decomposition
            plt.figure(figsize=(12, 8))

            plt.subplot(411)
            plt.plot(short_filtered_data.index, short_data_arima['Value'], label='Original')
            plt.legend(loc='upper left')
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

            plt.subplot(412)
            plt.plot(short_filtered_data.index, short_result.trend, label='Trend')
            plt.legend(loc='upper left')
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

            plt.subplot(413)
            plt.plot(short_filtered_data.index, short_result.seasonal, label='Seasonal')
            plt.legend(loc='upper left')
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

            plt.subplot(414)
            plt.plot(short_filtered_data.index, short_result.resid, label='Residual')
            plt.legend(loc='upper left')
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format the x-axis as "YYYY-MM-DD"

            plt.tight_layout()
            plt.show()



            with col1.container(height=600):

                st.pyplot(plt)


            # model = auto_arima(
            #     short_df['Value'], 
            #     seasonal=False, 
            #     stepwise=True, 
            #     trace=True, 
            #     suppress_warnings=True
            # )
            # Fit ARIMA model
            # model = auto_arima(filtered_data['Value'], seasonal=True, m=24, trace=True, error_action='ignore', suppress_warnings=True)
            model = auto_arima(short_filtered_data['Value'], seasonal=True, m=24, trace=True, error_action='ignore', suppress_warnings=True)

            
            # model = auto_arima(
            #     filtered_data['Value'], 
            #     seasonal=False, 
            #     stepwise=True, 
            #     trace=True, 
            #     suppress_warnings=True
            # )

            


            print(model.summary())

            # Forecast the next 24 hours
            forecast = model.predict(n_periods=24)
            # print(forecast)

            # Plot the forecast
            plt.figure(figsize=(10, 5))
            plt.plot(short_filtered_data.index, short_filtered_data['Value'], label='Historical')
            # plt.plot(pd.date_range(data_arima.index[-1], periods=25, freq='H')[1:], forecast, label='Forecast', color='red')
            plt.plot(pd.date_range(short_filtered_data.index[-1], periods=25, freq='H')[1:], forecast, label='Forecast', color='red')

            
            plt.legend()
            plt.show()



            with col2.container(height=600):
                c1,c2,c3 = st.columns([2,6,1])
                c2.subheader('Graph: Forecast results by arima model ')
                st.pyplot(plt)

            with col1.container(height=600):
                c1,c2,c3 = st.columns([2,5,1])
                c2.subheader('Forecast results by arima model')
                st.dataframe(forecast)

                # st.write(data_arima.index[-1],forecast)
                # st.write(filtered_data,forecast)
                # st.write(filtered_data,forecast)
                # st.write(filtered_data.index[-1])



            # del data_arima # As it is no longer required
            # gc.collect()

            st.session_state.arima_run = True
            show_conclusion()
    arima_forecast()


        
    def show_conclusion():
        if st.session_state.models_run == True and st.session_state.arima_run == True:      
            with st.container(border=2):    
                st.markdown("""
                    ### **Conclusion**
                    - The **Random Forest Regression** model outperforms the other models in terms of both MSE and R² score, making it the most effective model for this dataset.
                        
                    - **ARIMA models** are ideal for simpler, univariate time series with clear trends and seasonality, where interpretability is important.
                    - **Machine learning models** are better suited for complex, high-dimensional datasets with non-linear relationships, external factors, and long-term forecasting needs.

                    For Furture development we should look at the Hybrid Model to Combine ARIMA with RFR could leverage the strengths of both statistical and machine learning approaches, improving prediction accuracy and robustness.
                """)