import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.dates as mdates

def get_date_range(start_date="2023-01-01",end_date="2023-01-30"):
    date_rage = pd.date_range(start=start_date,end=end_date,freq='D')
    date_dataframe = pd.DataFrame(date_rage,columns=['Date'])
    return date_dataframe
def make_model(dataframe):
    df = dataframe.copy()
    categorical_features = list(df.drop(['Date','Sales'],axis=1).columns)
    if len(categorical_features) > 0:
        df = df.drop(categorical_features,axis=1)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    numerical_features = ['Year', 'Month', 'Day', 'Weekday']
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
            ])
    X = df[numerical_features]

    xgb_regressor = XGBRegressor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb_regressor)])

    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test,predictions)
    return pipeline, mse

def agent_inference(pipeline,date_dataframe,to_plot=True):
    df = date_dataframe.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    X = df.copy()
    predictions = pipeline.predict(X)
    if to_plot:
        plt.style.use('dark_background')
        plt.plot(df['Date'], predictions, color='cyan')
        plt.xticks(rotation=45, ha='right')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.grid(color='gray', linestyle=':', linewidth=0.5)
        plt.tick_params(colors='white', which='both')
        plt.tight_layout()
        plt.show()
        plot = st.pyplot(plt,clear_figure=True)
    return predictions

def feature_store_pants(sku_id='all', size='all', pant_type='all', fabric='all', waist='all', front_pockets='all', back_pockets='all', closure='all', belt_loops='all', cuff='all', store='all', region='all'):
  kwargs = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and v != 'all'}
  import pandas as pd
  try:
      df = pd.read_csv("dataset.csv")
  except:
      df = pd.read_csv("https://datasetsdatascienceagent.blob.core.windows.net/salesdatasets/final_pants_dataset.csv")
      df.to_csv("dataset.csv")
  column_mapping = {
        'sku_id': 'SKU ID',
        'size': 'Size',
        'pant_type': "Pants Type",
        'fabric': "Fabric",
        'waist': "Waist",
        'front_pockets': "Front Pockets",
        'back_pockets': "Back Pockets",
        'closure': "Closure",
        'belt_loops': "Belt Loops",
        'cuff': "Cuff",
        'pattern': "Pattern",
        'store': "Store",
        'region': "Region",
    }


  filters = {}
  for column, value in kwargs.items():
      if value != 'all':
          filters[column_mapping[column]] = value
  print(filters)
  filtered_df = df
  for k,v in filters.items():
    filtered_df = filtered_df[filtered_df[k] == v]

  filtered_df = filtered_df.groupby(['Date','Store'])['Sales'].sum().reset_index()

  print(filtered_df.head())
  return filtered_df[['Date', 'Sales',"Store"]]

def fit_predict(sales_dataframe,date_range,is_store_level_breakdown = False):
    df = sales_dataframe.copy()
    stores = list(df['Store'].unique())
    metrics = {}
    plot_metrics = {}
    plot_metrics_absolute = {}
    if is_store_level_breakdown:
        for store in stores:
            df_store = df[df['Store'] == store]
            pipeline,mse = make_model(df_store)
            metrics["mse"] = mse
            prediction = agent_inference(pipeline,date_range,to_plot=False)
            metrics["predictions"] = prediction
            plot_metrics_absolute[store] = prediction
            avg_forcast = np.average(prediction)
            total_forcast  = np.sum(prediction)
            metrics['total_forecast'] = total_forcast
            metrics['average_forecast'] = avg_forcast
            plot_metrics[store] = avg_forcast
            print("Total forecast for " + store + ": "  + str(metrics['total_forecast']))
        store_names = list(plot_metrics.keys())
        forecasts = list(plot_metrics.values())
        extracted_date_range = pd.to_datetime(date_range['Date'])
        df_results = pd.DataFrame(plot_metrics_absolute, index=extracted_date_range)
        for column in store_names:
            plt.plot(df_results.index, df_results[column], marker='', linewidth=2, label=column)

        plt.legend(title='Store Name')
        plt.title('Store Predictions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Prediction')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt, clear_figure=True)
        plt.show()

        plt.bar(store_names, forecasts)
        st.pyplot(plt, clear_figure=True)
        plt.show()
    else:
        df = df.groupby(['Date'])['Sales'].sum().reset_index()
        df = df[['Date','Sales']]
        pipeline,mse = make_model(df)
        metrics['mse'] = mse
        prediction = agent_inference(pipeline,date_range,to_plot=True)
        metrics['predictions'] = prediction
        avg_forcast = np.average(prediction)
        total_forcast = np.sum(prediction)
        metrics['total_forecast'] = total_forcast
        metrics['average_forecast'] = avg_forcast
        print("Total forecast: " + str(metrics['total_forecast']))
    return metrics
