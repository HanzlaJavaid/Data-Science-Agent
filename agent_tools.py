import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
# from datetime import datetime, timedelta
import matplotlib.dates as mdates


# Comments: Old version
def fit_predict(sales_dataframe,date_dataframe):
    import pandas as pd
    metrics = {}
    feature_map = {}
    pandas_dataframe = sales_dataframe.copy()
    df = pandas_dataframe.copy()
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
    metrics['mse'] = mean_squared_error(y_test,predictions)


    results,plot = agent_inference(pipeline,date_dataframe)
    metrics['results'] = results
    metrics['average_forcast'] = np.average(results)
    metrics['plot'] = plot
    metrics['total_forecast'] = np.sum(results)
    print(metrics)
    return metrics

def agent_inference(pipeline,date_dataframe):
    df = date_dataframe.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    X = df.copy()
    predictions = pipeline.predict(X)
    plt.style.use('dark_background')
    plt.plot(df['Date'], predictions, color='cyan')  # Cyan stands out on a dark background
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.tick_params(colors='white', which='both')  # Change the colors of the tick marks to white
    plt.tight_layout()
    plt.show()
    plot = st.pyplot(plt,clear_figure=False)
    return predictions,plot

def feature_store_pants(sku_id='all', size='all', pant_type='all', fabric='all', waist='all', front_pockets='all', back_pockets='all', closure='all', belt_loops='all', cuff='all', store='all', region='all'):
  kwargs = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and v != 'all'}
  import pandas as pd
  df = pd.read_csv('datasets/final_pants_dataset.csv')
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

  filtered_df = filtered_df.groupby(['Date'])['Sales'].sum().reset_index()

  return filtered_df[['Date', 'Sales']]
