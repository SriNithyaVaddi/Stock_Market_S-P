import streamlit as st
# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import pandas as pd
import yfinance as yf
yf.pdr_override()
from datetime import datetime
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
pd.pandas.set_option('display.max_columns',None)
import seaborn as sns
from pandas_datareader.data import DataReader
import seaborn as sns
st.title('Stock Forecast of S&P 500')

stock = '^GSPC'
start = datetime(2000,1,1)
df = pdr.DataReader(stock,data_source='yahoo', start=start)
df = df.reset_index()
st.subheader("The Present Stock Market_Rate")
def plot_raw_data(df,title):
       fig = px.line(title = title)
       for i in df.columns[1:]:
           fig.add_scatter(x = df['Date'], y = df[i], name = i)
       st.plotly_chart(fig)
       fig.show()
print(plot_raw_data(df, 'The Present Stock Market_Rate'))
activites = ["Daily_Returns","Moving_average",'Stocks_during_COVID-19']
choice =st.sidebar.selectbox('Select_ Task',activites)
if choice == "Daily_Returns":
    def daily_return(df):
        df_daily_return = df.copy()
        for i in df.columns[1:]:
            for j in range(1,len(df)):
                   # Calculate the percentage of change from the previous day
                   df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1])*100
            df_daily_return[i][0]=0
            return df_daily_return
    print(daily_return(df))
    stocks_daily_return = daily_return(df)
    print(stocks_daily_return)
    def plot_raw_datas(stocks_daily_return,title):
       fig = px.line(title = title)
       for i in stocks_daily_return.columns[2:]:
           fig.add_scatter(x = stocks_daily_return['Date'], y = stocks_daily_return[i], name = i)
       st.plotly_chart(fig)
       fig.show()
    print(plot_raw_datas(stocks_daily_return, 'Daily_Returns of the Stocks'))
  
if choice == "Moving_average":
    moving_average_df = df.copy()
    for i in df.columns[1:]:
        moving_average_df[i] = df[i].rolling(21).mean()

    st.write(moving_average_df)
    def ma(df):
        for i in df.columns[2:]:
            fig = px.line(x = df.Date[20:],y = df[i][20:],  title =  i + ' Original Stock Price vs. 21-days Moving Average change')
            fig.add_scatter(x = moving_average_df.Date[20:], y = moving_average_df[i][20:])
        st.plotly_chart(fig)
        fig.show()
    print(ma(df))
    st.text('buy (1) and sell (-1) signal')
    st.text("If price is above the moving average, we buy since the trend is up")
    st.text('If the price is below the moving average, we sell since the trend is down')
    st.text('The moving average is a technical stock analysis tool that works by smoothing out price data or noise due to random short-term price fluctuations.')
    st.text('As you increase the averaging window (look back window), the curve tend to become more smoother')
    signal = df.copy()
    print(signal)
    for i in range(21,len(df)):
        for j in df.columns[1:]:
            if df[j][i]> moving_average_df[j][i]:
                signal[j][i] = 1
            elif df[j][i]< moving_average_df[j][i]:
                signal[j][i] = -1
            else:
                signal[j][i] = 0
    print(signal[21:])
    st.write(signal[21:])
if choice == 'Stocks_during_COVID-19':
    def stocklockdown(df,title):
        for i in df.columns[1:]:
            fig = px.line(df, x='Date', y='Volume',title='First_wave in 2020,Second_wave in 2021)',range_x=['2020-01-01','2022-01-01'])
            fig.update_layout(
                shapes=[
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0="2020-03-19",
                        y0=0,
                        x1="2020-06-30",
                        y1=1,
                        fillcolor="LightSalmon",
                        opacity=0.5,
                        layer="below",
                        line_width=0,
                        ),
                    
                # Second phase Lockdown
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0="2021-02-01",
                    y0=0,
                    x1="2021-05-20",
                    y1=1,
                    fillcolor="Green",
                    opacity=0.5,
                    layer="below",
                    line_width=0,)],
        annotations=[dict(x='2020-03-19', y=0.99, xref='x', yref='paper',showarrow=False, xanchor='right', text='Phase 1 '),
                     dict(x='2021-01-01', y=0.99, xref='x', yref='paper', showarrow=False, xanchor='right', text='Phase 2 ')])
        st.plotly_chart(fig)
        fig.show()
    print(stocklockdown(df,"h"))
st.subheader('CAPM ')
#calculating CAPM
st.info("The Capital Asset Pricing Model (CAPM) is a model that describes the relationship between the expected return and risk of investing in a security")
st.info("It shows that the expected return on a security is equal to the risk-free return plus a risk premium, which is based on the beta of that security.")
beta =1.11
print(stocks_daily_return['Adj Close'].mean())
rf =0
rm = stocks_daily_return['Adj Close'].mean()*252
Expected_return_Security = rf+(beta *(rm-rf)) 
st.write("Expected_return_Security of given dataset using CAPM model  :" ,Expected_return_Security,"%") 
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x
print(normalize(df))
#calculating the portfolio
st.info('A portfolio investment is an asset that is purchased in the expectation that it will earn a return or grow in value, or both')
st.info('It aims to maximize returns by investing in different stocks')
st.subheader('Portfolio of S&P500')
np.random.seed()
weights = np.array(np.random.random(6))
weights  = weights/np.sum(weights)
print(weights)
df_portfolio = normalize(df)
print(df_portfolio)
for i,stock in enumerate(df_portfolio.columns[1:]):
    df_portfolio[stock] = df_portfolio[stock] * weights[i]
    df_portfolio[stock] = df_portfolio[stock] * 1000000
print(df_portfolio[stock])
df_portfolio['Portfolio in $'] = df_portfolio[df_portfolio[stock] != 'Date'].sum(axis =1)
print(df_portfolio)
#portfolio in daily returns
df_portfolio['daily_returns'] = 0.000
for i in range(1,len(df_portfolio)):
    df_portfolio['daily_returns'][i] = ((df_portfolio['Portfolio in $'][i]-df_portfolio['Portfolio in $'][i-1])/df_portfolio['Portfolio in $'][i-1])*100
print(df_portfolio['daily_returns'])
st.write(df_portfolio)
print(plot_raw_data(df_portfolio, 'Portfolio of S&P500'))
st.subheader('Sharpe_Model')
st.info('Sharpe ratio is the measure of risk-adjusted return of a financial portfolio')

rf = 0
sharp_Ratio = ((df_portfolio['daily_returns'].mean()-rf)/df_portfolio['daily_returns'].std())* np.sqrt(252)
st.write("The Sharpe_ratio of S&P 500 is ", sharp_Ratio)
#training and testing the data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dt = sc.fit_transform(np.array(df['Close']).reshape(-1,1))
training_data = int(len(dt)*0.80)
test_data = len(dt)- training_data
train_data,test_data = dt[0:training_data,:], dt[training_data:len(dt),:1]
train_data.shape,test_data.shape
def traintest(df,timestep):
    datax, datay= [],[]
    for i in range(len(df) - timestep - 1):
        a = df[i:(i+timestep),0]
        datax.append(a)
        datay.append(df[i+timestep ,0])
    return np.array(datax), np.array(datay)
timestep = 500
x_train,y_train =  traintest(train_data, timestep)
x_test , y_test = traintest(test_data,timestep)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test  = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from keras import optimizers
from keras.layers.core import Activation
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(500, 1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.summary()
history = model.fit(x_train,y_train,validation_data = (x_test,y_test), batch_size = 63, epochs = 10, verbose = 1)

train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
predictions = model.predict(x_test)
predictions = sc.inverse_transform(predictions)
print(predictions)
import math
from sklearn.metrics import mean_squared_error
print("Mean_sraured_error:" ,mean_squared_error(y_train,train_predict))
print("Root_Mean_sraured_error:" ,np.sqrt(mean_squared_error(y_train,train_predict)))
print("Mean_sraured_error:" ,mean_squared_error(y_test,test_predict))
print("Root_Mean_sraured_error:" ,np.sqrt(mean_squared_error(y_test,test_predict)))

print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

# Forecasting the future
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import plot_plotly
from fbprophet import Prophet
fmodel = Prophet()
fmodel.fit(df[['Date','Close']].rename(columns = {"Date":'ds','Close':"y"}))
df_future = fmodel.make_future_dataframe(periods=365)

df_future.tail()
df_prd = fmodel.predict(df_future)

st.subheader('Forecast data')
st.write(df_prd.tail())

fig1 = plot_plotly(fmodel, df_prd)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = fmodel.plot_components(df_prd)
st.write(fig2)
