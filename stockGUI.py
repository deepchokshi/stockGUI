import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import yfinance as yf
import altair.vegalite.v

start = "2010-04-01"
end = "2021-03-30"
user_input = st.text_input("Enter Stock Ticker","EXIDEIND.NS")
stock = yf.Ticker(user_input)
name = stock.info["shortName"]
st.title(name +" PRICE PREDICTION")
df = yf.download(user_input,start,end)
#df = data.DataReader(user_input,"yahoo",start,end)
st.subheader("Date from 2010-2021")

st.subheader("Closing Price")
fig = plt.figure(figsize=(15,9))
plt.plot(df.Close)
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig)

df_train = pd.DataFrame(df["Close"][0:int(len(df)*0.95)])
df_test = pd.DataFrame(df["Close"][int(len(df)*0.95):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

train = scaler.fit_transform(df_train)

x_train = []
y_train = []

for i in range(60, len(train)):
    x_train.append(train[i - 60:i, 0])
    y_train.append(train[i, 0])

x_test = []
y_test = []
test = scaler.fit_transform(df_test)
for i in range(60, len(test)):
    x_test.append(test[i-60:i, 0])
    y_test.append(test[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)
x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model("LSTM.h5")
y_prediction = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler
y_prediction = y_prediction * scale_factor
y_test = y_test * scale_factor

st.subheader("Original vs Prediction")
fig2 = plt.figure(figsize=(15,9))
plt.plot(y_test,"b",label="Original")
plt.plot(y_prediction,"r",label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig2)



