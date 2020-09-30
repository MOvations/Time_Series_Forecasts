# Basics:
import numpy as np
import pandas as pd
import datetime as dt

# Solvers:
import fbprophet as fb

# Sources:
import yfinance as fyf

# Plotting:
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")  # Best for White Backgrounds
# plt.style.use('dark_background')     #Best for Black Backgrounds


# easy manual portfolio import meathod
tickers = "FB AMZN GOOG INTC CBRE KO DJP IAU GLD"
tickers = tickers.split()

end = pd.Timestamp.utcnow()
# Go back 2500 business days
start = end - 3500 * pd.tseries.offsets.BDay()

# start = '2016-10-01'
# # end = str(dt.date.today().strftime('%Y-%m-%d'))
# end = dt.datetime.strftime(dt.datetime.now() - dt.timedelta(1), '%Y-%m-%d')
print(end)


# import from file
######## TBD FUTURE WORK #########

# get all the data
data_raw = fyf.download(tickers, start)

# just what we need for now
data = data_raw["Adj Close"]
data.tail()

# plot raw imported data
# plt.figure(figsize=(20, 9))
# plt.plot(data)
# plt.legend(tickers, loc=2)
# plt.show()

stock = "GOOG"

amazon = data[[stock]]
amazon.index

amazon = amazon.rename(columns={stock: "y"})
amazon["ds"] = amazon.index
# amazon.set_index(['ds'], inplace=True)
amazon.reset_index(inplace=True)
amazon.head()

df_max = amazon.max()

m = fb.Prophet()
m.fit(amazon)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()



fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
plt.show()