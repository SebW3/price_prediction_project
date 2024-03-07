import pandas as pd
from sqlalchemy import create_engine
from api_keys import MySQL_login
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Creating connection and reading data
host, user, password, database = MySQL_login()

engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")

data = pd.read_sql("SELECT * FROM project.btc_market_data order by 'timestamp'", engine)
data.sort_values(by='timestamp', inplace=True)

# Adding indicators
# MA (Moving Average)
data["MA"] = data["close"].rolling(window=5).mean()

# RSI
def calculate_rsi(data):
    delta = data['close'].astype(float).diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


data['rsi'] = calculate_rsi(data)

# MACD (Moving Average Convergence Divergence)
exp12 = data["close"].ewm(span=12, adjust=False).mean()
exp26 = data["close"].ewm(span=26, adjust=False).mean()
macd = exp12 - exp26
signal = macd.ewm(span=9, adjust=False).mean()

data["MACD"] = macd

# this fragment works in Jupyter notebook but not here
# # updating database
# with engine.connect() as connection:
#     for index, row in data.iterrows():
#         if str(row["MA"]) == "nan" or str(row["rsi"]) == "nan" or str(row["MACD"]) == "nan":
#             continue  # can't insert "nan" (None) value
#         update_query = f"""
#         UPDATE project.btc_market_data
#         SET MA = {row['MA']}, rsi = {row['rsi']}, MACD = {row['MACD']}
#         WHERE id = {row['id']}
#         """
#         connection.execute(update_query)

engine.dispose()

# Cleaning data
previous_date = datetime.strptime(data["timestamp"][0], "%Y-%m-%d %H:%M:%S")

data.dropna(inplace=True)
tolerance = 1  # Define tolerance for time difference

for x in data.index:
    if data.loc[x, "rsi"] > 99 or data.loc[x, "rsi"] < 1:
        data.drop(x, inplace=True)
    try:
        data_object = datetime.strptime(data.loc[x, "timestamp"], "%Y-%m-%d %H:%M:%S")
    except:
        continue

    # Calculate the time difference in seconds
    time_difference = abs((previous_date - data_object).total_seconds())

    if abs(time_difference - 3600) > tolerance:
        print(time_difference)
        data.drop(x, inplace=True)

    previous_date = data_object


# Creating graph
fig, ax = plt.subplots(figsize=(10, 6))
MA = np.array(data["MA"])
points = np.array(data["close"])
dates = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in data["timestamp"]]

ax.plot(dates, points, label="close price")
ax.plot(dates, MA, label="MA")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xlabel("date")
plt.xticks(rotation=45)
plt.ylabel("close price $")
plt.legend()
plt.show()
