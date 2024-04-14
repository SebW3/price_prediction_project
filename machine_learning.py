from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import MySQL_functions
import pandas as pd
import numpy as np

def convert_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except ValueError:
        return None

data, target = MySQL_functions.divide_the_data(train_data_percentage=0.8)

data.dropna(inplace=True)
target.dropna(inplace=True)

length = abs(len(data)-len(target))
if len(data) > len(target):
    data.drop(data.tail(length).index, inplace = True)
elif len(data) < len(target):
    target.drop(target.tail(length).index, inplace = True)

label_encoder = LabelEncoder()
data['timestamp'] = pd.to_datetime(data['timestamp'])

data['timestamp'] = (data['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

target['timestamp'] = pd.to_datetime(target['timestamp'])
target['timestamp'] = (target['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

data['timestamp'] = data['timestamp'].astype('float64')
target['timestamp'] = target['timestamp'].astype('float64')
data.drop('id', axis=1, inplace=True)
target.drop('id', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)



# Training models
models = [KNeighborsRegressor(), RandomForestRegressor(), LinearRegression()]
mse_list = []
mae_list = []
rmse_list = []
r2_list = []

for model in models:
    print("\n", str(model))
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)


    prices_pred = model.predict(X_test)
    mse = mean_squared_error(X_test, prices_pred)
    mse_list.append(mse)
    print(f'Błąd średniokwadratowy: {mse}')
    mae = mean_absolute_error(X_test, prices_pred)
    mae_list.append(mae)
    print(f'Błąd bezwzględny średni: {mae}')
    rmse = np.sqrt(mse)
    rmse_list.append(rmse)
    print(f'Pierwiastek z błędu średniokwadratowego: {rmse}')
    r2 = r2_score(X_test, prices_pred)
    r2_list.append(r2)
    print(f'Współczynnik determinacji: {r2}')

lists = [mse_list, mae_list, rmse_list, r2_list]
for item in lists:
    for score in item:
        print(score)
