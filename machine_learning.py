from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, BayesianRidge, RANSACRegressor, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


data = pd.read_csv("car_deatils_for_ml_model.csv", index_col=0)
test = data["selling_price"].copy()
data.drop(columns=["selling_price"], inplace=True)
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data, test, test_size=0.2, random_state=0)



# Training models
models = [KNeighborsRegressor(), RandomForestRegressor(), LinearRegression(), Ridge(), Lasso(), ElasticNet(), SVR(), DecisionTreeRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), GaussianProcessRegressor(), HuberRegressor(), PassiveAggressiveRegressor(), OrthogonalMatchingPursuit(), BayesianRidge(), RANSACRegressor(), TheilSenRegressor()]
mse_list = []
mae_list = []
rmse_list = []
r2_list = []

for model in models:
    try:
        print("\n", str(model))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        print(f'Błąd średniokwadratowy: {mse}')
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
        print(f'Błąd bezwzględny średni: {mae}')
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
        print(f'Pierwiastek z błędu średniokwadratowego: {rmse}')
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
        print(f'Współczynnik determinacji: {r2}')
    except:
        print("error")

# lists = [mse_list, mae_list, rmse_list, r2_list]
# for item in lists:
#     for score in item:
#         print(score)
