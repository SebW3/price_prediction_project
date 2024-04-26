from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, BayesianRidge, RANSACRegressor, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


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

pd.options.display.float_format = '{:.6f}'.format

def model_evaluation(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2_Score = metrics.r2_score(y_test, y_pred)

    return pd.DataFrame([MAE, MSE, RMSE, R2_Score], index=['MAE', 'MSE', 'RMSE', 'R2-Score'], columns=[model_name])


def residuals(model, X_test, y_test):
    '''
    Creates predictions on the features with the model and calculates residuals
    '''
    y_pred = model.predict(X_test)
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    return df_results

def linear_assumption(model, X_test, y_test):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model
    '''
    df_results = residuals(model, X_test, y_test)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=80)
    sns.regplot(x='Predicted', y='Actual', data=df_results, lowess=True, ax=ax[0],
                color='#0055ff', line_kws={'color': '#ff7000', 'ls': '--', 'lw': 2.5})
    ax[0].set_title('Actual vs. Predicted Values', fontsize=15)
    ax[0].set_xlabel('Predicted', fontsize=12)
    ax[0].set_ylabel('Actual', fontsize=12)

    sns.regplot(x='Predicted', y='Residuals', data=df_results, lowess=True, ax=ax[1],
                color='#0055ff', line_kws={'color': '#ff7000', 'ls': '--', 'lw': 2.5})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=15)
    ax[1].set_xlabel('Predicted', fontsize=12)
    ax[1].set_ylabel('Residuals', fontsize=12)
    plt.show()

def compare_plot(df_comp):
    df_comp.reset_index(inplace=True)
    df_comp_head = df_comp.head(20)
    df_comp_head.plot(y=['Actual','Predicted'], kind='bar', figsize=(20,7), width=0.8)
    plt.title('Predicted vs. Actual Target Values for Test Data', fontsize=20)
    plt.ylabel('Selling_Price', fontsize=15)
    plt.show()

for model in models:
    try:
        print("\n", str(model))
        model.fit(X_train, y_train)

        evaluation = model_evaluation(model, X_test, y_test, str(model))
        #print(evaluation)

        if evaluation[str(model)]["R2-Score"] > 0.95:
            print("good model")
            linear_assumption(model, X_test, y_test)

            y_test_pred = model.predict(X_test)
            df_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
            compare_plot(df_comp)

    except:
        print("error")
