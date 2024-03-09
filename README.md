# Price prediction project

This project will try to predict future prices of cryptocurrencies by using data science and machine learning

## Setup
### Required libraries:
- sqlalchemy
- python-binance
- mysql-connector-python
- pandas

<code>def get_market_data(days=None, start_date=None)</code><br>
start_date must be in yyyy-mm-dd format. Code is made in such a way that that is date from witch to download into the past

<code>def divide_the_data(train_data_percentage=0.8, delete_from_db=False, save_test_data=False)</code><br>
this function divides the data to training and test for machine learning purposes<br>
"train_data_percentage" determine the % of the data that will be used for training<br>
if "delete_from_db" is true it will delete test data from database and it will change "save_test_data" to true<br>
if "save_test_data" will save test data to test_data.csv file<br>
