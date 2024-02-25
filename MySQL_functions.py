import mysql.connector
from api_keys import MySQL_login

host, user, password, database = MySQL_login()

def temp_instert_into_db(df):
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor()


    for i in range(len(df)):
        cursor.execute(
            f"INSERT INTO project.btc_market_data (timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (df['timestamp'][i], df['open'][i], df['high'][i], df['low'][i], df['close'][i], df['volume'][i],
             float(df['close_time'][i]), df['quote_asset_volume'][i], int(df['number_of_trades'][i]),
             df['taker_buy_base_asset_volume'][i], df['taker_buy_quote_asset_volume'][i]))
    print("data uploaded to database")
