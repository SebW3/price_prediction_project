import mysql.connector
from sqlalchemy import create_engine, MetaData, select, Table, delete
import pandas as pd
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

    conn.commit()
    print("data uploaded to database")

    cursor.close()
    conn.close()


def get_last_date():
    engine = create_engine(f"mysql://{user}:{password}@localhost/{database}", echo=True)
    conn = engine.connect()
    metadata = MetaData()

    btc_market_data = Table("btc_market_data", metadata, schema="project", autoload_with=engine, extend_existing=True)
    select_stmt = select(btc_market_data).order_by(btc_market_data.c.timestamp)

    result = conn.execute(select_stmt).fetchall()

    conn.close()

    print(result[-1][1][:10])
    return result[0][1][:10]


def divide_the_data(train_data_percentage=0.8, delete_from_db=False, save_test_data=False):
    engine = create_engine(f"mysql://{user}:{password}@localhost/{database}", echo=True)
    conn = engine.connect()
    metadata = MetaData()

    btc_market_data = Table("btc_market_data", metadata, schema="project", autoload_with=engine, extend_existing=True)
    select_stmt = select(btc_market_data).order_by(btc_market_data.c.timestamp)

    result = conn.execute(select_stmt).fetchall()

    df = pd.DataFrame(result, columns=btc_market_data.columns)

    split_index = int(train_data_percentage * len(df))  # how much % training data to test data
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    if delete_from_db:
        save_test_data = True
        conn.execute(delete(btc_market_data).where(btc_market_data.c.timestamp >= test_data.iloc[:, 1].min()))
        conn.commit()

    conn.close()

    if save_test_data:
        test_data.to_csv('test_data.csv', index=False)

    return train_data, test_data
