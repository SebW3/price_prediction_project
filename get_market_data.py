from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from api_keys import binance_api_key
import MySQL_functions


def get_market_data(days=None, start_date=None):
    if days is None:
        print("enter how many days to download")
        return 1

    start_date = MySQL_functions.get_last_date()
    if start_date:
        start_date_temp = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=42)

    def calculate_rsi(data):
        delta = data['close'].astype(float).diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


    limit = days*24  # number of days * 24 hours

    client = Client(binance_api_key())


    data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    while limit > 0:
        print(limit, "left to download")
        start_date_temp = start_date_temp - timedelta(days=41)
        start_date = int(datetime.strptime(str(start_date_temp)[:10], '%Y-%m-%d').timestamp()) * 1000
        print(start_date)
        symbol = "BTCUSDT"

        if limit < 1000:
            candles_btc = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, endTime=start_date, limit=limit)
        else:
            candles_btc = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, endTime=start_date ,limit=984)

        df = pd.DataFrame(candles_btc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['rsi'] = calculate_rsi(df)
        data = data._append(df, ignore_index=True)
        limit -= 984

    print(len(data))
    data.drop_duplicates(inplace=True)
    data.sort_values(by="timestamp")
    data.drop("ignore", axis=1, inplace=True)
    print(data)

    MySQL_functions.temp_instert_into_db(data)
