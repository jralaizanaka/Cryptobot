import json
import requests
import pandas as pd
import datetime
from crypto_list import*


class Ticker():
    """Cryptocurrencies real time module"""

    def __init__(self, ticker_name):
        self.ticker_name = ticker_name

    def get_current_time(self):
        """Give the current time"""
        # defining key/request url
        key_time = "https://api.binance.com/api/v3/time"  # to get the current time

        # requesting data from url
        data_time = requests.get(key_time)

        # Use json module
        data_time = data_time.json()

        # Print real time Price and Time
        timestamp = data_time["serverTime"]
        your_dt = datetime.datetime.fromtimestamp(int(timestamp) / 1000)  # using the local timezone
        realtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
        return realtime

    def get_ticker_price(self):
        """Retrieve the real time price about specific ticker.
           Arguments: the ticker of the cryptocurrency
           Use of Binance API to get the different elements
           Return the real time quote of the cryptocurrency
        """

        # defining key/request url
        key_price = f"https://api.binance.com/api/v3/ticker/price?symbol={self.ticker_name}"  # to get the real time price

        # requesting data from url
        data_price = requests.get(key_price)

        # Use json module
        data_price = data_price.json()

        return (data_price)
        # return (data_price ['symbol'],data_price ['price'])

    def get_ticker_depth(self, nb_depth):
        """Retrieve the order book of a specific ticker.
           Arguments: the ticker of the cryptocurrency, and the depth level
           Use of Binance API associated to the "depth"
           Return the real time order book of the cryptocurrency
        """

        # defining key/request url
        key_depth = f"https://api.binance.com/api/v3/depth?limit={nb_depth}&symbol={self.ticker_name}"  # to get the order book

        # requesting data from url
        data_depth = requests.get(key_depth)
        # Use json module
        data_depth = data_depth.json()

        return data_depth

    def get_ticker_order_book(self, nb_depth):
        """Retrieve the order book of a specific ticker.
           Arguments: the ticker of the cryptocurrency, and the depth level
           Use of Binance API associated to the "depth"
           Return the real time order book of the cryptocurrency
        """

        # defining key/request url
        key_depth = "https://api.binance.com/api/v3/depth"  # to get the order book

        # requesting data from url
        data_depth = requests.get(key_depth,
                                  params=dict(symbol=self.ticker_name))
        # Use json module
        data_depth = data_depth.json()

        # Get order book
        frames = {side: pd.DataFrame(data=data_depth[side], columns=["price", "quantity"],
                                     dtype=float)
                  for side in ["bids", "asks"]}

        frames_list = [frames[side].assign(side=side) for side in frames]
        data = pd.concat(frames_list, axis="index",
                         ignore_index=True, sort=True)

        databids = data[(data.side == "bids")][["quantity", "price"]]
        databids.columns = ["quantity_bid", "bid_price"]
        # display(databids.head())

        dataasks = data[(data.side == "asks")][["price", "quantity"]]
        dataasks.columns = ["ask_price", "quantity_ask"]
        dataasks.reset_index(drop=True, inplace=True)
        # display(dataasks.head())

        result = pd.concat([databids, dataasks], axis=1, join="inner")

        display(result.head(nb_depth))

    def get_info(self):
        NB_LAST_TRADE = 5

        # defining key/request url
        key_change = f"https://api.binance.com/api/v1/ticker/24hr?symbol={self.ticker_name}"  # to get the order book

        key_info = f"https://api.binance.com/api/v3/exchangeInfo?symbol={self.ticker_name}"

        key_last_trade = f"https://api.binance.com/api/v3/trades?limit={NB_LAST_TRADE}&symbol={self.ticker_name}"

        # requesting data from url
        data_change = requests.get(key_change)
        data_info = requests.get(key_info)
        data_last_trade = requests.get(key_last_trade)

        # Use json module
        data_change = data_change.json()
        data_info = data_info.json()
        data_last_trade = data_last_trade.json()

        return (data_change["priceChangePercent"],
                data_change["quoteVolume"],
                data_info["symbols"][0]['baseAsset'],
                dict_crypto_list_name()[data_info["symbols"][0]['baseAsset']][0],
                data_last_trade[0]["price"],
                data_last_trade[0]["qty"],
                data_last_trade[1]["price"],
                data_last_trade[1]["qty"],
                data_last_trade[2]["price"],
                data_last_trade[2]["qty"],
                data_last_trade[3]["price"],
                data_last_trade[3]["qty"],
                data_last_trade[4]["price"],
                data_last_trade[4]["qty"]
                )


    def convert_to_json(self, dict, file_name):
        with open(file_name, 'w') as fp:
            json.dump(dict, fp, indent=4)


class Historic_data:
    """Cryptocurrencies historival data module"""

    def __init__(self, ticker_name):
        self.ticker_name = ticker_name

    def get_historical_data(self, frequency, limit):

        url = f"https://api.binance.com/api/v3/klines?limit={limit}&symbol={self.ticker_name}&interval={frequency}"
        data_url = requests.get(url)
        data_url = data_url.json()

        liste_column = ["Open price", "High price", "Low price", "Close price", "Volume", "Kline Close time",
                        "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                        "Taker buy quote asset volume", "ignore"]
        liste = data_url[1]
        timestamp = liste[0]
        your_dt = datetime.datetime.fromtimestamp(int(timestamp) / 1000)  # using the local timezone
        realtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
        dict_kline = {datetime.datetime.fromtimestamp(int(elem[0]) / 1000).strftime("%Y-%m-%d"): {x: elem[i] for x, i in
                                                                                                  zip(liste_column,
                                                                                                      range(1, 11))} for
                      elem in data_url}

        return dict_kline


    def load_hd_to_json(self, dict, file_name):
        with open(file_name, 'w') as fp:
            json.dump(dict, fp, indent=4)



