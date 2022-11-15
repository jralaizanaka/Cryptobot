import requests
import numpy as np
import pandas as pd
import datetime
import json


def signal_model(actual, predicted):
    """retrieve the signal of the model"""
    if actual > predicted:
        return "Sell"
    else:
        return "Buy"


def get_ticker_order_book(ticker_name, nb_depth):
    """Retrieve the order book of a specific ticker.
    Arguments: the ticker of the cryptocurrency, and the depth level
    Use of Binance API associated to the "depth"
    Return the real time order book of the cryptocurrency
    """

    # defining key/request url
    key_depth = "https://api.binance.com/api/v3/depth"  # to get the order book

    # requesting data from url
    data_depth = requests.get(key_depth, params=dict(symbol=ticker_name))
    # Use json module
    data_depth = data_depth.json()

    # Get order book
    frames = {
        side: pd.DataFrame(
            data=data_depth[side], columns=["price", "quantity"], dtype=float
        )
        for side in ["bids", "asks"]
    }

    frames_list = [frames[side].assign(side=side) for side in frames]
    data = pd.concat(frames_list, axis="index", ignore_index=True, sort=True)

    databids = data[(data.side == "bids")][["quantity", "price"]]
    databids.columns = ["quantity_bid", "bid_price"]
    # display(databids.head())

    dataasks = data[(data.side == "asks")][["price", "quantity"]]
    dataasks.columns = ["ask_price", "quantity_ask"]
    dataasks.reset_index(drop=True, inplace=True)
    # display(dataasks.head())

    result = pd.concat([databids, dataasks], axis=1, join="inner").head(nb_depth)

    return result


def get_current_time():
    """Give the current time"""
    # defining key/request url
    key_time = "https://api.binance.com/api/v3/time"  # to get the current time

    # requesting data from url
    data_time = requests.get(key_time)

    # Use json module
    data_time = data_time.json()

    # Print real time Price and Time
    timestamp = data_time["serverTime"]
    your_dt = datetime.datetime.fromtimestamp(
        int(timestamp) / 1000
    )  # using the local timezone
    realtime = your_dt.strftime("%Y-%m-%d %H:%M:%S")
    return realtime


def load_hd_to_json(dict, file_name):
    with open(file_name, "w") as fp:
        json.dump(dict, fp, indent=4)


def update_ptf_json(entry, file_name):
    # 1. Read file contents
    with open(file_name, "r") as file:
        data = json.load(file)
    # 2. Update json object
    data.append(entry)
    # 3. Write json file
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)


def valo(price, df_direction, df_state, df_quantity, df_amount):
    if df_state.iloc[-1] == "c":
        return df_amount.sum()
    else:
        if df_direction.iloc[-1] == "Buy":
            return df_amount.sum() + (df_quantity.iloc[-1] * price)
        else:
            return df_amount.sum() + (-df_quantity.iloc[-1] * price)
