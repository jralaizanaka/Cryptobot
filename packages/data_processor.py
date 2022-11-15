import numpy as np
import pandas as pd
import datetime
import requests


def normalise_zero_base(df):
    """ " Data Preprocessing: normalization"""
    return df / df.iloc[0] - 1


def extract_window_data(df, window_len=5, zero_base=True):
    """extract window data for deep learning"""
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx : (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def get_historical_data_from_binance(ticker_name, frequency, limit, format_date):
    """
    :param frequency: frequency of the quotes (1m,1d,1w...)
    :param limit: Number of maximum line
    :param fomat_date: Format of the date
    :return: the date, open, high, low, close price of the cryptocurrency
    """
    url = f"https://api.binance.com/api/v3/klines?limit={limit}&symbol={ticker_name}&interval={frequency}"
    data_url = requests.get(url)
    data_url = data_url.json()
    liste_column = [
        "Date",
        "Open price",
        "High price",
        "Low price",
        "Close price",
        "Volume",
    ]
    liste_kline = [
        {liste_column[i]: elem[i] for i in range(0, 6)} for elem in data_url
    ]  # built the list of dictionnary
    for elem in liste_kline:  # transform the "Date" format
        your_dt = datetime.datetime.fromtimestamp(int(elem["Date"]) / 1000)
        reformat = your_dt.strftime(format_date)
        elem["Date"] = reformat
    return liste_kline


def extract_transform(ticker_name, frequency, limit, format_date):
    mydict = get_historical_data_from_binance(
        ticker_name, frequency, limit, format_date
    )
    df_live = pd.DataFrame(mydict)
    df_live.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df_live = df_live[["Date", "Open", "High", "Low", "Close"]]
    convert_dict = {
        "Date": str,
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
    }
    df_live = df_live.astype(convert_dict)
    df_live["Date"] = df_live["Date"].astype("datetime64[s]")
    df_live = df_live.set_index("Date")
    return df_live
