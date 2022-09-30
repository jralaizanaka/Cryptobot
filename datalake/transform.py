import pandas as pd

NAME_FILE = "ADABKRW-1h-2020-08.csv"
kline1 = pd.read_csv(NAME_FILE, sep=",", header=None)


header_name = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]
kline1.columns = header_name
# print(kline1.head())
print(kline1.describe)
