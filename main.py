import streamlit as st
from extract import*
from crypto_list import*


def get_generic_data(ticker, nb_depth):
    ticker = Ticker(ticker)  # Instanciate ticker

    current_time = ticker.get_current_time()  # retrieve current time
    symbol_price = ticker.get_ticker_price()  # retrieve current price
    order_book = ticker.get_ticker_depth(nb_depth)  # retrieve order book of nb_depth
    pricechange, quotevolume, baseAsset, nameAsset, data_last_trade_price1, data_last_trade_qty1, data_last_trade_price2, data_last_trade_qty2, data_last_trade_price3, data_last_trade_qty3, data_last_trade_price4, data_last_trade_qty4, data_last_trade_price5, data_last_trade_qty5 = ticker.get_info()
    del order_book['lastUpdateId']

    # create the dict_info dictionnary
    dict_time = {"time": current_time}
    dict_info = {**dict_time, **symbol_price}
    dict_info["priceChangePercent24h"] = pricechange
    dict_info["quotevolume24h"] = quotevolume
    dict_info["orderbook"] = order_book
    dict_info["info"] = {"baseAsset": baseAsset,
                         "namebaseAsset": nameAsset,
                         "1st last trade": {"price": data_last_trade_price1,
                                            "qty": data_last_trade_qty1},

                         "2nd last trade": {"price": data_last_trade_price2,
                                            "qty": data_last_trade_qty2},

                         "3rd last trade": {"price": data_last_trade_price3,
                                            "qty": data_last_trade_qty3},

                         "4th last trade": {"price": data_last_trade_price4,
                                            "qty": data_last_trade_qty4},

                         "5th last trade": {"price": data_last_trade_price5,
                                            "qty": data_last_trade_qty5},

                         }
    return dict_info


def execute():
    st.title("CRYPTOBOT")
    ticker = st.selectbox("Liste des Tickers", ticker_crypto_list())
    TICKER_NAME = ticker
    LIMIT = 30
    FREQUENCY = "1d"
    NB_DEPTH = 1

    st.write(f"LIVE QUOTES of {TICKER_NAME } ")
    st.write(get_generic_data(TICKER_NAME, NB_DEPTH))

    st.write(f"HISTORICAL DATA {FREQUENCY}")
    hd = Historic_data(TICKER_NAME)
    st.write(hd.get_historical_data(FREQUENCY, LIMIT))





if __name__ == '__main__':
    execute()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
