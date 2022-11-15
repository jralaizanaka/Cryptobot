import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
import time
from datetime import date
from numpy import loadtxt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import time
import boto3
import tempfile
import packages.data_processor as dp
import packages.trading_bot as tb
import data.crypto_list as dc
from smart_open import open


# *********************************LOAD MODEL FROM S3********************************************************
@st.cache(allow_output_mutation=True)
def load_model2(name_crypto):

    session = boto3.Session(
        aws_access_key_id="AKIAUAPRSMJOC4NHKNO4",
        aws_secret_access_key="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",
        region_name="eu-west-3",
    )

    s3 = session.resource("s3")
    bucket_name = "datalakedatascientest"
    folder = "crypto_models"
    name_xxxEUR = f"{name_crypto}.h5"
    OBJECT_NAME = f"{folder}/{name_crypto}.h5"

    with tempfile.TemporaryDirectory() as tempdir:
        s3.Bucket(bucket_name).download_file(OBJECT_NAME, f"{tempdir}/{name_xxxEUR}")
        model = load_model(f"{tempdir}/{name_xxxEUR}")
    return model


# ------------------------------------PARAM-----------------------------------------------------------

TARGET_COL = "Close"
WINDOW_LEN = 5
FREQUENCY = "1m"
LIMIT = 10
LIMIT_CHART = 50
FORMAT_DATE = "%Y-%m-%d %X"
NB_DEPTH = 1
TAILLE_POSE = 20000
SEUIL_LONG = 0.25 / 100
SEUILS_SHORT = -0.25 / 100


class Crypto:
    def __init__(self, name, up_trigger, down_trigger):
        self.name = name
        self.up_trigger = up_trigger
        self.down_trigger = down_trigger

    def g_name(self):
        return dc.dict_crypto_list_name()[self.name[:-3]][0]

    def g_ptf(self):
        """Retrieve tradinf portfilios from s3 bucket"""
        bucket_name = "datalakedatascientest"
        session = boto3.Session(
            aws_access_key_id="AKIAUAPRSMJOC4NHKNO4",
            aws_secret_access_key="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",
            region_name="eu-west-3",
        )
        folder = "crypto_portfolios"
        file_crypto_name = f"{self.name}.json"

        file_name = f"{self.name}.json"
        url = f"s3://{bucket_name}/{folder}/{file_crypto_name}"
        with open(url, "rb", transport_params={"client": session.client("s3")}) as f:
            data_url = json.load(f)

        data_url = json.dumps(data_url)
        df = pd.read_json(data_url)
        df = df[
            ["Timestamp", "Name", "Direction", "Price", "Quantity", "State", "Amount"]
        ]

        return df

    def update_ptf(self, entry):
        """Retrieve tradinf portfilios from s3 bucket"""
        bucket_name = "datalakedatascientest"
        session = boto3.Session(
            aws_access_key_id="AKIAUAPRSMJOC4NHKNO4",
            aws_secret_access_key="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",
            region_name="eu-west-3",
        )
        folder = "crypto_portfolios"
        file_crypto_name = f"{self.name}.json"

        file_name = f"{self.name}.json"
        save_to = f"{folder}/{file_name}"
        url = f"s3://{bucket_name}/{folder}/{file_crypto_name}"
        # 1. Read file contents
        with open(url, "rb", transport_params={"client": session.client("s3")}) as f:
            data_url = json.load(f)
        # 2. Update json object
        data_url.append(entry)
        # 3. Write json file
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = f"{tempdir}/{file_name}"
            with open(temp_path, "w") as file:
                json.dump(data_url, file, indent=4)
            # Upload the file
            s3_client = session.client("s3")
            s3_client.upload_file(temp_path, bucket_name, save_to)

    def g_valo(self, price, df_direction, df_state, df_quantity, df_amount):
        return tb.valo(price, df_direction, df_state, df_quantity, df_amount)

    def g_order_book(self, nb_depth):
        return tb.get_ticker_order_book(self.name, nb_depth)

    def g_chart(self):
        df = dp.extract_transform(self.name, FREQUENCY, LIMIT_CHART, FORMAT_DATE)
        fig = go.Figure()

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                )
            ]
        )

        fig.update_layout(
            title=self.name,
            xaxis_title="Time",
            yaxis_title="Price",
            font=dict(family="Courier New, monospace", size=10, color="#7f7f7f"),
        )

        return fig

    def get_signal(self):
        """return :
        -the live price
        -the predicted price
        -the signal gave by model"""

        df = dp.extract_transform(self.name, FREQUENCY, LIMIT, FORMAT_DATE)
        X_test_live = dp.extract_window_data(df, window_len=WINDOW_LEN)
        model_crypto = load_model2(self.name)
        preds = model_crypto.predict(X_test_live).squeeze()
        preds = df[TARGET_COL].values[:-WINDOW_LEN] * (preds + 1)
        res_pred = preds[0]
        live_price = df["Close"][-1]
        perf_pred = round((res_pred / live_price - 1), 4)
        signal_model = tb.signal_model(live_price, res_pred)
        return (live_price, res_pred, perf_pred, signal_model)


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                                                  SCRIPT STREAMLIT
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ****************************************SET STREAMLIT PAGE********************************************"
st.set_page_config(layout="wide")
with st.sidebar:

    new_title = (
        '<p style="font-family:sans-serif; color:Green; font-size: 42px;">CryptoBot</p>'
    )
    st.markdown(new_title, unsafe_allow_html=True)

    NAME_COL_A = st.selectbox(
        "Selectionnez la 1ère crytpomonnaie", dc.ticker_crypto_list_EUR(), key="bcm1"
    )
    NAME_COL_Bxx = st.selectbox(
        "Selectionnez la 2ème crytpomonnaie", dc.ticker_crypto_list_EUR(), key="bcm2"
    )
    NAME_COL_Cxx = st.selectbox(
        "Selectionnez la 3ème crytpomonnaie", dc.ticker_crypto_list_EUR(), key="bcm3"
    )
    NAME_COL_Dxx = st.selectbox(
        "Selectionnez la 4ème crytpomonnaie", dc.ticker_crypto_list_EUR(), key="bcm4"
    )

# **************************************************************************************************************************

crypto_A = Crypto(NAME_COL_A, up_trigger=SEUIL_LONG, down_trigger=SEUILS_SHORT)
crypto_Bxx = Crypto(NAME_COL_Bxx, up_trigger=SEUIL_LONG, down_trigger=SEUILS_SHORT)
crypto_Cxx = Crypto(NAME_COL_Cxx, up_trigger=SEUIL_LONG, down_trigger=SEUILS_SHORT)
crypto_Dxx = Crypto(NAME_COL_Dxx, up_trigger=SEUIL_LONG, down_trigger=SEUILS_SHORT)

col_A, col_Bxx, col_Cxx, col_Dxx = st.columns(4)

# *************************************traitement du 1er crypto***********************************************
with col_A:
    if crypto_A.g_ptf()["State"].iloc[-1] == "c":
        st.markdown(f"""## {NAME_COL_A} (pas de pose)""")
    else:
        conseil = crypto_A.g_ptf()["Direction"].iloc[-1]
        if conseil == "Buy":
            sens = "Long"
        else:
            sens = "Short"
        st.markdown(f"""## {NAME_COL_A} (en pose {sens} )""")

    live_price_A, res_pred_A, perf_pred_A, signal_model_A = crypto_A.get_signal()

    with st.expander("Voir portefeuille de trading"):
        st.write(crypto_A.g_ptf())

    st.plotly_chart(crypto_A.g_chart(), use_container_width=True)
    st.metric(
        label="Price:",
        value=f"{live_price_A}",
        delta=f"{round(perf_pred_A*100,2)}%",
    )

    # VALO
    st.markdown(f"""## VALO""")
    res_A = crypto_A.g_valo(
        live_price_A,
        crypto_A.g_ptf()["Direction"],
        crypto_A.g_ptf()["State"],
        crypto_A.g_ptf()["Quantity"],
        crypto_A.g_ptf()["Amount"],
    )

    res_percent_A = round(res_A / TAILLE_POSE * 100, 2)
    st.write(f"P&L = {round(res_A,0)} / P&L% = {res_percent_A}%")

    if signal_model_A == "Buy":
        new_title_A = (
            '<p style="font-family:sans-serif; color:Green; font-size: 20px;">BUY</p>'
        )
        st.markdown(new_title_A, unsafe_allow_html=True)
    else:
        new_title_A = (
            '<p style="font-family:sans-serif; color:Red; font-size: 20px;">SELL</p>'
        )
        st.markdown(new_title_A, unsafe_allow_html=True)

    # Order book
    pd_order_book_A = crypto_A.g_order_book(NB_DEPTH)
    best_bid_price_A = pd_order_book_A.iloc[0, 1]
    best_ask_price_A = pd_order_book_A.iloc[0, 2]

    st.write(pd_order_book_A)

    # BOT

    if crypto_A.g_ptf()["State"].iloc[-1] == "c":
        if (signal_model_A == "Buy") & (perf_pred_A >= crypto_A.up_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_A,
                "Direction": "Buy",
                "Price": best_ask_price_A,
                "Quantity": round(TAILLE_POSE / best_ask_price_A, 4),
                "State": "i",
                "Amount": -TAILLE_POSE / best_ask_price_A * best_ask_price_A,
            }
            crypto_A.update_ptf(entry)

        elif (signal_model_A == "Sell") & (perf_pred_A <= crypto_A.down_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_A,
                "Direction": "Sell",
                "Price": best_bid_price_A,
                "Quantity": round(TAILLE_POSE / best_bid_price_A, 4),
                "State": "i",
                "Amount": TAILLE_POSE / best_bid_price_A * best_bid_price_A,
            }
            crypto_A.update_ptf(entry)

    elif crypto_A.g_ptf()["State"].iloc[-1] == "i":

        if (crypto_A.g_ptf()["Direction"].iloc[-1] == "Buy") & (
            signal_model_A == "Sell"
        ):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_A,
                "Direction": "Sell",
                "Price": best_bid_price_A,
                "Quantity": crypto_A.g_ptf()["Quantity"].iloc[-1],
                "State": "c",
                "Amount": crypto_A.g_ptf()["Quantity"].iloc[-1] * best_bid_price_A,
            }

            crypto_A.update_ptf(entry)

        elif (crypto_A.g_ptf()["Direction"].iloc[-1] == "Sell") & (
            signal_model_A == "Buy"
        ):
            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_A,
                "Direction": "Buy",
                "Price": best_ask_price_A,
                "Quantity": crypto_A.g_ptf()["Quantity"].iloc[-1],
                "State": "c",
                "Amount": -crypto_A.g_ptf()["Quantity"].iloc[-1] * best_ask_price_A,
            }
            crypto_A.update_ptf(entry)

# **************************************traitement du deuxième crypto*************************************************
with col_Bxx:

    if crypto_Bxx.g_ptf()["State"].iloc[-1] == "c":
        st.markdown(f"""## {NAME_COL_Bxx} (pas de pose)""")
    else:
        conseil = crypto_Bxx.g_ptf()["Direction"].iloc[-1]
        if conseil == "Buy":
            sens = "Long"
        else:
            sens = "Short"
        st.markdown(f"""## {NAME_COL_Bxx} (en pose {sens} )""")

    (
        live_price_Bxx,
        res_pred_Bxx,
        perf_pred_Bxx,
        signal_model_Bxx,
    ) = crypto_Bxx.get_signal()

    with st.expander("Voir portefeuille de trading"):
        st.write(crypto_Bxx.g_ptf())

    st.plotly_chart(crypto_Bxx.g_chart(), use_container_width=True)
    st.metric(
        label="Price:",
        value=f"{live_price_Bxx}",
        delta=f"{round(perf_pred_Bxx*100,2)}%",
    )

    # VALO
    st.markdown(f"""## VALO""")
    res_Bxx = crypto_Bxx.g_valo(
        live_price_Bxx,
        crypto_Bxx.g_ptf()["Direction"],
        crypto_Bxx.g_ptf()["State"],
        crypto_Bxx.g_ptf()["Quantity"],
        crypto_Bxx.g_ptf()["Amount"],
    )

    res_percent_Bxx = round(res_Bxx / TAILLE_POSE * 100, 2)
    st.write(f"P&L = {round(res_Bxx,0)} / P&L% = {res_percent_Bxx}%")

    if signal_model_Bxx == "Buy":
        new_title_Bxx = (
            '<p style="font-family:sans-serif; color:Green; font-size: 20px;">BUY</p>'
        )
        st.markdown(new_title_Bxx, unsafe_allow_html=True)
    else:
        new_title_Bxx = (
            '<p style="font-family:sans-serif; color:Red; font-size: 20px;">SELL</p>'
        )
        st.markdown(new_title_Bxx, unsafe_allow_html=True)

    # Order book
    pd_order_book_Bxx = crypto_Bxx.g_order_book(NB_DEPTH)
    best_bid_price_Bxx = pd_order_book_Bxx.iloc[0, 1]
    best_ask_price_Bxx = pd_order_book_Bxx.iloc[0, 2]

    st.write(pd_order_book_Bxx)

    # BOT

    if crypto_Bxx.g_ptf()["State"].iloc[-1] == "c":
        if (signal_model_Bxx == "Buy") & (perf_pred_Bxx >= crypto_Bxx.up_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Bxx,
                "Direction": "Buy",
                "Price": best_ask_price_Bxx,
                "Quantity": round(TAILLE_POSE / best_ask_price_Bxx, 4),
                "State": "i",
                "Amount": -TAILLE_POSE / best_ask_price_Bxx * best_ask_price_Bxx,
            }
            crypto_Bxx.update_ptf(entry)

        elif (signal_model_Bxx == "Sell") & (perf_pred_Bxx <= crypto_Bxx.down_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Bxx,
                "Direction": "Sell",
                "Price": best_bid_price_Bxx,
                "Quantity": round(TAILLE_POSE / best_bid_price_Bxx, 4),
                "State": "i",
                "Amount": TAILLE_POSE / best_bid_price_Bxx * best_bid_price_Bxx,
            }
            crypto_Bxx.update_ptf(entry)

    elif crypto_Bxx.g_ptf()["State"].iloc[-1] == "i":

        if (crypto_Bxx.g_ptf()["Direction"].iloc[-1] == "Buy") & (
            signal_model_Bxx == "Sell"
        ):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Bxx,
                "Direction": "Sell",
                "Price": best_bid_price_Bxx,
                "Quantity": crypto_Bxx.g_ptf["Quantity"].iloc[-1],
                "State": "c",
                "Amount": crypto_Bxx.g_ptf["Quantity"].iloc[-1] * best_bid_price_A,
            }

            crypto_Bxx.update_ptf(entry)

        elif (crypto_Bxx.g_ptf()["Direction"].iloc[-1] == "Sell") & (
            signal_model_Bxx == "Buy"
        ):
            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Bxx,
                "Direction": "Buy",
                "Price": best_ask_price_Bxx,
                "Quantity": crypto_Bxx.g_ptf()["Quantity"].iloc[-1],
                "State": "c",
                "Amount": -crypto_Bxx.g_ptf()["Quantity"].iloc[-1] * best_ask_price_Bxx,
            }
            crypto_Bxx.update_ptf(entry)

# ********************************************** traitement de la troisième crypto*********************************************

with col_Cxx:

    if crypto_Cxx.g_ptf()["State"].iloc[-1] == "c":
        st.markdown(f"""## {NAME_COL_Cxx} (pas de pose)""")
    else:
        conseil = crypto_Cxx.g_ptf()["Direction"].iloc[-1]
        if conseil == "Buy":
            sens = "Long"
        else:
            sens = "Short"
        st.markdown(f"""## {NAME_COL_Cxx} (en pose {sens} )""")

    (
        live_price_Cxx,
        res_pred_Cxx,
        perf_pred_Cxx,
        signal_model_Cxx,
    ) = crypto_Cxx.get_signal()

    with st.expander("Voir portefeuille de trading"):
        st.write(crypto_Cxx.g_ptf())

    st.plotly_chart(crypto_Cxx.g_chart(), use_container_width=True)
    st.metric(
        label="Price:",
        value=f"{live_price_Cxx}",
        delta=f"{round(perf_pred_Cxx*100,2)}%",
    )

    # VALO
    st.markdown(f"""## VALO""")
    res_Cxx = crypto_Cxx.g_valo(
        live_price_Cxx,
        crypto_Cxx.g_ptf()["Direction"],
        crypto_Cxx.g_ptf()["State"],
        crypto_Cxx.g_ptf()["Quantity"],
        crypto_Cxx.g_ptf()["Amount"],
    )

    res_percent_Cxx = round(res_Cxx / TAILLE_POSE * 100, 2)
    st.write(f"P&L = {round(res_Cxx,0)} / P&L% = {res_percent_Cxx}%")

    if signal_model_Cxx == "Buy":
        new_title_Cxx = (
            '<p style="font-family:sans-serif; color:Green; font-size: 20px;">BUY</p>'
        )
        st.markdown(new_title_Cxx, unsafe_allow_html=True)
    else:
        new_title_Cxx = (
            '<p style="font-family:sans-serif; color:Red; font-size: 20px;">SELL</p>'
        )
        st.markdown(new_title_Cxx, unsafe_allow_html=True)

    # Order book
    pd_order_book_Cxx = crypto_Cxx.g_order_book(NB_DEPTH)
    best_bid_price_Cxx = pd_order_book_Cxx.iloc[0, 1]
    best_ask_price_Cxx = pd_order_book_Cxx.iloc[0, 2]

    st.write(pd_order_book_Cxx)

    # BOT

    if crypto_Cxx.g_ptf()["State"].iloc[-1] == "c":
        if (signal_model_Cxx == "Buy") & (perf_pred_Cxx >= crypto_Cxx.up_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Cxx,
                "Direction": "Buy",
                "Price": best_ask_price_Cxx,
                "Quantity": round(TAILLE_POSE / best_ask_price_Cxx, 4),
                "State": "i",
                "Amount": -TAILLE_POSE / best_ask_price_Cxx * best_ask_price_Cxx,
            }
            crypto_Cxx.update_ptf(entry)

        elif (signal_model_Cxx == "Sell") & (perf_pred_Cxx <= crypto_Cxx.down_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Cxx,
                "Direction": "Sell",
                "Price": best_bid_price_Cxx,
                "Quantity": round(TAILLE_POSE / best_bid_price_Cxx, 4),
                "State": "i",
                "Amount": TAILLE_POSE / best_bid_price_Cxx * best_bid_price_Cxx,
            }
            crypto_Cxx.update_ptf(entry)

    elif crypto_Cxx.g_ptf()["State"].iloc[-1] == "i":

        if (crypto_Cxx.g_ptf()["Direction"].iloc[-1] == "Buy") & (
            signal_model_Cxx == "Sell"
        ):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Cxx,
                "Direction": "Sell",
                "Price": best_bid_price_Cxx,
                "Quantity": crypto_Cxx.g_ptf["Quantity"].iloc[-1],
                "State": "c",
                "Amount": crypto_Cxx.g_ptf["Quantity"].iloc[-1] * best_bid_price_A,
            }

            crypto_Cxx.update_ptf(entry)

        elif (crypto_Cxx.g_ptf()["Direction"].iloc[-1] == "Sell") & (
            signal_model_Cxx == "Buy"
        ):
            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Cxx,
                "Direction": "Buy",
                "Price": best_ask_price_Cxx,
                "Quantity": crypto_Cxx.g_ptf()["Quantity"].iloc[-1],
                "State": "c",
                "Amount": -crypto_Cxx.g_ptf()["Quantity"].iloc[-1] * best_ask_price_Cxx,
            }
            crypto_Cxx.update_ptf(entry)


# ********************************************** traitement de la quatrième crypto*********************************************
with col_Dxx:

    if crypto_Dxx.g_ptf()["State"].iloc[-1] == "c":
        st.markdown(f"""## {NAME_COL_Dxx} (pas de pose)""")
    else:
        conseil = crypto_Dxx.g_ptf()["Direction"].iloc[-1]
        if conseil == "Buy":
            sens = "Long"
        else:
            sens = "Short"
        st.markdown(f"""## {NAME_COL_Dxx} (en pose {sens} )""")

    (
        live_price_Dxx,
        res_pred_Dxx,
        perf_pred_Dxx,
        signal_model_Dxx,
    ) = crypto_Dxx.get_signal()

    with st.expander("Voir portefeuille de trading"):
        st.write(crypto_Dxx.g_ptf())

    st.plotly_chart(crypto_Dxx.g_chart(), use_container_width=True)
    st.metric(
        label="Price:",
        value=f"{live_price_Dxx}",
        delta=f"{round(perf_pred_Dxx*100,2)}%",
    )

    # VALO
    st.markdown(f"""## VALO""")
    res_Dxx = crypto_Dxx.g_valo(
        live_price_Dxx,
        crypto_Dxx.g_ptf()["Direction"],
        crypto_Dxx.g_ptf()["State"],
        crypto_Dxx.g_ptf()["Quantity"],
        crypto_Dxx.g_ptf()["Amount"],
    )

    res_percent_Dxx = round(res_Dxx / TAILLE_POSE * 100, 2)
    st.write(f"P&L = {round(res_Dxx,0)} / P&L% = {res_percent_Dxx}%")

    if signal_model_Dxx == "Buy":
        new_title_Dxx = (
            '<p style="font-family:sans-serif; color:Green; font-size: 20px;">BUY</p>'
        )
        st.markdown(new_title_Dxx, unsafe_allow_html=True)
    else:
        new_title_Dxx = (
            '<p style="font-family:sans-serif; color:Red; font-size: 20px;">SELL</p>'
        )
        st.markdown(new_title_Dxx, unsafe_allow_html=True)

    # Order book
    pd_order_book_Dxx = crypto_Dxx.g_order_book(NB_DEPTH)
    best_bid_price_Dxx = pd_order_book_Dxx.iloc[0, 1]
    best_ask_price_Dxx = pd_order_book_Dxx.iloc[0, 2]

    st.write(pd_order_book_Dxx)

    # BOT

    if crypto_Dxx.g_ptf()["State"].iloc[-1] == "c":
        if (signal_model_Dxx == "Buy") & (perf_pred_Dxx >= crypto_Dxx.up_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Dxx,
                "Direction": "Buy",
                "Price": best_ask_price_Dxx,
                "Quantity": round(TAILLE_POSE / best_ask_price_Dxx, 4),
                "State": "i",
                "Amount": -TAILLE_POSE / best_ask_price_Dxx * best_ask_price_Dxx,
            }
            crypto_Dxx.update_ptf(entry)

        elif (signal_model_Dxx == "Sell") & (perf_pred_Dxx <= crypto_Dxx.down_trigger):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Dxx,
                "Direction": "Sell",
                "Price": best_bid_price_Dxx,
                "Quantity": round(TAILLE_POSE / best_bid_price_Dxx, 4),
                "State": "i",
                "Amount": TAILLE_POSE / best_bid_price_Dxx * best_bid_price_Dxx,
            }
            crypto_Dxx.update_ptf(entry)

    elif crypto_Dxx.g_ptf()["State"].iloc[-1] == "i":

        if (crypto_Dxx.g_ptf()["Direction"].iloc[-1] == "Buy") & (
            signal_model_Dxx == "Sell"
        ):

            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Dxx,
                "Direction": "Sell",
                "Price": best_bid_price_Dxx,
                "Quantity": crypto_Dxx.g_ptf["Quantity"].iloc[-1],
                "State": "c",
                "Amount": crypto_Dxx.g_ptf["Quantity"].iloc[-1] * best_bid_price_A,
            }

            crypto_Dxx.update_ptf(entry)

        elif (crypto_Dxx.g_ptf()["Direction"].iloc[-1] == "Sell") & (
            signal_model_Dxx == "Buy"
        ):
            entry = {
                "Timestamp": tb.get_current_time(),
                "Name": NAME_COL_Dxx,
                "Direction": "Buy",
                "Price": best_ask_price_Dxx,
                "Quantity": crypto_Dxx.g_ptf()["Quantity"].iloc[-1],
                "State": "c",
                "Amount": -crypto_Dxx.g_ptf()["Quantity"].iloc[-1] * best_ask_price_Dxx,
            }
            crypto_Dxx.update_ptf(entry)


# ******************************************************************************************************
time.sleep(4)
st.experimental_rerun()
