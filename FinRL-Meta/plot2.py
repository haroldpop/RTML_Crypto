import numpy as np
import math
from finrl_meta.data_processor import DataProcessor
from finrl import config
import pyfolio
import pandas as pd
from copy import deepcopy

TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet


#no changement with the function from finrl.plot 
#like backtest stat no need to make edits with we rename the columns time by date 
def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def get_baseline(ticker, start, end, time_interval):
    DP = DataProcessor('binance', start, end, time_interval)
    price_array, tech_array, turbulence_array = DP.run(ticker,
                                                    INDICATORS, 
                                                    if_vix=False, cache=True)
    df_baseline = DP.dataframe
    df_baseline.rename(columns = {'time': 'date'}, inplace=True)
    # assert ticker in TICKER_LIST, "Please select a ticker in " + TICKER_LIST
    df_baseline = df_baseline.loc[df_baseline['tic']==ticker[0]]
    return df_baseline

def backtest_plot(
        account_value,
        baseline_start=config.TRADE_START_DATE,
        baseline_end=config.TRADE_END_DATE,
        baseline_ticker="^DJI",
        value_col_name="account_value",
        time_interval="5m"
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end, time_interval=time_interval
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"])
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )
