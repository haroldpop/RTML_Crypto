# script to evaluate the performance of one agent on one crypto
# write a python script instead of the Jupyter Notebook will be easier


#importation of modules
import numpy as np
import math
import gym
from finrl_meta.env_crypto_trading.env_multiple_crypto import CryptoEnv
from finrl.plot import backtest_stats
from finrl import config
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.elegantrl_models import DRLAgent as DRLAgent_erl

import pandas as pd
from plot2 import get_baseline, get_daily_return, backtest_plot

from finrl_meta.data_processor import DataProcessor

# ignore pandas warning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from test import test
from train import train
from cryptoenv import CryptoEnv


TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
TICKER = [TICKER_LIST[0]]
print(TICKER)

TRAIN_START_DATE = '2021-09-01'
TRAIN_END_DATE = '2021-09-02'

TEST_START_DATE = '2021-09-21'
TEST_END_DATE = '2021-09-30'
time_interval = '5m'

env = CryptoEnv


INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {"learning_rate": 2**-15,"batch_size": 2**11,
                "gamma": 0.99, "seed":312,"net_dimension": 2**9, 
                "target_step": 5000, "eval_gap": 30, "eval_times": 1}

RLLIB_PARAMS = {"lr": 2**-15,"train_batch_size": 2**11,
                "gamma": 0.99, "num_workers": 0,
                "horizon": 1e10, 
               "disable_env_checkin": True}


# training of the agent

print('----------Training-----------------')

# train(start_date=TRAIN_START_DATE, 
#        end_date=TRAIN_END_DATE,
#        ticker_list=TICKER,    #only one ticker for the moment (deal with get_baseline for multiple ticker)
#        data_source='binance',
#        time_interval=time_interval, 
#        technical_indicator_list=INDICATORS,
#        drl_lib='rllib', 
#        env=env, 
#        model_name='ppo', 
#        cwd='./test_rllib_ppo',
#        rllib_params=RLLIB_PARAMS,
#        break_step=5e4,
#        if_vix=False
#        )

print('----------End Training-----------------')

#testing of the agent

print('----------- Testing -----------------')


#with rllib have to check the path sometimes, have to fix it to make it work properly

account_value_erl = test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = TICKER, 
                        data_source = 'binance',
                        time_interval= time_interval, 
                        technical_indicator_list= INDICATORS,
                        drl_lib='rllib', 
                        env=env, 
                        model_name='ppo', 
                        cwd='./test_rllib_ppo/checkpoint_000100/checkpoint-100', 
                        net_dimension = 2**9, 
                        if_vix=False
                        )

print('--------- End Testing ------------------')


baseline_df = get_baseline(TICKER, TEST_START_DATE, TEST_END_DATE, time_interval)
account_value_erl_pd = pd.DataFrame({'date':baseline_df.date,'account_value':account_value_erl})
assert account_value_erl_pd.shape[0] == baseline_df.shape[0]

backtest_plot(account_value_erl_pd, 
             baseline_ticker = TICKER, 
             baseline_start = TEST_START_DATE,
             baseline_end = TEST_END_DATE)
