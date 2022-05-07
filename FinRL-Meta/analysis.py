import matplotlib.pyplot as plt

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
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from plot2 import get_baseline, get_daily_return, backtest_plot

from finrl_meta.data_processor import DataProcessor

# ignore pandas warning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from test import test
from train import train
from cryptoenv import CryptoEnv, CryptoEnv2

#hyperparameters
TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
TICKER = [TICKER_LIST[0]]
print(TICKER)

TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2022-03-19'

TEST_START_DATE = '2022-03-20'
TEST_END_DATE = '2022-04-01'
time_interval = '5m'

env = CryptoEnv2

INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {"learning_rate": 2**-15,"batch_size": 2**11,
                "gamma": 0.99, "seed":312,"net_dimension": 2**9, 
                "target_step": 5000, "eval_gap": 30, "eval_times": 1}

RLLIB_PARAMS = {"lr": 2**-15,"train_batch_size": 2**11,
                "gamma": 0.99, "num_workers": 0,
                "horizon": 1e10, 
               "disable_env_checking": True}

print('----------Training-----------------')

# train(start_date=TRAIN_START_DATE, 
#        end_date=TRAIN_END_DATE,
#        ticker_list=TICKER,    #only one ticker for the moment (deal with get_baseline for multiple ticker)
#        data_source='binance',
#        time_interval=time_interval, 
#        technical_indicator_list=INDICATORS,
#        drl_lib='rllib', 
#        env=env, 
#        model_name='a2c', 
#        cwd='./test_a2c_rllib',
#        rllib_params=RLLIB_PARAMS,
#        break_step=5e4,
#        if_vix=False
#        )

# train(start_date=TRAIN_START_DATE, 
#        end_date=TRAIN_END_DATE,
#        ticker_list=TICKER,    #only one ticker for the moment (deal with get_baseline for multiple ticker)
#        data_source='binance',
#        time_interval=time_interval, 
#        technical_indicator_list=INDICATORS,
#        drl_lib='elegantrl', 
#        env=env, 
#        model_name='ppo', 
#        cwd='./draft_ppo',
#        erl_params=ERL_PARAMS,
#        break_step=5e4,
#        if_vix=False
#        )

print('----------End Training-----------------')

#testing of the agent

print('----------- Testing -----------------')


#with rllib have to check the path sometimes, have to fix it to make it work properly

# account_value_erl, action_value_erl = test(start_date = TEST_START_DATE, 
#                         end_date = TEST_END_DATE,
#                         ticker_list = TICKER, 
#                         data_source = 'binance',
#                         time_interval= time_interval, 
#                         technical_indicator_list= INDICATORS,
#                         drl_lib='elegantrl', 
#                         env=env, 
#                         model_name='ppo', 
#                         cwd='./draft_ppo', 
#                         net_dimension = 2**9, 
#                         if_vix=False
#                         )


account_value_erl, action_value_erl = test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = TICKER, 
                        data_source = 'binance',
                        time_interval= time_interval, 
                        technical_indicator_list= INDICATORS,
                        drl_lib='rllib', 
                        env=env, 
                        model_name='a2c', 
                        cwd='./test_a2c_rllib/checkpoint_000100/checkpoint-100', 
                        net_dimension = 2**9, 
                        if_vix=False
                        )




print('--------- End Testing ------------------')

account_value_erl = np.array(account_value_erl).reshape(-1, 1)
action_value_erl = np.array(action_value_erl).reshape(-1, )
print(action_value_erl.shape, account_value_erl.shape)
# print(action_value_erl)

def plot_action_market(action_value, start_date, end_date, time_interval):
   data_source = 'binance'
   start_date = TEST_START_DATE
   end_date = TEST_END_DATE
   time_interval = time_interval
   technical_indicator_list= INDICATORS
   if_vix=False
   ticker_list = TICKER

   dp = DataProcessor(data_source, start_date, end_date, time_interval)
   price_array, tech_array, turbulence_array = dp.run(ticker_list, 
                                                      technical_indicator_list,
                                                         if_vix)
   actions_p = list()
   for i in range(action_value.shape[0]):
      if action_value[i] > 0 :
         actions_p.append("Buy")
      elif action_value[i] < 0 :
         actions_p.append("Sell")
      else:
         actions_p.append("No position")
   sell_actions = np.where(action_value < 0)[0]
   buy_actions = np.where(action_value > 0)[0]
   no_actions = np.where(action_value ==0)[0]
   x = np.arange(price_array.shape[0])

   d = {'idx': x, 'price': price_array.reshape(-1, ), 'position': action_value}
   df = pd.DataFrame(d)
   
   fig = px.scatter(df, x='idx', y='price', color='position',
                        title = 'Position of the agents according to the price of the Bitcoin')

   fig.show()

plot_action_market(action_value_erl, TEST_START_DATE, TEST_END_DATE, time_interval)
