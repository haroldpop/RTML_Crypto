# script to evaluate the performance of one agent on one crypto
# write a python script instead of the Jupyter Notebook will be easier


#importation of modules
import numpy as np
import math
import torch
import gym
from finrl.plot import backtest_stats
from finrl import config
import pandas as pd
from plot2 import get_baseline, get_daily_return, backtest_plot
from finrl_meta.data_processor import DataProcessor
from test import test
from train import train
from cryptoenv import CryptoEnv3

# ignore pandas warning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
TICKER = [TICKER_LIST[0]]
print(TICKER)

TRAIN_START_DATE = '2021-09-01'
TRAIN_END_DATE = '2021-09-10'
TEST_START_DATE = '2021-09-15'
TEST_END_DATE = '2021-09-16'
time_interval = '5m'

INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

# intialization of the agent 

from custom_agents.custom_ppo.agent import Agent
from custom_agents.custom_ppo.trainer import Train

data_source = 'binance'

dp = DataProcessor(data_source, TRAIN_START_DATE, TRAIN_END_DATE, time_interval)
price_array, tech_array, turbulence_array = dp.run(TICKER, 
                                                    INDICATORS,
                                                    if_vix=False)
print(dp.dataframe.shape)
print(price_array.shape, tech_array.shape, turbulence_array.shape)
config = {"price_array": price_array, "tech_array": tech_array, "turbulence_array": turbulence_array}
env = CryptoEnv3(config = config, lookback=10)
env_name = "CryptoEnv"

# dp = DataProcessor(data_source, TEST_START_DATE, TEST_END_DATE, time_interval)
price_array, tech_array, turbulence_array = dp.run(TICKER, 
                                                   INDICATORS,
                                                   if_vix=False)
config = {"price_array": price_array, "tech_array": tech_array, "turbulence_array": turbulence_array}
test_env = CryptoEnv3(config = config, lookback=10)

assert test_env.lookback == env.lookback

n_iter = 1000
n_states = 2 + tech_array.shape[1]*env.lookback #cash, stock and technical indicators acroos the lookback previous days
n_actions = price_array.shape[1] #only one action per iteration the trade of the crypto
lr = 1e-5
num_epochs = 100
mini_batch_size=64
clip_range = 0.2

print(n_states)

agent = Agent(env_name = env_name, 
                n_iter=n_iter, 
                n_states=n_states, 
                n_actions=n_actions, 
                lr=lr)


state, reward, done, info = env.step(torch.Tensor(1).float())


TRAIN_FLAG = True
if TRAIN_FLAG:
    print('----------Training-----------------')
    trainer = Train(env=env, 
                    env_name=env_name,
                   test_env=test_env,
                    n_iterations=int(n_iter),
                    agent=agent,
                    epochs=num_epochs,
                    mini_batch_size=mini_batch_size,
                    epsilon=clip_range,
                    horizon=100,
                    cwd='./custom_ppo' )
    trainer.step()
    print('----------End Training-----------------')




# baseline_df = get_baseline(TICKER, TEST_START_DATE, TEST_END_DATE, time_interval)
# account_value_erl_pd = pd.DataFrame({'date':baseline_df.date,'account_value':account_value_erl})
# assert account_value_erl_pd.shape[0] == baseline_df.shape[0]

# backtest_plot(account_value_erl_pd, 
#              baseline_ticker = TICKER, 
#              baseline_start = TEST_START_DATE,
#              baseline_end = TEST_END_DATE)
