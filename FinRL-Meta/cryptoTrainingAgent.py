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
from cryptoenv import CryptoEnv2

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

TRAIN_START_DATE = '2021-10-01'
TRAIN_END_DATE = '2022-02-26'

TEST_START_DATE = '2022-03-01'
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
               "disable_env_checkin": True}

STABLE_PARAMS = {'total_timesteps':5e6,
                    'agent_params': None}


# training of the agent

algorithm_stables_rllib = ['ppo', 'ddpg', 'td3', 'sac', 'a2c']
algorithm_elegantrl = ['ppo', 'ddpg', 'td3', 'sac']

libraries = ['elegantrl', 'rllib', 'stable_baselines3']
libraries = ['rllib', 'stable_baselines3']

for TICKER in TICKER_LIST[:1]: #only use the BTC for the moment
    TICKER = [TICKER]
    for library in libraries:
        if library == 'elegantrl':
            algos = algorithm_elegantrl
        else:
            algos = algorithm_stables_rllib
        for algo in algos:
            print('----------Training and Testing for ' + str(TICKER[0]) + ' ' + library + ' ' + algo + ' -----------------')
            print(TICKER[0], library, algo)

            if library == 'elegantrl':
                model_name = algo
                cwd = './test_' + algo + library + str(TICKER[0])
                train(start_date=TRAIN_START_DATE, 
                    end_date=TRAIN_END_DATE,
                    ticker_list=TICKER,    #only one ticker for the moment (deal with get_baseline for multiple ticker)
                    data_source='binance',
                    time_interval=time_interval, 
                    technical_indicator_list=INDICATORS,
                    drl_lib=library, 
                    env=env, 
                    model_name=algo, 
                    cwd=cwd,
                    erl_params=ERL_PARAMS,
                    break_step=5e4,
                        if_vix=False
                        )

                #with rllib have to check the path sometimes, have to fix it to make it work properly

                account_value_erl = test(start_date = TEST_START_DATE, 
                                        end_date = TEST_END_DATE,
                                        ticker_list = TICKER, 
                                        data_source = 'binance',
                                        time_interval= time_interval, 
                                        technical_indicator_list= INDICATORS,
                                        drl_lib=library, 
                                        env=env, 
                                        model_name=model_name, 
                                        cwd=cwd, 
                                        net_dimension = 2**9, 
                                        if_vix=False
                                        )

            elif library == 'rllib':
                model_name = algo
                cwd = './test_' + algo + library + str(TICKER[0])

                train(start_date=TRAIN_START_DATE, 
                    end_date=TRAIN_END_DATE,
                    ticker_list=TICKER,    #only one ticker for the moment (deal with get_baseline for multiple ticker)
                    data_source='binance',
                    time_interval=time_interval, 
                    technical_indicator_list=INDICATORS,
                    drl_lib=library, 
                    env=env, 
                    model_name=algo, 
                    cwd=cwd,
                    rllib_params=RLLIB_PARAMS,
                    break_step=5e4,
                        if_vix=False
                        )

                #testing of the agent


                #with rllib have to check the path sometimes, have to fix it to make it work properly
                
                account_value_erl = test(start_date = TEST_START_DATE, 
                                        end_date = TEST_END_DATE,
                                        ticker_list = TICKER, 
                                        data_source = 'binance',
                                        time_interval= time_interval, 
                                        technical_indicator_list= INDICATORS,
                                        drl_lib=library, 
                                        env=env, 
                                        model_name=model_name, 
                                        cwd=cwd + '/checkpoint_000100/checkpoint-100', 
                                        net_dimension = 2**9, 
                                        if_vix=False
                                        )

            elif library == 'stable_baselines3':
                model_name = algo
                cwd = './test_' + algo + library + str(TICKER[0])

                train(start_date=TRAIN_START_DATE, 
                    end_date=TRAIN_END_DATE,
                    ticker_list=TICKER,    #only one ticker for the moment (deal with get_baseline for multiple ticker)
                    data_source='binance',
                    time_interval=time_interval, 
                    technical_indicator_list=INDICATORS,
                    drl_lib=library, 
                    env=env, 
                    model_name=algo, 
                    cwd=cwd,
                    total_timesteps = 5e4,
                        if_vix=False
                        )

                #testing of the agent

                #with rllib have to check the path sometimes, have to fix it to make it work properly
                
                account_value_erl = test(start_date = TEST_START_DATE, 
                                        end_date = TEST_END_DATE,
                                        ticker_list = TICKER, 
                                        data_source = 'binance',
                                        time_interval= time_interval, 
                                        technical_indicator_list= INDICATORS,
                                        drl_lib=library, 
                                        env=env, 
                                        model_name=model_name, 
                                        cwd=cwd, 
                                        net_dimension = 2**9, 
                                        if_vix=False
                                        )
            print('----------End Training and Testing for ' + str(TICKER[0]) + ' ' + library + ' ' + algo + ' -----------------')
            

