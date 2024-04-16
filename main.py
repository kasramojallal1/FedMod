import client
import server
import main_server
import config
import preprocess
import nodes
import train_test

import math
import os
import time
import numpy as np
import pandas as pd
import sympy
import random
import secrets
import resource

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

pio.renderers.default = "browser"
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(43)
tf.random.set_seed(43)
random.seed(45)

if __name__ == "__main__":
    dataset_address_1 = './datasets/boston_housing.csv'
    dataset_address_2 = './datasets/heart.csv'
    dataset_address_4 = './datasets/PhishingData.arff'

    # X_train, X_test, y_train, y_test = preprocess.setup_dataframe_1(dataset_address_1)
    # party1, party2, server1, server2, main_server1 = nodes.create_nodes_1(X_train, y_train)
    # train_test.train_parties_1(10, [party1, party2], [server1, server2], main_server1, X_test, y_test)

    # X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)
    # party1, party2, server1, server2, main_server1 = nodes.create_nodes_2(X_train, y_train)
    # run_time, cpu_usage, memory_usage = train_test.train_parties_2(30, [party1, party2], [server1, server2], main_server1, X_train, y_train, X_test, y_test)

    # X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()
    # party1, party2, server1, server2, main_server1 = nodes.create_nodes_3(X_train, y_train, [17, 17])
    # run_time, cpu_usage, memory_usage = train_test.train_parties_3(500, party1, party2, server1, server2, main_server1, X_train, y_train, X_test,y_test)

    X_train, X_test, y_train, y_test = preprocess.setup_dataframe_4(dataset_address_4)
    party1, party2, server1, server2, main_server1 = nodes.create_nodes_4(X_train, y_train, 2, 'multi-classification', number_of_classes=3)
    run_time, cpu_usage, memory_usage = train_test.train_parties_4(100, [party1, party2], [server1, server2], main_server1, X_train, y_train, X_test, y_test, n_classes=3)


    print("------ FedMod ------")
    print("--- FedMod: %s secs ---" % (run_time))
    print(f"Memory Usage: {memory_usage} KB")
    print(f"CPU Time Used: {cpu_usage} seconds")
