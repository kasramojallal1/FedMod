import preprocess
import nodes
import train_test

import os
import numpy as np
import pandas as pd
import random
import plotly.io as pio
import tensorflow as tf

pio.renderers.default = "browser"
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(43)
tf.random.set_seed(43)
random.seed(45)

dataset_address_1 = './datasets/boston_housing.csv'
dataset_address_2 = './datasets/heart.csv'
dataset_address_4 = './datasets/phishing.arff'

run_dataset_1 = False
run_dataset_2 = False
run_dataset_3 = True
run_dataset_4 = False

if __name__ == "__main__":
    if run_dataset_1:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_1(dataset_address_1)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_1(X_train, y_train)
        train_test.train_parties_1(10, [party1, party2], [server1, server2], main_server1, X_test, y_test)

    elif run_dataset_2:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_2(X_train, y_train)
        run_time, cpu_usage, memory_usage = train_test.train_parties_2(30, [party1, party2], [server1, server2], main_server1, X_train, y_train, X_test, y_test)

    elif run_dataset_3:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_3(X_train, y_train, [17, 17])
        run_time, cpu_usage, memory_usage = train_test.train_parties_3(30, party1, party2, server1, server2, main_server1, X_train, y_train, X_test,y_test)

    elif run_dataset_4:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_4(dataset_address_4)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_4(X_train, y_train, 2, 'multi-classification', number_of_classes=3)
        run_time, cpu_usage, memory_usage = train_test.train_parties_4(30, [party1, party2], [server1, server2], main_server1, X_train, y_train, X_test, y_test, n_classes=3)

    if run_dataset_1 or run_dataset_2 or run_dataset_3 or run_dataset_4:
        print("------ FedMod ------")
        print("--- FedMod: %s secs ---" % (run_time))
        print(f"Memory Usage: {memory_usage} KB")
        print(f"CPU Time Used: {cpu_usage} seconds")
