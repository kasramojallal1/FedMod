import preprocess
import nodes
import train_test
import config

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
run_dataset_2_new = False
run_dataset_2_HE = True
run_dataset_3 = False
run_dataset_3_new = False
run_dataset_3_HE = False
run_dataset_4 = False

if __name__ == "__main__":
    if run_dataset_1:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_1(dataset_address_1)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_1(X_train, y_train)
        train_test.train_parties_1(10, [party1, party2], [server1, server2], main_server1, X_test, y_test)

    elif run_dataset_2:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_2(X_train, y_train)
        run_time, cpu_usage, memory_usage = train_test.train_parties_2(30, [party1, party2], [server1, server2],
                                                                       main_server1, X_train, y_train, X_test, y_test)

    elif run_dataset_3:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_3(X_train, y_train, [17, 17])
        run_time, cpu_usage, memory_usage = train_test.train_parties_3(30, party1, party2, server1, server2,
                                                                       main_server1, X_train, y_train, X_test, y_test)

    elif run_dataset_4:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_4(dataset_address_4)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_4(X_train, y_train, 2,
                                                                              'multi-classification',
                                                                              number_of_classes=3)
        run_time, cpu_usage, memory_usage = train_test.train_parties_4(50, [party1, party2], [server1, server2],
                                                                       main_server1, X_train, y_train, X_test, y_test,
                                                                       n_classes=3)

    elif run_dataset_2_new:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)
        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        train_test.train_model_binary_classification(dataset_name='heart', n_epochs=40, party_list=party_list,
                                                     server_list=server_list, main_server=main_server, X_train=X_train,
                                                     y_train=y_train, X_test=X_test, y_test=y_test)

    elif run_dataset_3_new:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()
        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        train_test.train_model_binary_classification(dataset_name='ionosphere', n_epochs=360, party_list=party_list,
                                                     server_list=server_list, main_server=main_server, X_train=X_train,
                                                     y_train=y_train, X_test=X_test, y_test=y_test)

    elif run_dataset_2_HE:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)
        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        train_loss_HE, test_loss_HE, train_accuracy_HE, test_accuracy_HE = train_test.train_HE_binary_classification(
            dataset_name='heart', n_epochs=40, party_list=party_list,
            server_list=server_list, main_server=main_server, X_train=X_train,
            y_train=y_train, X_test=X_test, y_test=y_test)
        train_loss_FedMod, test_loss_FedMod, train_accuracy_FedMod, test_accuracy_FedMod = train_test.train_model_binary_classification(
            dataset_name='heart', n_epochs=40, party_list=party_list,
            server_list=server_list, main_server=main_server, X_train=X_train,
            y_train=y_train, X_test=X_test, y_test=y_test)


    elif run_dataset_3_HE:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()
        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        train_test.train_HE_binary_classification(dataset_name='ionosphere', n_epochs=360, party_list=party_list,
                                                  server_list=server_list, main_server=main_server, X_train=X_train,
                                                  y_train=y_train, X_test=X_test, y_test=y_test)

    # if run_dataset_1 or run_dataset_2 or run_dataset_3 or run_dataset_4:
    #     print("------ FedMod ------")
    #     print("--- FedMod: %s secs ---" % (run_time))
    #     print(f"Memory Usage: {memory_usage} KB")
    #     print(f"CPU Time Used: {cpu_usage} seconds")

    print('--------------------------------------------------')
    print(f'Learning Rate: {config.learning_rate}')
    print(f'Regularization Rate: {config.regularization_rate}')
    print(f'K_value: {config.k_value}')
    print(f'N# Parties: {config.n_parties}')
    print(f'N# Servers: {config.n_servers}')
