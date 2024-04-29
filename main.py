import preprocess
import nodes
import train_test
import config

import os
import time
import numpy as np
import pandas as pd
import random
import plotly.io as pio
import tensorflow as tf

import plotly.express as px
import plotly.graph_objects as go

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
run_dataset_2_compare = True
run_dataset_3 = False
run_dataset_3_new = False
run_dataset_3_compare = False
run_dataset_4 = False


def create_graphs_classification(history, dataset_name, type_name):
    trace1 = go.Scatter(y=history[0], mode='lines', name='FedMod',
                        line=dict(color='blue', width=3))
    trace2 = go.Scatter(y=history[1], mode='lines', name='HE',
                        line=dict(color='red', width=3))
    trace3 = go.Scatter(y=history[2], mode='lines', name='Baseline',
                        line=dict(color='purple', width=3))

    markers_interval = config.plot_intervals
    trace1_markers = train_test.add_markers_at_intervals(history[0], 'diamond', 'blue',
                                                         markers_interval)
    trace2_markers = train_test.add_markers_at_intervals(history[1], 'diamond', 'red',
                                                         markers_interval)
    trace3_markers = train_test.add_markers_at_intervals(history[2], 'diamond', 'purple',
                                                         markers_interval)

    layout = go.Layout(title=f'Dataset: {dataset_name}',
                       xaxis=dict(title='Epochs',
                                  tickvals=list(range(0, len(history[0]) + 1, markers_interval)),
                                  showgrid=True,
                                  zeroline=False),
                       yaxis=dict(title=type_name),
                       legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                       plot_bgcolor='rgba(242, 242, 242, 1)')

    fig1 = go.Figure(
        data=[trace1, trace2, trace3, trace1_markers, trace2_markers, trace3_markers],
        layout=layout)
    fig1.show()


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

    elif run_dataset_2_compare:
        n_epochs = 35
        dataset_name = 'heart'
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)

        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        start_HE = time.time()
        train_loss_HE, test_loss_HE, train_accuracy_HE, test_accuracy_HE, input_shape1, size_of_HE_data_transfer = train_test.train_HE_binary_classification(
            dataset_name=dataset_name, n_epochs=n_epochs, party_list=party_list,
            server_list=server_list, main_server=main_server, X_train=X_train,
            y_train=y_train, X_test=X_test, y_test=y_test)
        end_HE = time.time()

        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        start_FedMod = time.time()
        train_loss_FedMod, test_loss_FedMod, train_accuracy_FedMod, test_accuracy_FedMod, input_shape2, size_of_FedMod_data_transfer = train_test.train_model_binary_classification(
            dataset_name=dataset_name, n_epochs=n_epochs, party_list=party_list,
            server_list=server_list, main_server=main_server, X_train=X_train,
            y_train=y_train, X_test=X_test, y_test=y_test)
        end_FedMod = time.time()

        start_baseline = time.time()
        baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss = train_test.train_mlp_binary_baseline(
            n_epochs=n_epochs, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, input_shape=input_shape1, output_shape=1,
            dataset_name=dataset_name)
        end_baseline = time.time()

        create_graphs_classification([train_loss_FedMod, train_loss_HE, baseline_train_loss],
                                     dataset_name=dataset_name, type_name='Train Loss')
        create_graphs_classification([train_accuracy_FedMod, train_accuracy_HE, baseline_train_accuracy],
                                     dataset_name=dataset_name, type_name='Train Accuracy')
        create_graphs_classification([test_loss_FedMod, test_loss_HE, baseline_test_loss],
                                     dataset_name=dataset_name, type_name='Loss')
        create_graphs_classification([test_accuracy_FedMod, test_accuracy_HE, baseline_test_accuracy],
                                     dataset_name=dataset_name, type_name='Accuracy')

    elif run_dataset_3_compare:
        n_epochs = 360
        dataset_name = 'ionosphere'
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()

        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        start_HE = time.time()
        train_loss_HE, test_loss_HE, train_accuracy_HE, test_accuracy_HE, input_shape1 = train_test.train_HE_binary_classification(
            dataset_name=dataset_name, n_epochs=n_epochs, party_list=party_list,
            server_list=server_list, main_server=main_server, X_train=X_train,
            y_train=y_train, X_test=X_test, y_test=y_test)
        end_HE = time.time()

        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)
        start_FedMod = time.time()
        train_loss_FedMod, test_loss_FedMod, train_accuracy_FedMod, test_accuracy_FedMod, input_shape2 = train_test.train_model_binary_classification(
            dataset_name=dataset_name, n_epochs=n_epochs, party_list=party_list,
            server_list=server_list, main_server=main_server, X_train=X_train,
            y_train=y_train, X_test=X_test, y_test=y_test)
        end_FedMod = time.time()

        start_baseline = time.time()
        baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss = train_test.train_mlp_binary_baseline(
            n_epochs=n_epochs, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, input_shape=input_shape1, output_shape=1,
            dataset_name=dataset_name)
        end_baseline = time.time()

        create_graphs_classification([train_loss_FedMod, train_loss_HE, baseline_train_loss],
                                     dataset_name=dataset_name, type_name='Train Loss')
        create_graphs_classification([train_accuracy_FedMod, train_accuracy_HE, baseline_train_accuracy],
                                     dataset_name=dataset_name, type_name='Train Accuracy')
        create_graphs_classification([test_loss_FedMod, test_loss_HE, baseline_test_loss],
                                     dataset_name=dataset_name, type_name='Loss')
        create_graphs_classification([test_accuracy_FedMod, test_accuracy_HE, baseline_test_accuracy],
                                     dataset_name=dataset_name, type_name='Accuracy')


    size_of_HE_data_transfer = size_of_HE_data_transfer / 1024
    size_of_FedMod_data_transfer = size_of_FedMod_data_transfer / 1024


    print('--------------------------------------------')
    print(f"FedMod: {round((end_FedMod - start_FedMod), 3)} secs")
    print(f"HE: {round((end_HE - start_HE), 3)} secs")
    print(f"Baseline: {round((end_baseline - start_baseline), 3)} secs")
    print('--------------------------------------------')
    print(f'FedMod Accuracy: {round((test_accuracy_FedMod[-1]), 3)}')
    print(f'HE Accuracy: {round((test_accuracy_HE[-1]), 3)}')
    print(f'Baseline Accuracy: {round((baseline_test_accuracy[-1]), 3)}')
    print('--------------------------------------------')
    print(f'Size of FE Data Transfer: {round(size_of_HE_data_transfer, 3)} KB')
    print(f'Size of FedMod Data Transfer: {round(size_of_FedMod_data_transfer, 3)} KB')
    print('--------------------------------------------')
    print(f'Learning Rate: {config.learning_rate}')
    print(f'Regularization Rate: {config.regularization_rate}')
    print(f'K_value: {config.k_value}')
    print(f'N# Parties: {config.n_parties}')
    print(f'N# Servers: {config.n_servers}')
    # print(f'Poly Modulus Degree: {config.poly_mod_degree}')
    # print(f'Coeff Mod Bit Sizes: {config.coeff_mod_bit_sizes}')
    # print(f'Context Global Scale: {config.context.global_scale}')
    print('--------------------------------------------')
