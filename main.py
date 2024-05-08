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

name_of_encryption = None
if config.type_HE:
    name_of_encryption = 'Tenseal'
elif config.type_paillier:
    name_of_encryption = 'Paillier'
elif config.type_DP:
    name_of_encryption = 'DP'

dataset_address_1 = './datasets/boston_housing.csv'
dataset_address_2 = './datasets/heart.csv'
dataset_address_4 = './datasets/phishing.arff'

run_dataset_1 = False
run_dataset_2 = False
run_dataset_2_new = False
run_dataset_2_FE = False
run_dataset_2_compare = False
run_dataset_3 = False
run_dataset_3_new = False
run_dataset_3_compare = False
run_dataset_4 = False
new_data_2_compare = True


def create_graphs_classification(history, dataset_name, type_name):
    trace1 = go.Scatter(y=history[0], mode='lines', name='FedMod',
                        line=dict(color='blue', width=3))
    trace2 = go.Scatter(y=history[1], mode='lines', name=name_of_encryption,
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


def get_sets_of_entities(n_sets):
    party_sets = []
    server_sets = []
    main_server_sets = []

    for i in range(n_sets):
        party_list_n, server_list_n, main_server_n = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                        'binary-classification', 2, X_train, y_train)
        party_sets.append(party_list_n)
        server_sets.append(server_list_n)
        main_server_sets.append(main_server_n)

    return party_sets, server_sets, main_server_sets


def draw_graphs(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, dataset_name):
    create_graphs_classification([train_loss_list[0], train_loss_list[1], train_loss_list[2]],
                                 dataset_name=dataset_name, type_name='Train Loss')
    create_graphs_classification([train_accuracy_list[0], train_accuracy_list[1], train_accuracy_list[2]],
                                 dataset_name=dataset_name, type_name='Train Accuracy')
    create_graphs_classification([test_loss_list[0], test_loss_list[1], test_loss_list[2]],
                                 dataset_name=dataset_name, type_name='Loss')
    create_graphs_classification([test_accuracy_list[0], test_accuracy_list[1], test_accuracy_list[2]],
                                 dataset_name=dataset_name, type_name='Accuracy')


def print_results(name_list, accuracy_list, loss_list, time_list, size_transfer_list):
    round_parameter = 3

    last_accuracy_list = []
    last_loss_list = []

    for i in range(len(accuracy_list)):
        last_accuracy_list.append(accuracy_list[i][-1])

    for i in range(len(loss_list)):
        last_loss_list.append(loss_list[i][-1])

    for i in range(last_accuracy_list):
        last_accuracy_list[i] = round(last_accuracy_list[i], round_parameter)

    for i in range(last_loss_list):
        last_loss_list[i] = round(last_loss_list[i], round_parameter)

    for i in range(time_list):
        time_list[i] = round(time_list[i], round_parameter)

    for i in range(size_transfer_list):
        size_transfer_list[i] = size_transfer_list[i] / 1024
        size_transfer_list[i] = round(size_transfer_list[i], round_parameter)

    print('--------------------------------------------')
    for i in range(len(name_list)):
        print(f'{name_list[i]} Accuracy: {last_accuracy_list[i]}')
    print('--------------------------------------------')
    for i in range(len(name_list)):
        print(f'{name_list[i]} Loss: {last_loss_list[i]}')
    print('--------------------------------------------')
    for i in range(len(name_list)):
        print(f'{name_list[i]} Time: {time_list[i]} sec')
    print('--------------------------------------------')
    for i in range(len(name_list)):
        print(f'{name_list[i]} Data Transfer: {size_transfer_list[i]} KB')
    print('--------------------------------------------')

    print(f'Learning Rate: {config.learning_rate}')
    print(f'Regularization Rate: {config.regularization_rate}')
    print(f'K_value: {config.k_value}')
    print(f'N# Parties: {config.n_parties}')
    print(f'N# Servers: {config.n_servers}')
    print('--------------------------------------------')


def run_fedmod(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name):
    start_time = time.time()
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer = train_test.train_model_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, extra_results, time_taken


def run_he(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name):
    start_time = time.time()
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer = train_test.train_HE_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, extra_results, time_taken


def run_fe(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name):
    start_time = time.time()
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer = train_test.train_FE_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, extra_results, time_taken


def run_baseline(X_train, X_test, y_train, y_test, input_shape, n_epochs, dataset_name):
    start_time = time.time()
    train_accuracy, test_accuracy, train_loss, test_loss = train_test.train_mlp_binary_baseline(
        n_epochs=n_epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        input_shape=input_shape, output_shape=1, dataset_name=dataset_name)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    extra_results = [0, 0]
    time_taken = end_time - start_time

    return algorithm_results, extra_results, time_taken


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

    elif run_dataset_2_FE:
        n_epochs = 35
        dataset_name = 'heart'
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)
        party_list, server_list, main_server = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                  'binary-classification', 2, X_train, y_train)

        start_HE = time.time()
        train_loss_FE, test_loss_FE, train_accuracy_FE, test_accuracy_FE, input_shape1, size_of_HE_data_transfer = train_test.train_FE_binary_classification(
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

        create_graphs_classification([train_loss_FedMod, train_loss_FE, baseline_train_loss],
                                     dataset_name=dataset_name, type_name='Train Loss')
        create_graphs_classification([train_accuracy_FedMod, train_accuracy_FE, baseline_train_accuracy],
                                     dataset_name=dataset_name, type_name='Train Accuracy')
        create_graphs_classification([test_loss_FedMod, test_loss_FE, baseline_test_loss],
                                     dataset_name=dataset_name, type_name='Loss')
        create_graphs_classification([test_accuracy_FedMod, test_accuracy_FE, baseline_test_accuracy],
                                     dataset_name=dataset_name, type_name='Accuracy')


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

    elif new_data_2_compare:
        n_epochs = 35
        dataset_name = 'heart'
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)

        party_sets, server_sets, main_server_sets = get_sets_of_entities(n_sets=3)

        fe_results, fe_extra, fe_time = run_fe(party_sets[0], server_sets[0], main_server_sets[0],
                                               X_train, y_train, X_test, y_test, n_epochs, dataset_name)

        he_results, he_extra, he_time = run_he(party_sets[1], server_sets[1], main_server_sets[1],
                                               X_train, y_train, X_test, y_test, n_epochs, dataset_name)

        fed_results, fed_extra, fed_time = run_fedmod(party_sets[2], server_sets[2], main_server_sets[2],
                                                      X_train, y_train, X_test, y_test, n_epochs, dataset_name)

        baseline_results, baseline_extra, baseline_time = run_baseline(X_train, X_test, y_train, y_test,
                                                                       fed_extra[0], n_epochs, dataset_name)

        train_loss_list = [fe_results[0], fed_results[0], baseline_results[0]]
        train_accuracy_list = [fe_results[1], fed_results[1], baseline_results[1]]
        test_loss_list = [fe_results[2], fed_results[2], baseline_results[2]]
        test_accuracy_list = [fe_results[3], fed_results[3], baseline_results[3]]

        draw_graphs(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, dataset_name)
