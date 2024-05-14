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
run_dataset_2_FE = False
run_dataset_2_compare = False
run_dataset_3 = False
run_dataset_3_new = False
run_dataset_3_compare = False
run_dataset_4 = False
dataset_2_comparison = True


def create_graphs_classification(history, dataset_name, type_name, name_list, file_path):
    file_path = file_path + f'-{type_name}.png'

    color_list = ['red', 'blue', 'purple', 'green', 'orange', 'yellow']
    trace_list = []
    for i in range(len(history)):
        trace_list.append(go.Scatter(y=history[i], mode='lines', name=f'{name_list[i]}',
                                     line=dict(color=f'{color_list[i]}', width=3)))

    layout = go.Layout(title=f'Dataset: {dataset_name}',
                       xaxis=dict(title='Epochs',
                                  showgrid=True,
                                  zeroline=False),
                       yaxis=dict(title=type_name),
                       legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                       plot_bgcolor='rgba(242, 242, 242, 1)')

    fig1 = go.Figure(data=trace_list, layout=layout)
    fig1.write_image(file_path, width=1920, height=1080, scale=2)
    # fig1.show()


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


def draw_graphs(train_loss_list, train_accuracy_list,
                test_loss_list, test_accuracy_list,
                test_precision_list, test_recall_list,
                dataset_name, name_list, file_path):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []

    for i in range(len(train_loss_list)):
        train_losses.append(train_loss_list[i])
        train_accuracies.append(train_accuracy_list[i])
        test_losses.append(test_loss_list[i])
        test_accuracies.append(test_accuracy_list[i])
        test_precisions.append(test_precision_list[i])
        test_recalls.append(test_recall_list[i])

    # create_graphs_classification(history=train_losses, dataset_name=dataset_name, type_name='Train Loss')
    # create_graphs_classification(history=train_accuracies, dataset_name=dataset_name, type_name='Train Accuracy')
    # create_graphs_classification(history=test_losses, dataset_name=dataset_name, type_name='Loss')
    create_graphs_classification(history=test_accuracies, dataset_name=dataset_name, type_name='Accuracy',
                                 name_list=name_list, file_path=file_path)
    create_graphs_classification(history=test_precisions, dataset_name=dataset_name, type_name='Precision',
                                 name_list=name_list, file_path=file_path)
    create_graphs_classification(history=test_recalls, dataset_name=dataset_name, type_name='Recall',
                                 name_list=name_list, file_path=file_path)


def print_results(name_list, accuracy_list, loss_list, precision_list, recall_list, time_list, size_transfer_list,
                  file_path):
    file_path = file_path + '.txt'

    round_parameter = 4

    last_accuracies = []
    for i in range(len(name_list)):
        last_accuracies.append(round(accuracy_list[i][-1], round_parameter))

    last_losses = []
    for i in range(len(name_list)):
        last_losses.append(round(loss_list[i][-1], round_parameter))

    last_precisions = []
    for i in range(len(name_list)):
        last_precisions.append(round(precision_list[i][-1], round_parameter))

    last_recalls = []
    for i in range(len(name_list)):
        last_recalls.append(round(recall_list[i][-1], round_parameter))

    last_f1s = []
    for i in range(len(name_list)):
        temp_f1 = 2 * (last_precisions[i] * last_recalls[i]) / ((last_precisions[i] + last_recalls[i]) + 1e-10)
        last_f1s.append(round(temp_f1, round_parameter))

    for i in range(len(name_list)):
        time_list[i] = round(time_list[i], round_parameter)

    for i in range(len(name_list)):
        size_transfer_list[i] = size_transfer_list[i] / 1024
        size_transfer_list[i] = round(size_transfer_list[i], round_parameter)

    with open(file_path, 'w') as file:
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} Accuracy: {last_accuracies[i]}\n')
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} Loss: {last_losses[i]}\n')
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} Precision: {last_precisions[i]}\n')
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} Recall: {last_recalls[i]}\n')
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} F1: {last_f1s[i]}\n')
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} Time: {time_list[i]} secs\n')
        file.write('--------------------------------------------\n')
        for i in range(len(name_list)):
            file.write(f'{name_list[i]} Data Transfer: {size_transfer_list[i]} KB\n')
        file.write('--------------------------------------------\n')

        file.write(f'Dataset Name: {dataset_name}\n')
        file.write(f'Learning Rate: {config.learning_rate}\n')
        file.write(f'Regularization Rate: {config.regularization_rate}\n')
        file.write(f'K_value: {config.k_value}\n')
        file.write(f'N# Parties: {config.n_parties}\n')
        file.write(f'N# Servers: {config.n_servers}\n')
        file.write('--------------------------------------------\n')


def run_fedmod(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name):
    start_time = time.time()
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer, test_precision, test_recall = train_test.train_model_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    algorithm_scores = [test_precision, test_recall]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, algorithm_scores, extra_results, time_taken


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
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer, test_precision, test_recall = train_test.train_FE_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    algorithm_scores = [test_precision, test_recall]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, algorithm_scores, extra_results, time_taken


def run_baseline(X_train, X_test, y_train, y_test, input_shape, n_epochs, dataset_name):
    start_time = time.time()
    train_accuracy, test_accuracy, train_loss, test_loss, test_precision, test_recall = train_test.train_mlp_binary_baseline(
        n_epochs=n_epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        input_shape=input_shape, output_shape=1, dataset_name=dataset_name)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    algorithm_scores = [test_precision, test_recall]
    extra_results = [0, 0]
    time_taken = end_time - start_time

    return algorithm_results, algorithm_scores, extra_results, time_taken


if __name__ == "__main__":
    if run_dataset_1:
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_1(dataset_address_1)
        party1, party2, server1, server2, main_server1 = nodes.create_nodes_1(X_train, y_train)
        train_test.train_parties_1(10, [party1, party2], [server1, server2], main_server1, X_test, y_test)

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

    elif dataset_2_comparison:

        for i in range(4, 13):
            config.n_parties = i
            n_epochs = 35
            dataset_name = 'heart'
            X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)

            party_sets, server_sets, main_server_sets = get_sets_of_entities(n_sets=3)

            fe_results, fe_scores, fe_extra, fe_time = run_fe(party_sets[0], server_sets[0], main_server_sets[0],
                                                              X_train, y_train, X_test, y_test, n_epochs, dataset_name)

            # he_results, he_extra, he_time = run_he(party_sets[1], server_sets[1], main_server_sets[1],
            #                                        X_train, y_train, X_test, y_test, n_epochs, dataset_name)

            fed_results, fed_scores, fed_extra, fed_time = run_fedmod(party_sets[2],
                                                                      server_sets[2],
                                                                      main_server_sets[2],
                                                                      X_train, y_train, X_test, y_test,
                                                                      n_epochs, dataset_name)

            baseline_results, baseline_scores, baseline_extra, baseline_time = run_baseline(X_train, X_test,
                                                                                            y_train, y_test,
                                                                                            fed_extra[0],
                                                                                            n_epochs, dataset_name)

            train_loss_list = [fe_results[0], fed_results[0], baseline_results[0]]
            train_accuracy_list = [fe_results[1], fed_results[1], baseline_results[1]]
            test_loss_list = [fe_results[2], fed_results[2], baseline_results[2]]
            test_accuracy_list = [fe_results[3], fed_results[3], baseline_results[3]]
            test_precision_list = [fe_scores[0], fed_scores[0], baseline_scores[0]]
            test_recall_list = [fe_scores[1], fed_scores[1], baseline_scores[1]]

            name_list = ['FedV', 'FedMod', 'Baseline']
            file_write_path = f"results/p-{config.n_parties}"
            draw_graphs(train_loss_list,
                        train_accuracy_list,
                        test_loss_list,
                        test_accuracy_list,
                        test_precision_list,
                        test_recall_list,
                        dataset_name,
                        name_list,
                        file_write_path)

            accuracy_list = [fe_results[3], fed_results[3], baseline_results[3]]
            precision_list = [fe_scores[0], fed_scores[0], baseline_scores[0]]
            recall_list = [fe_scores[1], fed_scores[1], baseline_scores[1]]
            loss_list = [fe_results[2], fed_results[2], baseline_results[2]]
            time_list = [fe_time, fed_time, baseline_time]
            size_transfer_list = [fe_extra[1], fed_extra[1], 0]

            print_results(name_list,
                          accuracy_list,
                          loss_list,
                          precision_list,
                          recall_list,
                          time_list,
                          size_transfer_list,
                          file_path=file_write_path)
