import config
import client

import math
import time
import numpy as np
import pandas as pd
import sympy
import secrets
import resource
import sys

import tenseal as ts
from phe import paillier
from diffprivlib.mechanisms import Laplace

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

def encrypt_vector(vector, encryptor, scale_factor=1000000):
    """Encrypt a vector element-wise, scaling floats to integers."""
    # Assuming the largest negative value that we might deal with for proper offset
    offset = min(0, min(vector)) * scale_factor
    scaled_vector = [int((v - offset) * scale_factor) for v in vector]
    encrypted_vector = [encryptor.encrypt(v.to_bytes((v.bit_length() + 7) // 8, byteorder='big', signed=False)) for v in
                        scaled_vector]
    return encrypted_vector, offset


def compute_inner_product(encrypted_vector, weights, decryptor, scale_factor=1000000, offset=0):
    """Decrypt and compute the inner product, scaling back to floats and adjusting for offset."""
    decrypted_vector = [int.from_bytes(decryptor.decrypt(v), byteorder='big', signed=False) for v in encrypted_vector]
    adjusted_vector = [(v / scale_factor) + offset for v in decrypted_vector]
    return np.dot(adjusted_vector, weights)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reset_servers(server_list):
    for server in server_list:
        server.reset()


def parties_create_shares(party_list, problem):
    k_value = config.k_value
    random_coef = config.random_coef

    party_shares = []

    for party in party_list:
        party_shares.append(party.create_shares(party.forward_pass(problem=problem), k_value, random_coef))

    return party_shares


def servers_get_from_clients(server_list, party_shares):
    for i in range(len(party_shares)):
        server_list[0].get_from_client(party_shares[i][0])
        server_list[1].get_from_client(party_shares[i][1])


def servers_sum_data(server_list):
    sumed_data = []

    for server in server_list:
        sumed_data.append(server.sum_data())

    return sumed_data


def main_server_get_data(main_server, sumed_data):
    for i in range(len(sumed_data)):
        main_server.get_data(sumed_data[i])


def parties_get_error(party_list, middle_servers_error):
    for i in range(len(party_list)):
        party_list[i].get_error(middle_servers_error)


def parties_update_weights(party_list):
    for party in party_list:
        party.update_weights()


def parties_reset(party_list):
    for party in party_list:
        party.reset()


def add_markers_at_intervals(y_data, marker_symbol, marker_color, interval=50):
    markers_x = list(range(0, len(y_data), interval))
    markers_y = [y_data[x] for x in markers_x]
    return go.Scatter(
        x=markers_x,
        y=markers_y,
        mode='markers',
        marker=dict(symbol=marker_symbol, size=8, color=marker_color),
        showlegend=False
    )


def figure_for_classification(type_name, train_history, test_history, baseline_train_history, baseline_test_history,
                              dataset_name=None):
    trace1 = go.Scatter(y=train_history, mode='lines', name='FedMod Train',
                        line=dict(color='rgba(100, 149, 237, 1)', width=2, dash='solid'))
    trace2 = go.Scatter(y=test_history, mode='lines', name='FedMod Test',
                        line=dict(color='rgba(65, 105, 225, 1)', width=2, dash='dash'))
    trace3 = go.Scatter(y=baseline_train_history, mode='lines', name='Baseline Train',
                        line=dict(color='rgba(255, 160, 122, 1)', width=2, dash='solid'))
    trace4 = go.Scatter(y=baseline_test_history, mode='lines', name='Baseline Test',
                        line=dict(color='rgba(205, 92, 92, 1)', width=2, dash='dash'))

    markers_interval = config.plot_intervals
    trace1_markers = add_markers_at_intervals(train_history, 'circle', 'rgba(100, 149, 237, 1)', markers_interval)
    trace2_markers = add_markers_at_intervals(test_history, 'diamond', 'rgba(65, 105, 225, 1)', markers_interval)
    trace3_markers = add_markers_at_intervals(baseline_train_history, 'circle', 'rgba(255, 160, 122, 1)',
                                              markers_interval)
    trace4_markers = add_markers_at_intervals(baseline_test_history, 'diamond', 'rgba(205, 92, 92, 1)',
                                              markers_interval)

    layout = go.Layout(title=f'Dataset: {dataset_name}',
                       xaxis=dict(title='Epochs',
                                  tickvals=list(range(0, len(train_history) + 1, markers_interval)),
                                  showgrid=True,
                                  zeroline=False),
                       yaxis=dict(title=type_name),
                       legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                       plot_bgcolor='rgba(242, 242, 242, 1)')

    fig1 = go.Figure(
        data=[trace1, trace2, trace3, trace4, trace1_markers, trace2_markers, trace3_markers, trace4_markers],
        layout=layout)
    fig1.show()

    print('------' + type_name + '------')
    print('Train:', train_history[-1])
    print('Test:', test_history[-1])
    print('Base Train:', baseline_train_history[-1])
    print('Base Test:', baseline_test_history[-1])


def train_mlp_binary_baseline(n_epochs, X_train, y_train, X_test, y_test, input_shape, output_shape, dataset_name):
    baseline_train_accuracy = []
    baseline_test_accuracy = []
    baseline_train_loss = []
    baseline_test_loss = []

    if dataset_name == 'ionosphere':
        X_train = X_train.astype('float32')
        y_train = y_train.astype('float32')
        X_test = X_test.astype('float32')
        y_test = y_test.astype('float32')

    model_tf = tf.keras.Sequential([
        tf.keras.layers.Dense(output_shape, activation='sigmoid', input_shape=(input_shape,),
                              kernel_regularizer=tf.keras.regularizers.l2(config.regularization_rate))
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)

    if dataset_name == 'ionosphere':
        model_tf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model_tf.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:Baseline, Epoch:{epoch + 1}')
        history = model_tf.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)

        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        baseline_train_loss.append(train_loss)
        baseline_train_accuracy.append(train_accuracy)

        test_loss, test_accuracy = model_tf.evaluate(X_test, y_test, verbose=0)

        baseline_test_loss.append(test_loss)
        baseline_test_accuracy.append(test_accuracy)

    return baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss


def train_mlp_multi_baseline(n_epochs, X_train, y_train, X_test, y_test, input_shape, output_shape):
    baseline_train_accuracy = []
    baseline_test_accuracy = []
    baseline_train_loss = []
    baseline_test_loss = []

    y_train_adjusted = y_train - 1
    y_test_adjusted = y_test - 1

    y_train_encoded = to_categorical(y_train_adjusted, num_classes=3)
    y_test_encoded = to_categorical(y_test_adjusted, num_classes=3)

    model_tf = tf.keras.Sequential([
        tf.keras.layers.Dense(output_shape, activation='softmax', input_shape=(input_shape,),
                              kernel_regularizer=tf.keras.regularizers.l2(config.regularization_rate))
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model_tf.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model_tf.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        history = model_tf.fit(X_train, y_train_encoded, epochs=1, batch_size=3, verbose=0)

        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        baseline_train_loss.append(train_loss)
        baseline_train_accuracy.append(train_accuracy)

        test_loss, test_accuracy = model_tf.evaluate(X_test, y_test_encoded, verbose=0)

        baseline_test_loss.append(test_loss)
        baseline_test_accuracy.append(test_accuracy)

    return baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss


def train_parties_1(n_epochs, party_list, server_list, main_server, X_test, y_test):
    error_history = []
    test_loss_history = []

    for epoch in range(n_epochs):
        for n_data in range(party_list[0].data.shape[0]):
            party_shares = parties_create_shares(party_list, 'regression')

            reset_servers(server_list)
            servers_get_from_clients(server_list, party_shares)
            sumed_data = servers_sum_data(server_list)

            main_server.reset()
            main_server_get_data(main_server, sumed_data)
            main_server.calculate_loss(problem='regression')

            middle_servers_error = main_server.error
            parties_get_error(party_list, middle_servers_error)

            parties_update_weights(party_list)

        error_history.append(abs(party_list[0].error))

        parties_coefs = []
        parties_biases = []
        for party in party_list:
            parties_coefs.append(party.weights)
            parties_biases.append(party.bias)
        test_loss_history.append(abs(test_parties_1(X_test, y_test,
                                                    parties_coefs,
                                                    parties_biases)))

        parties_reset(party_list)

    fig = px.line(y=error_history, markers=True)
    fig.add_scatter(y=test_loss_history)
    fig.show()


def test_parties_1(X_test, y_test, party_coefs, party_biases):
    data_party_test1 = X_test.iloc[:, :7]
    data_party_test2 = X_test.iloc[:, 7:]

    parties = []
    parties_data = [data_party_test1, data_party_test2]

    for i in range(len(party_coefs)):
        if i == 0:
            parties.append(client.Client(name=f'party_test{i}',
                                         weights=party_coefs[i],
                                         bias=party_biases[i],
                                         data=parties_data[i],
                                         lead=1,
                                         labels=y_test))
        else:
            parties.append(client.Client(name=f'party_test{i}',
                                         weights=party_coefs[i],
                                         bias=party_biases[i],
                                         data=parties_data[i],
                                         lead=0))

    test_loss_list = []

    for n_data in range(len(parties[0].data)):

        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass(problem='regression'))

        test_loss_list.append(sum(smashed_list))

    return np.average(test_loss_list)


def train_parties_2(n_epochs, party_list, server_list, main_server, X_train, y_train, X_test, y_test):
    start_resources = resource.getrusage(resource.RUSAGE_SELF)
    start_time_fedmod = time.time()

    train_accuracy_history = []
    train_loss_history = []

    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(n_epochs):
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            party_shares = parties_create_shares(party_list, 'classification')

            reset_servers(server_list)
            servers_get_from_clients(server_list, party_shares)
            sumed_data = servers_sum_data(server_list)

            main_server.reset()
            main_server_get_data(main_server, sumed_data)
            main_server.calculate_loss(problem='classification')

            middle_servers_error = main_server.error
            parties_get_error(party_list, middle_servers_error)

            parties_update_weights(party_list)

            error_history.append(abs(party_list[0].error))
            if main_server.correct == 1:
                correct_count += 1

        train_accuracy_history.append(correct_count / party_list[0].data.shape[0])
        train_loss_history.append(np.average(error_history))

        parties_coefs = []
        parties_biases = []
        for party in party_list:
            parties_coefs.append(party.weights)
            parties_biases.append(party.bias)

        test_accuracy, test_loss = test_parties_2(X_test, y_test,
                                                  parties_coefs,
                                                  parties_biases, 2)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        parties_reset(party_list)
        main_server.reset_round()

    end_time_fedmod = time.time()
    end_resources = resource.getrusage(resource.RUSAGE_SELF)

    baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss = train_mlp_binary_baseline(
        n_epochs,
        X_train,
        y_train,
        X_test,
        y_test,
        input_shape=13,
        output_shape=1,
        dataset_name='heart')

    figure_for_classification('Loss', train_loss_history, test_loss_history, baseline_train_loss, baseline_test_loss,
                              dataset_name='heart')
    figure_for_classification('Accuracy', train_accuracy_history, test_accuracy_history, baseline_train_accuracy,
                              baseline_test_accuracy, dataset_name='heart')

    runtime_fedmod = end_time_fedmod - start_time_fedmod
    cpu_time_used = end_resources.ru_utime - start_resources.ru_utime
    memory_usage = end_resources.ru_maxrss - start_resources.ru_maxrss

    return runtime_fedmod, cpu_time_used, memory_usage


def test_parties_2(X_test, y_test, party_coefs, party_biases, n_parties):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    data_party_test1 = X_test.iloc[:, :column_share + extra_columns]
    data_party_test2 = X_test.iloc[:, column_share + extra_columns:]

    parties = []
    parties_data = [data_party_test1, data_party_test2]

    for i in range(len(party_coefs)):
        parties.append(client.Client(name=f'party_test{i}',
                                     weights=party_coefs[i],
                                     bias=party_biases[i],
                                     data=parties_data[i],
                                     lead=0))

    count_test_data = 0
    count_correct = 0
    test_loss_list = []

    for n_data in range(len(parties[0].data)):
        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass(problem='classification'))

        label_for_test = y_test.loc[count_test_data]
        label_for_test = label_for_test.to_numpy()
        label_for_test = label_for_test[0]

        a = sigmoid(sum(smashed_list))

        test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            if (label_for_test == a):
                count_correct += 1
        else:
            a = 0
            if (label_for_test == a):
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    return accuracy, loss


def train_parties_3(n_epochs, party1, party2, server1, server2, main_server, X_train, y_train, X_test, y_test):
    start_resources = resource.getrusage(resource.RUSAGE_SELF)
    start_time_fedmod = time.time()

    k_value = config.k_value

    error_history = []

    train_accuracy_history = []
    train_loss_history = []

    test_loss_history = []
    test_accuracy_history = []

    # epoch_data = 5
    epoch_data = len(party1.data)

    for epoch in range(n_epochs):
        error_history = []
        correct_count = 0
        for n_data in range(epoch_data):
            party1_smashed_data = party1.forward_pass(problem='classification')
            party2_smashed_data = party2.forward_pass(problem='classification')

            p1_s1, p1_s2 = party1.create_shares(party1_smashed_data, k_value, config.random_coef)
            p2_s1, p2_s2 = party2.create_shares(party2_smashed_data, k_value, config.random_coef)

            server1.reset()
            server2.reset()

            server1.get_from_client(p1_s1)
            server1.get_from_client(p2_s1)
            server2.get_from_client(p1_s2)
            server2.get_from_client(p2_s2)

            server_1_data = server1.sum_data()
            server_2_data = server2.sum_data()

            main_server.reset()

            main_server.get_data(server_1_data)
            main_server.get_data(server_2_data)

            main_server.calculate_loss(problem='classification')

            server1_error = server1.get_from_main_server(main_server.error)
            server2_error = server2.get_from_main_server(main_server.error)

            party1.get_error(server1_error)
            party2.get_error(server2_error)
            party1.update_weights()
            party2.update_weights()

            error_history.append(abs(party1.error))

            if main_server.correct == 1:
                correct_count += 1

        train_accuracy_history.append(correct_count / epoch_data)
        train_loss_history.append(np.average(error_history))
        test_accuracy, test_loss = test_parties_3(X_test, y_test,
                                                  party1.weights,
                                                  party2.weights,
                                                  party1.bias,
                                                  party2.bias)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        party1.reset()
        party2.reset()
        main_server.reset_round()

    end_time_fedmod = time.time()
    end_resources = resource.getrusage(resource.RUSAGE_SELF)

    new_df_y_train = pd.DataFrame()
    new_df_y_test = pd.DataFrame()
    class_mapping = {0: 'b', 1: 'g'}
    new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
    new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss = train_mlp_binary_baseline(
        n_epochs,
        X_train,
        y_train,
        X_test,
        y_test,
        input_shape=34,
        output_shape=1,
        dataset_name='ionosphere')

    figure_for_classification('Loss', train_loss_history, test_loss_history, baseline_train_loss, baseline_test_loss,
                              dataset_name='ionosphere')
    figure_for_classification('Accuracy', train_accuracy_history, test_accuracy_history, baseline_train_accuracy,
                              baseline_test_accuracy, dataset_name='ionosphere')

    runtime_fedmod = end_time_fedmod - start_time_fedmod
    cpu_time_used = end_resources.ru_utime - start_resources.ru_utime
    memory_usage = end_resources.ru_maxrss - start_resources.ru_maxrss

    return runtime_fedmod, cpu_time_used, memory_usage


def test_parties_3(X_test, y_test, party1_coef, party2_coef, party1_bias, party2_bias):
    data_party_test1 = X_test.iloc[:, :17]
    data_party_test2 = X_test.iloc[:, 17:]

    party_test1 = client.Client(name='party_test1',
                                weights=party1_coef,
                                bias=party1_bias,
                                data=data_party_test1,
                                lead=0)
    party_test2 = client.Client(name='party_test2',
                                weights=party2_coef,
                                bias=party2_bias,
                                data=data_party_test2,
                                lead=0)

    count_test_data = 0
    count_correct = 0
    test_loss_list = []

    for n_data in range(len(party_test1.data)):
        smashed1 = party_test1.forward_pass(problem='classification')
        smashed2 = party_test2.forward_pass(problem='classification')

        label_for_test = y_test.loc[count_test_data]
        label_for_test = label_for_test.to_numpy()
        label_for_test = label_for_test[0]

        a = sigmoid(smashed1 + smashed2)

        test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            if (label_for_test == a):
                count_correct += 1
        else:
            a = 0
            if (label_for_test == a):
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    return accuracy, loss


def train_parties_4(n_epochs, party_list, server_list, main_server, X_train, y_train, X_test, y_test, n_classes):
    start_resources = resource.getrusage(resource.RUSAGE_SELF)
    start_time_fedmod = time.time()

    train_accuracy_history = []
    train_loss_history = []

    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            party1_smashed_list = party_list[0].forward_pass_multi_classification(n_classes)
            party2_smashed_list = party_list[1].forward_pass_multi_classification(n_classes)

            party_1_shares = []
            party_2_shares = []
            for i in range(n_classes):
                party_1_shares.append(
                    party_list[0].create_shares(party1_smashed_list[i], config.k_value, config.random_coef))
                party_2_shares.append(
                    party_list[1].create_shares(party2_smashed_list[i], config.k_value, config.random_coef))

            reset_servers(server_list)
            main_server.reset()
            for i in range(n_classes):
                servers_get_from_clients(server_list, [party_1_shares[i], party_2_shares[i]])
                sumed_data = servers_sum_data(server_list)
                main_server.get_multi_data(sumed_data)

            main_server.calculate_multi_loss(n_classes)

            party_list[0].update_weights_multi(main_server.error_multi, n_classes)
            party_list[1].update_weights_multi(main_server.error_multi, n_classes)

            temp_error = 0
            for i in range(n_classes):
                temp_error += abs(main_server.error_multi[i]) / 3
            error_history.append(temp_error)

            if main_server.correct == 1:
                correct_count += 1

        train_accuracy_history.append(correct_count / party_list[0].data.shape[0])
        train_loss_history.append(np.average(error_history))

        parties_coefs = []
        parties_biases = []
        for party in party_list:
            parties_coefs.append(party.weights)
            parties_biases.append(party.bias)

        test_accuracy, test_loss = test_parties_4(X_test, y_test,
                                                  parties_coefs,
                                                  parties_biases, 2, n_classes)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        parties_reset(party_list)
        main_server.reset_round()

    end_time_fedmod = time.time()
    end_resources = resource.getrusage(resource.RUSAGE_SELF)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights[0])
    baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss = train_mlp_multi_baseline(
        n_epochs,
        X_train,
        y_train,
        X_test,
        y_test,
        input_shape=input_shape,
        output_shape=3)

    figure_for_classification('Loss', train_loss_history, test_loss_history,
                              baseline_train_loss, baseline_test_loss, dataset_name='phishing')
    figure_for_classification('Accuracy', train_accuracy_history, test_accuracy_history,
                              baseline_train_accuracy, baseline_test_accuracy, dataset_name='phishing')

    runtime_fedmod = end_time_fedmod - start_time_fedmod
    cpu_time_used = end_resources.ru_utime - start_resources.ru_utime
    memory_usage = end_resources.ru_maxrss - start_resources.ru_maxrss

    return runtime_fedmod, cpu_time_used, memory_usage


def test_parties_4(X_test, y_test, party_coefs, party_biases, n_parties, n_classes):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    data_party_test1 = X_test.iloc[:, :column_share + extra_columns]
    data_party_test2 = X_test.iloc[:, column_share + extra_columns:]

    parties = []
    parties_data = [data_party_test1, data_party_test2]

    for i in range(len(party_coefs)):
        parties.append(client.Client(name=f'party_test{i}',
                                     weights=party_coefs[i],
                                     bias=party_biases[i],
                                     data=parties_data[i],
                                     lead=0))

    count_test_data = 0
    count_correct = 0
    test_loss_list = []

    for n_data in range(len(parties[0].data)):

        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass_multi_classification(n_classes))

        label_for_test = y_test.loc[count_test_data]
        label_for_test = label_for_test.to_numpy()
        label_for_test = label_for_test[0]

        sigmoid_results = []
        for i in range(n_classes):
            sigmoid_results.append(sigmoid(sum(smashed_list[0][i], smashed_list[1][i])))

        predict = np.argmax(sigmoid_results) + 1

        loss_multi = []
        for i in range(1, n_classes + 1):
            if i == label_for_test:
                loss_multi.append(abs(sigmoid_results[i - 1] - 1))
            else:
                loss_multi.append(abs(sigmoid_results[i - 1] - 0))
        test_loss_list.append(sum(loss_multi) / 3)

        if predict == label_for_test:
            count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    return accuracy, loss


def train_model_binary_classification(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train,
                                      X_test, y_test):
    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []

    size_of_transfer_data = 0

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:FedMod, Epoch:{epoch + 1}')
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            party_shares = []
            for party in party_list:
                party_shares.append(party.create_shares(party.forward_pass(problem='classification'), config.k_value,
                                                        config.random_coef))

            reset_servers(server_list)

            for i in range(len(party_shares)):
                server_list[0].get_from_client(party_shares[i][0])
                server_list[1].get_from_client(party_shares[i][1])

            sumed_data = []
            for server in server_list:
                sumed_data.append(server.sum_data())

            main_server.reset()
            main_server_get_data(main_server, sumed_data)
            main_server.calculate_loss(problem='classification')

            middle_servers_error = main_server.error
            parties_get_error(party_list, middle_servers_error)
            parties_update_weights(party_list)

            error_history.append(abs(party_list[0].error))
            if main_server.correct == 1:
                correct_count += 1

            if n_data == 0:
                for i in range(len(party_list)):
                    for j in range(len(server_list)):
                        size_of_transfer_data += sys.getsizeof(party_shares[i][j])
                for i in range(len(server_list)):
                    size_of_transfer_data += sys.getsizeof(sumed_data[i])
                for i in range(len(party_list)):
                    size_of_transfer_data += sys.getsizeof(main_server.error)

        train_accuracy_history.append(correct_count / party_list[0].data.shape[0])
        train_loss_history.append(np.average(error_history))

        parties_coefs = []
        parties_biases = []
        for party in party_list:
            parties_coefs.append(party.weights)
            parties_biases.append(party.bias)

        test_accuracy, test_loss = test_model_binary_classification(len(party_list), X_test, y_test, parties_coefs,
                                                                    parties_biases, )

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        parties_reset(party_list)
        main_server.reset_round()

    if dataset_name == 'ionosphere':
        new_df_y_train = pd.DataFrame()
        new_df_y_test = pd.DataFrame()
        class_mapping = {0: 'b', 1: 'g'}
        new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
        new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data


def test_model_binary_classification(n_parties, X_test, y_test, party_coefs, party_biases):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    parties_data = []
    parties = []

    for i in range(n_parties):
        if i == n_parties - 1:
            data_party = X_test.iloc[:, i * column_share:]
        else:
            data_party = X_test.iloc[:, i * column_share:(i + 1) * column_share]
        parties_data.append(data_party)

    for i in range(n_parties):
        parties.append(client.Client(name=f'party_test{i + 1}',
                                     weights=party_coefs[i],
                                     bias=party_biases[i],
                                     data=parties_data[i],
                                     lead=0))

    count_test_data = 0
    count_correct = 0
    test_loss_list = []

    for n_data in range(len(parties[0].data)):
        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass(problem='classification'))

        label_for_test = y_test.loc[count_test_data]
        label_for_test = label_for_test.to_numpy()
        label_for_test = label_for_test[0]

        a = sigmoid(sum(smashed_list))
        test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            if label_for_test == a:
                count_correct += 1
        else:
            a = 0
            if label_for_test == a:
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    return accuracy, loss


def train_HE_binary_classification(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train,
                                   X_test, y_test):
    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []

    size_of_transfer_data = 0

    type_HE = config.type_HE
    type_paillier = config.type_paillier
    type_DP = config.type_DP
    if not type_HE and not type_paillier and not type_DP:
        config.type_HE = True

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Epoch:{epoch + 1}')
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            smashed_list = []
            for i in range(len(party_list)):
                smashed_list.append(party_list[i].forward_pass(problem='classification'))

            smashed_numbers = []
            for i in range(len(smashed_list)):
                smashed_numbers.append(smashed_list[i][0])

            encrypted_functional = []
            if type_HE:
                encrypted_numbers = [ts.ckks_vector(config.context, [num]) for num in smashed_numbers]
            elif type_paillier:
                # encrypted_number1 = config.public_key.encrypt(smashed_numbers[0])
                # encrypted_number2 = config.public_key.encrypt(smashed_numbers[1])
                for i in range(len(party_list)):
                    encrypted_functional.append(config.public_key.encrypt(smashed_numbers[i]))
            elif type_DP:
                epsilon = 1.0  # Privacy budget
                laplace_mech = Laplace(epsilon=epsilon, sensitivity=1)

            main_server.reset()

            if type_HE:
                if n_data == 0:
                    for i in range(len(party_list)):
                        size_of_transfer_data += sys.getsizeof(encrypted_numbers[i])
                main_server_error = main_server.calculate_HE_loss(encrypted_numbers)
            elif type_paillier:
                if n_data == 0:
                    for i in range(len(party_list)):
                        # size_of_transfer_data += sys.getsizeof(encrypted_number1)
                        # size_of_transfer_data += sys.getsizeof(encrypted_number2)
                        size_of_transfer_data += sys.getsizeof(encrypted_functional[i])
                # main_server_error = main_server.calculate_paillier_loss([encrypted_number1, encrypted_number2])
                main_server_error = main_server.calculate_paillier_loss(encrypted_functional)
            elif type_DP:
                if n_data == 0:
                    for i in range(len(party_list)):
                        size_of_transfer_data += sys.getsizeof(smashed_numbers[i])
                main_server_error = main_server.calculate_DP_loss(smashed_numbers, laplace_mech)

            parties_get_error(party_list, main_server_error)
            # if n_data == 0:
            #     error_enc = config.public_key.encrypt(main_server_error)
            #     for i in range(len(party_list)):
            #         size_of_transfer_data += sys.getsizeof(error_enc)
            if n_data == 0:
                for i in range(len(party_list)):
                    size_of_transfer_data += sys.getsizeof(main_server_error)

            parties_update_weights(party_list)

            error_history.append(abs(party_list[0].error))
            if main_server.correct == 1:
                correct_count += 1

        train_accuracy_history.append(correct_count / party_list[0].data.shape[0])
        train_loss_history.append(np.average(error_history))

        parties_coefs = []
        parties_biases = []
        for party in party_list:
            parties_coefs.append(party.weights)
            parties_biases.append(party.bias)

        test_accuracy, test_loss = test_HE_binary_classification(len(party_list), X_test, y_test,
                                                                 parties_coefs, parties_biases, )

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        parties_reset(party_list)
        main_server.reset_round()

    if dataset_name == 'ionosphere':
        new_df_y_train = pd.DataFrame()
        new_df_y_test = pd.DataFrame()
        class_mapping = {0: 'b', 1: 'g'}
        new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
        new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data


def test_HE_binary_classification(n_parties, X_test, y_test, party_coefs, party_biases):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    type_HE = config.type_HE
    type_paillier = config.type_paillier
    type_DP = config.type_DP
    if not type_HE and not type_paillier and not type_DP:
        config.type_HE = True

    parties_data = []
    parties = []

    for i in range(n_parties):
        if i == n_parties - 1:
            data_party = X_test.iloc[:, i * column_share:]
        else:
            data_party = X_test.iloc[:, i * column_share:(i + 1) * column_share]
        parties_data.append(data_party)

    for i in range(n_parties):
        parties.append(client.Client(name=f'party_test{i + 1}',
                                     weights=party_coefs[i],
                                     bias=party_biases[i],
                                     data=parties_data[i],
                                     lead=0))

    count_test_data = 0
    count_correct = 0
    test_loss_list = []

    for n_data in range(len(parties[0].data)):
        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass(problem='classification'))

        smashed_numbers = []
        for i in range(len(smashed_list)):
            smashed_numbers.append(smashed_list[i][0])

        encrypted_functional = []
        if type_HE:
            encrypted_numbers = [ts.ckks_vector(config.context, [num]) for num in smashed_numbers]
        elif type_paillier:
            # encrypted_number1 = config.public_key.encrypt(smashed_numbers[0])
            # encrypted_number2 = config.public_key.encrypt(smashed_numbers[1])
            for i in range(n_parties):
                encrypted_functional.append(config.public_key.encrypt(smashed_numbers[i]))
        elif type_DP:
            epsilon = 1.0  # Privacy budget
            laplace_mech = Laplace(epsilon=epsilon, sensitivity=1)

        label_for_test = y_test.loc[count_test_data]
        label_for_test = label_for_test.to_numpy()
        label_for_test = label_for_test[0]

        if type_HE:
            encrypted_sum = encrypted_numbers[0]
            for enc_num in encrypted_numbers[1:]:
                encrypted_sum += enc_num

            decrypted_sum = np.float64(encrypted_sum.decrypt()[0])
            a = sigmoid(decrypted_sum)
            test_loss_list.append(abs(a - label_for_test))

        elif type_paillier:
            # encrypted_sum = encrypted_number1 + encrypted_number2
            encrypted_sum = 0
            for i in range(n_parties):
                encrypted_sum += encrypted_functional[i]
            decrypted_sum = config.private_key.decrypt(encrypted_sum)
            a = sigmoid(decrypted_sum)
            test_loss_list.append(abs(a - label_for_test))

        elif type_DP:
            noisy_sum = sum(laplace_mech.randomise(value) for value in smashed_numbers)
            a = sigmoid(noisy_sum)
            test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            if label_for_test == a:
                count_correct += 1
        else:
            a = 0
            if label_for_test == a:
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    return accuracy, loss


def train_FE_binary_classification(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train,
                                   X_test, y_test):
    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Epoch:{epoch + 1}')
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            encrypted_data_list = []
            offset_list = []
            for i in range(len(party_list)):
                # data_for_round = party_list[i].give_data_for_round()
                # print(data_for_round)
                encrypted_data, offset = encrypt_vector(party_list[i].give_data_for_round(), config.encryptor)
                offset_list.append(offset)
                encrypted_data_list.append(encrypted_data)

            main_server.reset()
            main_server_error = main_server.calculate_FE_loss(party_list, encrypted_data_list, offset_list)

            parties_get_error(party_list, main_server_error)
            parties_update_weights(party_list)

            error_history.append(abs(party_list[0].error))
            if main_server.correct == 1:
                correct_count += 1

        train_accuracy_history.append(correct_count / party_list[0].data.shape[0])
        train_loss_history.append(np.average(error_history))

        parties_coefs = []
        parties_biases = []
        for party in party_list:
            parties_coefs.append(party.weights)
            parties_biases.append(party.bias)

        test_accuracy, test_loss = test_FE_binary_classification(len(party_list), X_test, y_test,
                                                                 parties_coefs, parties_biases, party_list)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        parties_reset(party_list)
        main_server.reset_round()

    if dataset_name == 'ionosphere':
        new_df_y_train = pd.DataFrame()
        new_df_y_test = pd.DataFrame()
        class_mapping = {0: 'b', 1: 'g'}
        new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
        new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights)

    #TODO
    size_of_transfer_data = 0

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data


def test_FE_binary_classification(n_parties, X_test, y_test, party_coefs, party_biases, party_list):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    parties_data = []
    parties = []

    for i in range(n_parties):
        if i == n_parties - 1:
            data_party = X_test.iloc[:, i * column_share:]
        else:
            data_party = X_test.iloc[:, i * column_share:(i + 1) * column_share]
        parties_data.append(data_party)

    for i in range(n_parties):
        parties.append(client.Client(name=f'party_test{i + 1}',
                                     weights=party_coefs[i],
                                     bias=party_biases[i],
                                     data=parties_data[i],
                                     lead=0))

    count_test_data = 0
    count_correct = 0
    test_loss_list = []

    for n_data in range(len(parties[0].data)):

        encrypted_data_list = []
        offset_list = []
        for i in range(len(party_list)):
            encrypted_data, offset = encrypt_vector(parties[i].give_data_for_round(), config.encryptor)
            offset_list.append(offset)
            encrypted_data_list.append(encrypted_data)

        intermediate_outputs = []
        for i in range(len(party_list)):
            intermediate_outputs.append(compute_inner_product(encrypted_data_list[i],
                                                              party_list[i].weights,
                                                              config.decryptor,
                                                              offset=offset_list[i]))

        label_for_test = y_test.loc[count_test_data]
        label_for_test = label_for_test.to_numpy()
        label_for_test = label_for_test[0]

        FE_sum = np.sum(intermediate_outputs)
        a = sigmoid(FE_sum)
        test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            if label_for_test == a:
                count_correct += 1
        else:
            a = 0
            if label_for_test == a:
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    return accuracy, loss
