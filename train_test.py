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
import os

import tenseal as ts
from phe import paillier
from diffprivlib.mechanisms import Laplace

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def encrypt_vector(vector, shared_key, scale_factor=1000000):
    """Encrypt a vector element-wise using AES, scaling floats to integers."""
    offset = min(0, min(vector)) * scale_factor
    scaled_vector = [int((v - offset) * scale_factor) for v in vector]
    encrypted_vector = []

    for v in scaled_vector:
        iv = os.urandom(16)  # Generate a random IV
        cipher = Cipher(algorithms.AES(shared_key), modes.CFB(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ct = encryptor.update(
            v.to_bytes((v.bit_length() + 7) // 8, byteorder='big', signed=False)) + encryptor.finalize()
        encrypted_vector.append((iv, ct))

    return encrypted_vector, offset


def compute_inner_product(encrypted_vector, weights, shared_key, scale_factor=1000000, offset=0):
    """Decrypt and compute the inner product using AES, scaling back to floats and adjusting for offset."""
    decrypted_vector = []

    for iv, ct in encrypted_vector:
        cipher = Cipher(algorithms.AES(shared_key), modes.CFB(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        pt = decryptor.update(ct) + decryptor.finalize()
        v = int.from_bytes(pt, byteorder='big', signed=False)
        decrypted_vector.append(v)

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


def train_mlp_binary_baseline(n_epochs, X_train, y_train, X_test, y_test, input_shape, output_shape, dataset_name):
    baseline_train_accuracy = []
    baseline_test_accuracy = []

    baseline_train_loss = []
    baseline_test_loss = []

    baseline_test_precision = []
    baseline_test_recall = []

    bce = BinaryCrossentropy()

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

    if dataset_name == 'ionosphere' or dataset_name == 'parkinson':
        model_tf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model_tf.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:Baseline, Epoch:{epoch + 1}')
        history = model_tf.fit(X_train, y_train, epochs=1, batch_size=8, verbose=0)

        # train_loss = history.history['loss'][0]
        # train_accuracy = history.history['accuracy'][0]
        # baseline_train_loss.append(train_loss)
        # baseline_train_accuracy.append(train_accuracy)

        # test_loss, test_accuracy = model_tf.evaluate(X_test, y_test, verbose=0)

        y_test_pred = (model_tf.predict(X_test) > 0.5).astype("int32")
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_loss = bce(y_test, y_test_pred).numpy()
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)

        baseline_test_loss.append(test_loss)
        baseline_test_accuracy.append(test_accuracy)
        baseline_test_precision.append(test_precision)
        baseline_test_recall.append(test_recall)

    return baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss, baseline_test_precision, baseline_test_recall


def train_mlp_multi_baseline(n_epochs, X_train, y_train, X_test, y_test, input_shape, output_shape, dataset_name, n_classes):
    baseline_train_accuracy = []
    baseline_test_accuracy = []

    baseline_train_loss = []
    baseline_test_loss = []

    baseline_test_precision = []
    baseline_test_recall = []

    y_train_adjusted = y_train - 1
    y_test_adjusted = y_test - 1

    y_train_encoded = to_categorical(y_train_adjusted, num_classes=n_classes)
    y_test_encoded = to_categorical(y_test_adjusted, num_classes=n_classes)

    model_tf = tf.keras.Sequential([
        tf.keras.layers.Dense(output_shape, activation='softmax', input_shape=(input_shape,),
                              kernel_regularizer=tf.keras.regularizers.l2(config.regularization_rate))
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model_tf.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:Baseline, Epoch:{epoch + 1}')
        history = model_tf.fit(X_train, y_train_encoded, epochs=1, batch_size=1, verbose=0)

        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        baseline_train_loss.append(train_loss)
        baseline_train_accuracy.append(train_accuracy)

        test_loss, test_accuracy, test_precision, test_recall = model_tf.evaluate(X_test, y_test_encoded, verbose=0)

        baseline_test_loss.append(test_loss)
        baseline_test_accuracy.append(test_accuracy)
        baseline_test_precision.append(test_precision)
        baseline_test_recall.append(test_recall)

    return baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss, baseline_test_precision, baseline_test_recall


def train_model_multi_classification(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train,
                                     X_test, y_test, n_classes):
    train_accuracy_history = []
    train_loss_history = []

    test_accuracy_history = []
    test_loss_history = []

    test_precision_history = []
    test_recall_history = []

    size_of_transfer_data = 0

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:FedMod, Epoch:{epoch + 1}')
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            smashed_list = []
            for i in range(len(party_list)):
                smashed_list.append(party_list[i].forward_pass_multi_classification(n_classes))

            party_shares = []
            for i in range(len(party_list)):
                party_m_shares = []
                for j in range(n_classes):
                    party_m_shares.append(
                        party_list[i].create_shares(smashed_list[i][j], config.k_value, config.random_coef))
                party_shares.append(party_m_shares)

            reset_servers(server_list)
            main_server.reset()

            for i in range(n_classes):
                servers_get_from_clients(server_list=server_list, party_shares=[party_shares[0][i], party_shares[1][i]])
                sumed_data = servers_sum_data(server_list=server_list)
                main_server.get_multi_data(sumed_data)

            main_server.calculate_multi_loss(n_classes)

            for i in range(len(party_list)):
                party_list[i].update_weights_multi(main_server.error_multi, n_classes)

            temp_error = 0
            for i in range(n_classes):
                temp_error += abs(main_server.error_multi[i]) / n_classes
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

        test_accuracy, test_loss, test_precision, test_recall = test_model_multi_classification(n_parties=len(party_list),
                                                                                                X_test=X_test, y_test=y_test,
                                                                                                party_coefs=parties_coefs,
                                                                                                party_biases=parties_biases,
                                                                                                n_classes=n_classes)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        test_precision_history.append(test_precision)
        test_recall_history.append(test_recall)

        parties_reset(party_list)
        main_server.reset_round()

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights[0])

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data, test_precision_history, test_recall_history


def test_model_multi_classification(n_parties, X_test, y_test, party_coefs, party_biases, n_classes):
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
    y_pred = []
    y_label = []

    for n_data in range(len(parties[0].data)):

        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass_multi_classification(n_classes))

        label_for_test = y_test.loc[count_test_data].to_numpy()[0]
        y_label.append(label_for_test)

        sigmoid_results = []
        for i in range(n_classes):
            sigmoid_results.append(sigmoid(sum(smashed_list[0][i], smashed_list[1][i])))

        predict = np.argmax(sigmoid_results) + 1
        y_pred.append(predict)

        loss_multi = []
        for i in range(1, n_classes + 1):
            if i == label_for_test:
                loss_multi.append(abs(sigmoid_results[i - 1] - 1))
            else:
                loss_multi.append(abs(sigmoid_results[i - 1] - 0))
        test_loss_list.append(sum(loss_multi) / n_classes)

        if predict == label_for_test:
            count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)

    precision_weighted = precision_score(y_label, y_pred, average='weighted')
    recall_weighted = recall_score(y_label, y_pred, average='weighted')

    return accuracy, loss, precision_weighted, recall_weighted


def train_model_binary_classification(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train,
                                      X_test, y_test):
    train_accuracy_history = []
    train_loss_history = []

    test_accuracy_history = []
    test_loss_history = []

    test_precision_history = []
    test_recall_history = []

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

            for i in range(len(party_list)):
                party_list[i].get_batch_error(middle_servers_error)

            if n_data % config.batch_size == config.batch_size - 1:
                for party in party_list:
                    party.update_weights_batch(config.batch_size)
                    party.reset_batch_errors()

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

        test_accuracy, test_loss, test_precision, test_recall = test_model_binary_classification(len(party_list),
                                                                                                 X_test, y_test,
                                                                                                 parties_coefs,
                                                                                                 parties_biases, )

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        test_precision_history.append(test_precision)
        test_recall_history.append(test_recall)

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

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data, test_precision_history, test_recall_history


def test_model_binary_classification(n_parties, X_test, y_test, party_coefs, party_biases):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

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

        if a == 1 and label_for_test == 1:
            true_positive += 1
        elif a == 0 and label_for_test == 0:
            true_negative += 1
        elif a == 1 and label_for_test == 0:
            false_positive += 1
        elif a == 0 and label_for_test == 1:
            false_negative += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)
    precision = true_positive / ((true_positive + false_positive) + 1e-10)
    recall = true_positive / ((true_positive + false_negative) + 1e-10)

    return accuracy, loss, precision, recall


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
                        size_of_transfer_data += sys.getsizeof(encrypted_functional[i])
                main_server_error = main_server.calculate_paillier_loss(encrypted_functional)
            elif type_DP:
                if n_data == 0:
                    for i in range(len(party_list)):
                        size_of_transfer_data += sys.getsizeof(smashed_numbers[i])
                main_server_error = main_server.calculate_DP_loss(smashed_numbers, laplace_mech)

            parties_get_error(party_list, main_server_error)
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

    test_precision_history = []
    test_recall_history = []

    size_of_transfer_data = 0

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Epoch:{epoch + 1}')
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            if epoch == 0:
                encrypted_data_list = []
                offset_list = []
                for i in range(len(party_list)):
                    encrypted_data, offset = encrypt_vector(party_list[i].give_data_for_round(), config.shared_key)
                    offset_list.append(offset)
                    encrypted_data_list.append(encrypted_data)

                main_server.add_to_encrypted_data(encrypted_data_list, offset_list)

            if epoch != 0:
                for i in range(len(party_list)):
                    party_list[i].give_data_for_round()

            main_server.reset()
            main_server_error = main_server.calculate_FE_loss(party_list)

            if n_data == 0:
                for i in range(len(party_list)):
                    size_of_transfer_data += sys.getsizeof(encrypted_data_list[i])
                    size_of_transfer_data += sys.getsizeof(offset_list[i])
                    size_of_transfer_data += sys.getsizeof(party_list[i].weights)

                    size_of_transfer_data += sys.getsizeof(main_server_error)

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

        test_accuracy, test_loss, test_precision, test_recall = test_FE_binary_classification(len(party_list),
                                                                                              X_test, y_test,
                                                                                              parties_coefs,
                                                                                              parties_biases,
                                                                                              party_list)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        test_precision_history.append(test_precision)
        test_recall_history.append(test_recall)

        parties_reset(party_list)
        main_server.reset_round()
        main_server.reset_encrypted_round()

    if dataset_name == 'ionosphere':
        new_df_y_train = pd.DataFrame()
        new_df_y_test = pd.DataFrame()
        class_mapping = {0: 'b', 1: 'g'}
        new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
        new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data, test_precision_history, test_recall_history


def test_FE_binary_classification(n_parties, X_test, y_test, party_coefs, party_biases, party_list):
    n_features = X_test.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

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
            encrypted_data, offset = encrypt_vector(parties[i].give_data_for_round(), config.shared_key)
            offset_list.append(offset)
            encrypted_data_list.append(encrypted_data)

        intermediate_outputs = []
        for i in range(len(party_list)):
            intermediate_outputs.append(compute_inner_product(encrypted_data_list[i],
                                                              party_list[i].weights,
                                                              config.shared_key,
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

        if a == 1 and label_for_test == 1:
            true_positive += 1
        elif a == 0 and label_for_test == 0:
            true_negative += 1
        elif a == 1 and label_for_test == 0:
            false_positive += 1
        elif a == 0 and label_for_test == 1:
            false_negative += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)
    precision = true_positive / ((true_positive + false_positive) + 1e-10)
    recall = true_positive / ((true_positive + false_negative) + 1e-10)

    return accuracy, loss, precision, recall
