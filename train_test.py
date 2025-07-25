import config
import client
import functions as func

import numpy as np
import pandas as pd
import sys
import tenseal as ts
from diffprivlib.mechanisms import Laplace
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score


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

    metrics = ['accuracy',
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')
               ]

    model_tf = tf.keras.Sequential([
        tf.keras.layers.Dense(output_shape, activation='sigmoid', input_shape=(input_shape,),
                              kernel_regularizer=tf.keras.regularizers.l2(config.regularization_rate))
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)

    if dataset_name == 'ionosphere' or dataset_name == 'parkinson':
        model_tf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    else:
        model_tf.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=metrics)

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:Baseline, Epoch:{epoch + 1}')
        model_tf.fit(X_train, y_train, epochs=1, batch_size=config.batch_size, verbose=0)

        test_loss, test_accuracy, test_precision, test_recall = model_tf.evaluate(X_test, y_test, verbose=0)

        baseline_test_loss.append(test_loss)
        baseline_test_accuracy.append(test_accuracy)
        baseline_test_precision.append(test_precision)
        baseline_test_recall.append(test_recall)

    return baseline_train_accuracy, baseline_test_accuracy, baseline_train_loss, baseline_test_loss, baseline_test_precision, baseline_test_recall


def train_mlp_multi_baseline(n_epochs, X_train, y_train, X_test, y_test, input_shape, output_shape, dataset_name,
                             n_classes):
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

            if n_data == 0 and epoch == 0:
                for j in range(n_classes):
                    for i in range(len(party_list)):
                        party_m_share = party_shares[i]
                        size_of_transfer_data += sys.getsizeof(party_m_share[0])
                        size_of_transfer_data += sys.getsizeof(party_m_share[1][0])

            for server in server_list:
                server.reset()
            main_server.reset()

            for i in range(n_classes):
                func.servers_get_from_clients(server_list=server_list,
                                              party_shares=[party_shares[0][i], party_shares[1][i]])
                sumed_data = func.servers_sum_data(server_list=server_list)
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

            if n_data == 0 and epoch == 0:
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

        test_accuracy, test_loss, test_precision, test_recall = test_model_multi_classification(
            n_parties=len(party_list),
            X_test=X_test, y_test=y_test,
            party_coefs=parties_coefs,
            party_biases=parties_biases,
            n_classes=n_classes)

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        test_precision_history.append(test_precision)
        test_recall_history.append(test_recall)

        func.parties_reset(party_list)
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
            sigmoid_results.append(func.sigmoid(sum(smashed_list[0][i], smashed_list[1][i])))

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
                party_shares.append(party.create_shares(party.forward_pass(problem='classification'), config.k_value, config.random_coef))

            # for i in range(len(party_shares)):
            #     for j in range(len(party_shares[0])):
            #         print(party_shares[i][j])

            # print(type(party_shares[0][0]))
            # print(type(party_shares[0][1]))

            if epoch == 0:
                for i in range(len(party_list)):
                    size_of_transfer_data += config.n_servers * 16
                for i in range(len(server_list)):
                    size_of_transfer_data += 16

            for server in server_list:
                server.reset()

            for i in range(len(party_shares)):
                for j in range(config.n_servers):
                    server_list[j].get_from_client(party_shares[i][j])
                # server_list[0].get_from_client(party_shares[i][0])
                # server_list[1].get_from_client(party_shares[i][1])

            sumed_data = []
            for server in server_list:
                sumed_data.append(server.sum_data())

            main_server.reset()
            func.main_server_get_data(main_server, sumed_data)
            main_server.calculate_loss(problem='classification')

            middle_servers_error = main_server.error
            func.parties_get_error(party_list, middle_servers_error)

            for i in range(len(party_list)):
                party_list[i].get_batch_error(middle_servers_error)

            if n_data % config.batch_size == config.batch_size - 1:
                for party in party_list:
                    party.update_weights_batch(config.batch_size)
                    party.reset_batch_errors()

            error_history.append(abs(party_list[0].error))
            if main_server.correct == 1:
                correct_count += 1

            if epoch == 0:
                for i in range(len(server_list)):
                    # size_of_transfer_data += sys.getsizeof(sumed_data[i])
                    size_of_transfer_data += 16 / config.batch_size
                    for i in range(len(party_list)):
                        size_of_transfer_data += 16 / config.batch_size

        # print(size_of_transfer_data/8)

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

        func.parties_reset(party_list)
        main_server.reset_round()

    # if dataset_name == 'ionosphere':
    #     new_df_y_train = pd.DataFrame()
    #     new_df_y_test = pd.DataFrame()
    #     class_mapping = {0: 'b', 1: 'g'}
    #     new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
    #     new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data, test_precision_history, test_recall_history


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
    y_label = []
    y_pred = []

    for n_data in range(len(parties[0].data)):
        smashed_list = []
        for i in range(len(parties)):
            smashed_list.append(parties[i].forward_pass(problem='classification'))

        label_for_test = y_test.loc[count_test_data].to_numpy()[0]
        y_label.append(label_for_test)

        a = func.sigmoid(sum(smashed_list))
        test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            y_pred.append(a)
            if label_for_test == a:
                count_correct += 1
        else:
            a = 0
            y_pred.append(a)
            if label_for_test == a:
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)
    precision_weighted = precision_score(y_label, y_pred, average='weighted')
    recall_weighted = recall_score(y_label, y_pred, average='weighted')

    return accuracy, loss, precision_weighted, recall_weighted


def train_binary_nosec(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train, X_test, y_test):
    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []
    test_precision_history = []
    test_recall_history = []

    size_of_transfer_data = 0

    for epoch in range(n_epochs):
        print(f'Dataset:{dataset_name}, Alg:Baseline, Epoch:{epoch + 1}')
        error_history = []
        correct_count = 0
        for n_data in range(party_list[0].data.shape[0]):

            intermediate_outputs = []
            for party in party_list:
                intermediate_outputs.append(party.forward_pass(problem='classification'))

            main_server.reset()
            func.main_server_get_data(main_server, intermediate_outputs)
            main_server.calculate_loss(problem='classification')

            func.parties_get_error(party_list, main_server.error)

            for i in range(len(party_list)):
                party_list[i].get_batch_error(main_server.error)

            if n_data % config.batch_size == config.batch_size - 1:
                for party in party_list:
                    party.update_weights_batch(config.batch_size)
                    party.reset_batch_errors()

            error_history.append(abs(party_list[0].error))
            if main_server.correct == 1:
                correct_count += 1

            if epoch == 0:
                for i in range(len(party_list)):
                    size_of_transfer_data += 16
                for i in range(len(party_list)):
                    size_of_transfer_data += 16 / config.batch_size

        # print(size_of_transfer_data / 8)

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

        func.parties_reset(party_list)
        main_server.reset_round()

    # if dataset_name == 'ionosphere':
    #     new_df_y_train = pd.DataFrame()
    #     new_df_y_test = pd.DataFrame()
    #     class_mapping = {0: 'b', 1: 'g'}
    #     new_df_y_train.loc[:, 'Class'] = y_train['Class'].map(class_mapping)
    #     new_df_y_test.loc[:, 'Class'] = y_test['Class'].map(class_mapping)

    input_shape = 0
    for i in range(len(party_list)):
        input_shape += len(party_list[i].weights)

    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history, input_shape, size_of_transfer_data, test_precision_history, test_recall_history


def train_HE_binary_classification(dataset_name, n_epochs, party_list, server_list, main_server, X_train, y_train,
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

            smashed_list = []
            for i in range(len(party_list)):
                smashed_list.append(party_list[i].forward_pass(problem='classification'))

            smashed_numbers = []
            for i in range(len(smashed_list)):
                smashed_numbers.append(smashed_list[i][0])

            encrypted_functional = []
            if config.type_HE:
                encrypted_numbers = [ts.ckks_vector(config.context, [num]) for num in smashed_numbers]
            elif config.type_paillier:
                for i in range(len(party_list)):
                    encrypted_functional.append(config.public_key.encrypt(smashed_numbers[i]))
            elif config.type_DP:
                epsilon = 0.5  # Privacy budget
                laplace_mech = Laplace(epsilon=epsilon, sensitivity=1)

            main_server.reset()

            if config.type_HE:

                if n_data == 0 and epoch == 0:
                    for i in range(len(encrypted_numbers)):
                        size_of_transfer_data += sys.getsizeof(encrypted_numbers[i])

                main_server_error = main_server.calculate_HE_loss(encrypted_numbers)

            elif config.type_paillier:

                if n_data == 0 and epoch == 0:
                    for i in range(len(encrypted_functional)):
                        size_of_transfer_data += sys.getsizeof(encrypted_functional[i])

                main_server_error = main_server.calculate_paillier_loss(encrypted_functional)

            elif config.type_DP:

                if n_data == 0 and epoch == 0:
                    for i in range(len(smashed_numbers)):
                        size_of_transfer_data += sys.getsizeof(smashed_numbers[i])

                main_server_error = main_server.calculate_DP_loss(smashed_numbers, laplace_mech)

            func.parties_get_error(party_list, main_server_error)
            if n_data == 0 and epoch == 0:
                for i in range(len(party_list)):
                    if config.type_paillier:
                        temp_enc = config.public_key.encrypt(main_server_error)
                        size_of_transfer_data += sys.getsizeof(temp_enc)
                        size_of_transfer_data += sys.getsizeof(config.context)
                    elif config.type_HE:
                        temp_enc = [ts.ckks_vector(config.context, [num]) for num in [main_server_error]]
                        size_of_transfer_data += sys.getsizeof(temp_enc)
                        size_of_transfer_data += sys.getsizeof(config.context)
                    else:
                        size_of_transfer_data += sys.getsizeof(main_server_error)

            # for party in party_list:
            #     party.update_weights()

            for i in range(len(party_list)):
                party_list[i].get_batch_error(main_server_error)

            if n_data % config.batch_size == config.batch_size - 1:
                for party in party_list:
                    party.update_weights_batch(config.batch_size)
                    party.reset_batch_errors()

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

        test_accuracy, test_loss, test_precision, test_recall = test_HE_binary_classification(len(party_list),
                                                                                              X_test, y_test,
                                                                                              parties_coefs,
                                                                                              parties_biases, )

        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        test_precision_history.append(test_precision)
        test_recall_history.append(test_recall)

        func.parties_reset(party_list)
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


def test_HE_binary_classification(n_parties, X_test, y_test, party_coefs, party_biases):
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
            smashed_list.append(parties[i].forward_pass(problem='classification'))

        smashed_numbers = []
        for i in range(len(smashed_list)):
            smashed_numbers.append(smashed_list[i][0])

        encrypted_functional = []
        if config.type_HE:
            encrypted_numbers = [ts.ckks_vector(config.context, [num]) for num in smashed_numbers]
        elif config.type_paillier:
            for i in range(n_parties):
                encrypted_functional.append(config.public_key.encrypt(smashed_numbers[i]))
        elif config.type_DP:
            epsilon = 0.5  # Privacy budget
            laplace_mech = Laplace(epsilon=epsilon, sensitivity=1)

        label_for_test = y_test.loc[count_test_data].to_numpy()[0]
        y_label.append(label_for_test)

        if config.type_HE:
            encrypted_sum = encrypted_numbers[0]
            for enc_num in encrypted_numbers[1:]:
                encrypted_sum += enc_num

            decrypted_sum = np.float64(encrypted_sum.decrypt()[0])
            a = func.sigmoid(decrypted_sum)
            test_loss_list.append(abs(a - label_for_test))

        elif config.type_paillier:
            # encrypted_sum = encrypted_number1 + encrypted_number2
            encrypted_sum = 0
            for i in range(n_parties):
                encrypted_sum += encrypted_functional[i]
            decrypted_sum = config.private_key.decrypt(encrypted_sum)
            a = func.sigmoid(decrypted_sum)
            test_loss_list.append(abs(a - label_for_test))

        elif config.type_DP:
            noisy_sum = sum(laplace_mech.randomise(value) for value in smashed_numbers)
            a = func.sigmoid(noisy_sum)
            test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            y_pred.append(a)
            if label_for_test == a:
                count_correct += 1
        else:
            a = 0
            y_pred.append(a)
            if label_for_test == a:
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)
    precision_weighted = precision_score(y_label, y_pred, average='weighted')
    recall_weighted = recall_score(y_label, y_pred, average='weighted')

    return accuracy, loss, precision_weighted, recall_weighted


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
                    encrypted_data, offset = func.encrypt_vector(party_list[i].give_data_for_round(), config.shared_key)
                    offset_list.append(offset)
                    encrypted_data_list.append(encrypted_data)

                main_server.add_to_encrypted_data(encrypted_data_list, offset_list)

            if epoch != 0:
                for i in range(len(party_list)):
                    party_list[i].give_data_for_round()

            main_server.reset()
            main_server_error = main_server.calculate_FE_loss(party_list)

            if n_data == 0 and epoch == 0:
                for i in range(len(party_list)):
                    size_of_transfer_data += sys.getsizeof(encrypted_data_list[i])
                    # size_of_transfer_data += sys.getsizeof(offset_list[i])
                    # size_of_transfer_data += sys.getsizeof(party_list[i].weights)

                    # size_of_transfer_data += sys.getsizeof(main_server_error)
                    size_of_transfer_data += 32

            func.parties_get_error(party_list, main_server_error)

            for i in range(len(party_list)):
                party_list[i].get_batch_error(main_server_error)

            # for party in party_list:
            #     party.update_weights()

            if n_data % config.batch_size == config.batch_size - 1:
                for party in party_list:
                    party.update_weights_batch(config.batch_size)
                    party.reset_batch_errors()

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

        func.parties_reset(party_list)
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

        encrypted_data_list = []
        offset_list = []
        for i in range(len(party_list)):
            encrypted_data, offset = func.encrypt_vector(parties[i].give_data_for_round(), config.shared_key)
            offset_list.append(offset)
            encrypted_data_list.append(encrypted_data)

        intermediate_outputs = []
        for i in range(len(party_list)):
            intermediate_outputs.append(func.compute_inner_product(encrypted_data_list[i],
                                                                   party_list[i].weights,
                                                                   config.shared_key,
                                                                   offset=offset_list[i]))

        label_for_test = y_test.loc[count_test_data].to_numpy()[0]
        y_label.append(label_for_test)

        FE_sum = np.sum(intermediate_outputs)
        a = func.sigmoid(FE_sum)
        test_loss_list.append(abs(a - label_for_test))

        if a > 0.5:
            a = 1
            y_pred.append(a)
            if label_for_test == a:
                count_correct += 1
        else:
            a = 0
            y_pred.append(a)
            if label_for_test == a:
                count_correct += 1

        count_test_data += 1

    accuracy = count_correct / count_test_data
    loss = np.average(test_loss_list)
    precision_weighted = precision_score(y_label, y_pred, average='weighted')
    recall_weighted = recall_score(y_label, y_pred, average='weighted')

    return accuracy, loss, precision_weighted, recall_weighted
