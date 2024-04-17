import client
import server
import main_server

import numpy as np


def create_nodes_1(X_train, y_train):
    data_party1 = X_train.iloc[:, :7]
    weights_party1 = np.random.randint(-4, 4, 7)
    bias_party1 = np.random.randint(-4, 4, 1)

    data_party2 = X_train.iloc[:, 7:]
    weights_party2 = np.random.randint(-4, 4, 6)
    bias_party2 = np.random.randint(-4, 4, 1)

    party1 = client.Client(name='party1',
                           weights=weights_party1,
                           bias=bias_party1,
                           data=data_party1,
                           lead=1,
                           labels=y_train)
    party2 = client.Client(name='party2',
                           weights=weights_party2,
                           bias=bias_party2,
                           data=data_party2,
                           lead=0)

    server1 = server.Server('server1')
    server2 = server.Server('server2')
    main_server1 = main_server.MainServer('main_server')

    return party1, party2, server1, server2, main_server1


def create_nodes_2(X_train, y_train):
    data_party1 = X_train.iloc[:, :7]
    weights_party1 = np.random.uniform(-1, 1, 7)
    bias_party1 = np.random.uniform(-1, 1, 1)

    data_party2 = X_train.iloc[:, 7:]
    weights_party2 = np.random.uniform(-1, 1, 6)
    bias_party2 = np.random.uniform(-1, 1, 1)

    party1 = client.Client(name='party1',
                           weights=weights_party1,
                           bias=bias_party1,
                           data=data_party1,
                           lead=0)

    party2 = client.Client(name='party2',
                           weights=weights_party2,
                           bias=bias_party2,
                           data=data_party2,
                           lead=0)

    server1 = server.Server('server1')
    server2 = server.Server('server2')
    main_server1 = main_server.MainServer('main_server', labels=y_train)

    return party1, party2, server1, server2, main_server1


def create_nodes_3(X_train, y_train, column_share):
    data_party1 = X_train.iloc[:, :column_share[0]]
    weights_party1 = np.random.uniform(-1, 1, column_share[0])
    bias_party1 = np.random.uniform(-1, 1, 1)

    data_party2 = X_train.iloc[:, column_share[0]:]
    weights_party2 = np.random.uniform(-1, 1, column_share[1])
    bias_party2 = np.random.uniform(-1, 1, 1)

    party1 = client.Client(name='party1',
                           weights=weights_party1,
                           bias=bias_party1,
                           data=data_party1,
                           lead=0)

    party2 = client.Client(name='party2',
                           weights=weights_party2,
                           bias=bias_party2,
                           data=data_party2,
                           lead=0)

    server1 = server.Server('server1')
    server2 = server.Server('server2')
    main_server1 = main_server.MainServer('main_server', labels=y_train)

    return party1, party2, server1, server2, main_server1


def create_nodes_4(X_train, y_train, number_of_clients, problem_type, number_of_classes):
    n_features = X_train.shape[1]
    column_share = n_features // number_of_clients
    extra_columns = n_features % number_of_clients

    if problem_type == 'classification':
        for i in range(number_of_clients):
            if i == 0:
                data_party1 = X_train.iloc[:, :column_share + extra_columns]
                weights_party1 = np.random.uniform(-1, 1, column_share + extra_columns)
                bias_party1 = np.random.uniform(-1, 1, 1)
            else:
                data_party2 = X_train.iloc[:, column_share + extra_columns:]
                weights_party2 = np.random.uniform(-1, 1, column_share)
                bias_party2 = np.random.uniform(-1, 1, 1)

    elif problem_type == 'multi-classification':
        for i in range(number_of_clients):
            if i == 0:
                data_party1 = X_train.iloc[:, :column_share + extra_columns]
                weights_party1 = []
                bias_party1 = []
                for j in range(number_of_classes):
                    weights_party1.append(np.random.uniform(-1, 1, column_share + extra_columns))
                    bias_party1.append(np.random.uniform(-1, 1, 1))

            else:
                data_party2 = X_train.iloc[:, column_share + extra_columns:]
                weights_party2 = []
                bias_party2 = []
                for j in range(number_of_classes):
                    weights_party2.append(np.random.uniform(-1, 1, column_share))
                    bias_party2.append(np.random.uniform(-1, 1, 1))

    for i in range(number_of_clients):
        if i == 0:
            party1 = client.Client(name='party1',
                                   weights=weights_party1,
                                   bias=bias_party1,
                                   data=data_party1,
                                   lead=0)
        else:
            party2 = client.Client(name='party2',
                                   weights=weights_party2,
                                   bias=bias_party2,
                                   data=data_party2,
                                   lead=0)

    server1 = server.Server('server1')
    server2 = server.Server('server2')
    main_server1 = main_server.MainServer('main_server', labels=y_train)

    return party1, party2, server1, server2, main_server1


import client
import server
import main_server

import numpy as np


def create_nodes(n_parties, n_servers, problem_type, n_classes, X_train, y_train):
    party_list = []
    server_list = []

    parties_data = []
    parties_weights = []
    parties_bias = []

    n_features = X_train.shape[1]
    column_share = n_features // n_parties
    extra_columns = n_features % n_parties

    if problem_type == 'binary-classification' or problem_type == 'regression':
        for i in range(n_parties):
            if i == n_parties - 1:
                data_party = X_train.iloc[:, i * column_share:]
                weights_party = np.random.uniform(-1, 1, column_share + extra_columns)
                bias_party = np.random.uniform(-1, 1, 1)
            else:
                data_party = X_train.iloc[:, i * column_share:(i + 1) * column_share]
                weights_party = np.random.uniform(-1, 1, column_share)
                bias_party = np.random.uniform(-1, 1, 1)

            parties_data.append(data_party)
            parties_weights.append(weights_party)
            parties_bias.append(bias_party)

    elif problem_type == 'multi-classification':
        for i in range(n_parties):
            if i == n_parties - 1:
                data_party = X_train.iloc[:, i * column_share:]
                weights_party = []
                bias_party = []
                for j in range(n_classes):
                    weights_party.append(np.random.uniform(-1, 1, column_share + extra_columns))
                    bias_party.append(np.random.uniform(-1, 1, 1))
            else:
                data_party = X_train.iloc[:, i * column_share:(i + 1) * column_share]
                weights_party = []
                bias_party = []
                for j in range(n_classes):
                    weights_party.append(np.random.uniform(-1, 1, column_share))
                    bias_party.append(np.random.uniform(-1, 1, 1))

            parties_data.append(data_party)
            parties_weights.append(weights_party)
            parties_bias.append(bias_party)

    if problem_type == 'regression':
        for i in range(n_parties):
            if i == 0:
                party_list.append(client.Client(name=f'party{i + 1}',
                                                weights=parties_weights[i],
                                                bias=parties_bias[i],
                                                data=parties_data[i],
                                                lead=1,
                                                labels=y_train))
            else:
                party_list.append(client.Client(name=f'party{i + 1}',
                                                weights=parties_weights[i],
                                                bias=parties_bias[i],
                                                data=parties_data[i],
                                                lead=0))

    else:
        for i in range(n_parties):
            party_list.append(client.Client(name=f'party{i + 1}',
                                            weights=parties_weights[i],
                                            bias=parties_bias[i],
                                            data=parties_data[i],
                                            lead=0))

    for i in range(n_servers):
        server_list.append(server.Server(f'server{i + 1}'))

    if problem_type != 'regression':
        output_main_server = main_server.MainServer('main_server', labels=y_train)
    else:
        output_main_server = main_server.MainServer('main_server')

    return party_list, server_list, output_main_server
