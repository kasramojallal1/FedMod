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
