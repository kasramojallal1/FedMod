import config
import nodes

import numpy as np
import os
import plotly.graph_objects as go

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


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


def derive_shared_key(private_key, public_key):
    shared_key = private_key.exchange(ec.ECDH(), public_key)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
        backend=default_backend()
    ).derive(shared_key)
    return derived_key


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def parties_reset(party_list):
    for party in party_list:
        party.reset()


def get_label_for_round(labels, round_n):
    label_for_round = labels.loc[round_n]
    label_for_round = label_for_round.to_numpy()
    label_for_round = label_for_round[0]

    return label_for_round


def check_correct_binary(sigmoid_result, label):
    if sigmoid_result > 0.5:
        predict = 1
    else:
        predict = 0

    if predict == label:
        return 1
    else:
        return 0


def create_graphs_classification(history, dataset_name, type_name, name_list, file_path):
    file_path = file_path + f'-{type_name}.png'

    color_list = ['darkorange', 'limegreen', 'purple', 'blue']
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
    fig1.write_image(file_path, width=1280, height=720, scale=2)
    # fig1.show()


def draw_graphs(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list,
                test_precision_list, test_recall_list, dataset_name, name_list, file_path):
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
    # create_graphs_classification(history=test_precisions, dataset_name=dataset_name, type_name='Precision',
    #                              name_list=name_list, file_path=file_path)
    # create_graphs_classification(history=test_recalls, dataset_name=dataset_name, type_name='Recall',
    #                              name_list=name_list, file_path=file_path)


def get_sets_of_entities(n_sets, problem_type, n_classes, X_train, y_train):
    party_sets = []
    server_sets = []
    main_server_sets = []

    if problem_type == 'binary':
        for i in range(n_sets):
            party_list_n, server_list_n, main_server_n = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                            'binary-classification', 2, X_train, y_train)
            party_sets.append(party_list_n)
            server_sets.append(server_list_n)
            main_server_sets.append(main_server_n)

    elif problem_type == 'multi':
        for i in range(n_sets):
            party_list_n, server_list_n, main_server_n = nodes.create_nodes(config.n_parties, config.n_servers,
                                                                            'multi-classification', n_classes, X_train, y_train)
            party_sets.append(party_list_n)
            server_sets.append(server_list_n)
            main_server_sets.append(main_server_n)

    return party_sets, server_sets, main_server_sets


def print_results(name_list, accuracy_list, loss_list, precision_list, recall_list, time_list, size_transfer_list,
                  file_path, dataset_name):
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
        # size_transfer_list[i] = size_transfer_list[i] / 1024
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
            file.write(f'{name_list[i]} Data Transfer: {size_transfer_list[i]} Bytes\n')
        file.write('--------------------------------------------\n')

        file.write(f'Dataset Name: {dataset_name}\n')
        file.write(f'Learning Rate: {config.learning_rate}\n')
        file.write(f'Regularization Rate: {config.regularization_rate}\n')
        file.write(f'Batch Size: {config.batch_size}\n')

        file.write(f'K_value: {config.k_value}\n')
        file.write(f'N# Parties: {config.n_parties}\n')
        file.write(f'N# Servers: {config.n_servers}\n')

        file.write(f'Poly Modulus Degree: {config.poly_mod_degree}\n')
        file.write(f'Coeff Mod Bit Sizes: {config.coeff_mod_bit_sizes}\n')
        file.write(f'Global Scale: {config.context.global_scale}\n')
        file.write('--------------------------------------------\n')
