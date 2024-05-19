import numpy as np
import pandas as pd
import math

import config
import train_test

k_value = config.k_value


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def check_correct_binary(sigmoid_result, label):
    if sigmoid_result > 0.5:
        predict = 1
    else:
        predict = 0

    if predict == label:
        return 1
    else:
        return 0


def get_label_for_round(labels, round_n):
    label_for_round = labels.loc[round_n]
    label_for_round = label_for_round.to_numpy()
    label_for_round = label_for_round[0]

    return label_for_round


class MainServer:
    def __init__(self, name, labels=None):
        self.name = name
        self.data = []
        self.error = None
        self.labels = labels
        self.round = 0
        self.correct = None
        self.multi_data = []
        self.error_multi = []

        self.enc_round = 0
        self.encrypted_data = []
        self.offset_list = []

    def calculate_loss(self, problem):

        if problem == 'regression':
            sum_data = np.sum(self.data)
            self.error = math.fmod(sum_data, k_value)

        elif problem == 'classification':
            self.correct = None
            label_for_round = get_label_for_round(self.labels, self.round)

            sum_data = np.sum(self.data)
            sum_data = math.fmod(sum_data, k_value)
            sum_data = np.float64(sum_data)

            a = sigmoid(sum_data)
            self.error = a - label_for_round
            self.correct = check_correct_binary(a, label_for_round)
            self.round += 1

    def calculate_HE_loss(self, encrypted_numbers):

        self.correct = None
        label_for_round = get_label_for_round(self.labels, self.round)

        encrypted_sum = encrypted_numbers[0]
        for enc_num in encrypted_numbers[1:]:
            encrypted_sum += enc_num

        decrypted_sum = encrypted_sum.decrypt()[0]
        decrypted_sum = np.float64(decrypted_sum)

        a = sigmoid(decrypted_sum)
        self.error = a - label_for_round
        self.correct = check_correct_binary(a, label_for_round)
        self.round += 1

        return self.error

    def calculate_paillier_loss(self, encrypted_numbers):
        self.correct = None
        label_for_round = get_label_for_round(self.labels, self.round)

        encrypted_sum = encrypted_numbers[0] + encrypted_numbers[1]
        decrypted_sum = config.private_key.decrypt(encrypted_sum)

        a = sigmoid(decrypted_sum)
        self.error = a - label_for_round
        self.correct = check_correct_binary(a, label_for_round)
        self.round += 1

        return self.error

    def calculate_DP_loss(self, smashed_numbers, laplace_mech):

        self.correct = None
        label_for_round = get_label_for_round(self.labels, self.round)

        noisy_sum = sum(laplace_mech.randomise(value) for value in smashed_numbers)

        a = sigmoid(noisy_sum)
        self.error = a - label_for_round
        self.correct = check_correct_binary(a, label_for_round)
        self.round += 1

        return self.error

    def add_to_encrypted_data(self, encrypted_data, offset):
        self.encrypted_data.append(encrypted_data)
        self.offset_list.append(offset)

    def reset_encrypted_round(self):
        self.enc_round = 0

    def calculate_FE_loss(self, party_list):

        intermediate_outputs = []
        for i in range(len(party_list)):
            intermediate_outputs.append(train_test.compute_inner_product(self.encrypted_data[self.enc_round][i],
                                                                         party_list[i].weights,
                                                                         config.shared_key,
                                                                         offset=self.offset_list[self.enc_round][i]))

        self.correct = None
        label_for_round = get_label_for_round(self.labels, self.round)

        sum_data = sum(intermediate_outputs)
        a = sigmoid(sum_data)

        self.error = a - label_for_round
        self.correct = check_correct_binary(a, label_for_round)
        self.round += 1

        return self.error

    def calculate_multi_loss(self, number_of_classes):
        self.correct = None
        label_for_round = self.labels.loc[self.round].to_numpy()[0]

        sigmoid_results = []
        for i in range(number_of_classes):
            sum_data = np.sum(self.multi_data[i])
            sum_data = math.fmod(sum_data, k_value)
            sum_data = np.float64(sum_data)

            a = sigmoid(sum_data)
            sigmoid_results.append(a)

        predict = np.argmax(sigmoid_results) + 1

        for i in range(1, number_of_classes + 1):
            if i == label_for_round:
                self.error_multi.append(sigmoid_results[i - 1] - 1)
            else:
                self.error_multi.append(sigmoid_results[i - 1] - 0)

        if predict == label_for_round:
            self.correct = 1
        else:
            self.correct = 0

        self.round += 1

    def send_data(self, server):
        pass

    def get_data(self, data):
        self.data.append(data)

    def get_multi_data(self, data):
        self.multi_data.append(data)

    def reset(self):
        self.data = []
        self.multi_data = []
        self.error_multi = []

    def reset_round(self):
        self.round = 0
