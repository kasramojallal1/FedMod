import numpy as np
import pandas as pd
import math

import config

k_value = config.k_value


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    def calculate_loss(self, problem):

        if problem == 'regression':
            sum_data = np.sum(self.data)
            self.error = math.fmod(sum_data, k_value)

        elif problem == 'classification':
            self.correct = None
            label_for_round = self.labels.loc[self.round]
            label_for_round = label_for_round.to_numpy()
            label_for_round = label_for_round[0]

            sum_data = np.sum(self.data)
            sum_data = math.fmod(sum_data, k_value)
            sum_data = np.float64(sum_data)

            # print(sum_data)

            a = sigmoid(sum_data)
            self.error = a - label_for_round

            if a > 0.5:
                a = 1
                if label_for_round == a:
                    self.correct = 1
                else:
                    self.correct = 0
            else:
                a = 0
                if label_for_round == a:
                    self.correct = 1
                else:
                    self.correct = 0

            self.round += 1

    def calculate_HE_loss(self, encrypted_numbers):

        self.correct = None
        label_for_round = self.labels.loc[self.round]
        label_for_round = label_for_round.to_numpy()
        label_for_round = label_for_round[0]

        encrypted_sum = encrypted_numbers[0]
        for enc_num in encrypted_numbers[1:]:
            encrypted_sum += enc_num

        decrypted_sum = encrypted_sum.decrypt()[0]
        decrypted_sum = np.float64(decrypted_sum)
        # print(f"Decrypted sum: {decrypted_sum}")

        a = sigmoid(decrypted_sum)
        self.error = a - label_for_round

        if a > 0.5:
            a = 1
            if label_for_round == a:
                self.correct = 1
            else:
                self.correct = 0
        else:
            a = 0
            if label_for_round == a:
                self.correct = 1
            else:
                self.correct = 0

        self.round += 1

        return self.error


    def calculate_paillier_loss(self, encrypted_numbers):
        self.correct = None
        label_for_round = self.labels.loc[self.round]
        label_for_round = label_for_round.to_numpy()
        label_for_round = label_for_round[0]

        encrypted_sum = encrypted_numbers[0] + encrypted_numbers[1]
        decrypted_sum = config.private_key.decrypt(encrypted_sum)

        a = sigmoid(decrypted_sum)
        self.error = a - label_for_round

        if a > 0.5:
            a = 1
            if label_for_round == a:
                self.correct = 1
            else:
                self.correct = 0
        else:
            a = 0
            if label_for_round == a:
                self.correct = 1
            else:
                self.correct = 0

        self.round += 1

        return self.error


    def calculate_DP_loss(self, smashed_numbers, laplace_mech):

        self.correct = None
        label_for_round = self.labels.loc[self.round]
        label_for_round = label_for_round.to_numpy()
        label_for_round = label_for_round[0]

        noisy_sum = sum(laplace_mech.randomise(value) for value in smashed_numbers)

        a = sigmoid(noisy_sum)
        self.error = a - label_for_round

        if a > 0.5:
            a = 1
            if label_for_round == a:
                self.correct = 1
            else:
                self.correct = 0
        else:
            a = 0
            if label_for_round == a:
                self.correct = 1
            else:
                self.correct = 0

        self.round += 1

        return self.error




    def calculate_multi_loss(self, number_of_classes):
        self.correct = None

        label_for_round = self.labels.loc[self.round]
        label_for_round = label_for_round.to_numpy()
        label_for_round = label_for_round[0]

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


