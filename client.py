import numpy as np
import secrets
import math
import socket

import config


class Client:
    def __init__(self, name, weights, bias, data, lead, labels=None, error=None):
        self.name = name
        self.weights = weights
        self.bias = bias
        self.data = data
        self.lead = lead
        self.labels = labels
        self.error = error
        self.round = 0
        self.error_list = []

    def forward_pass(self, problem):

        if problem == 'regression':

            label_for_round = None
            if self.lead == 1:
                label_for_round = self.labels.loc[self.round]
                label_for_round = label_for_round.to_numpy()
                label_for_round = label_for_round[0]

            data_for_round = self.data.loc[self.round]
            data_for_round = data_for_round.to_numpy()

            smashed_data = np.dot(data_for_round, self.weights)
            smashed_data = smashed_data + self.bias

            if self.lead == 1:
                smashed_data = smashed_data - label_for_round

            self.round += 1

            return smashed_data

        elif problem == 'classification':
            data_for_round = self.data.loc[self.round]
            data_for_round = data_for_round.to_numpy()
            data_for_round = data_for_round.astype(float)

            smashed_data = np.dot(data_for_round, self.weights)
            smashed_data = smashed_data + self.bias

            self.round += 1

            return smashed_data

    def forward_pass_multi_classification(self, number_of_classes):
        data_for_round = self.data.loc[self.round]
        data_for_round = data_for_round.to_numpy()
        data_for_round = data_for_round.astype(float)

        smashed_data_list = []
        for i in range(number_of_classes):
            smashed = np.dot(data_for_round, self.weights[i])
            smashed = smashed + self.bias[i]
            smashed_data_list.append(smashed)

        self.round += 1

        return smashed_data_list

    def update_weights(self):
        data_x = self.data.loc[self.round - 1]
        data_x = data_x.to_numpy()
        gradients = data_x * self.error

        # self.weights = self.weights - (config.learning_rate * gradients)
        self.weights = self.weights - (config.learning_rate * (gradients + config.regularization_rate * self.weights))
        self.bias = self.bias - (config.learning_rate * self.error)

    def update_weights_multi(self, error_list, number_of_classes):
        data_x = self.data.loc[self.round - 1]
        data_x = data_x.to_numpy()

        for i in range(number_of_classes):
            gradient = data_x * error_list[i]
            self.weights[i] = self.weights[i] - (config.learning_rate * (gradient + config.regularization_rate * self.weights[i]))
            self.bias[i] = self.bias[i] - (config.learning_rate * error_list[i])

    def send_shares(self, share1, share2):
        pass

    # def update_weights_batch(self):
    #
    #     gradient_list = []
    #
    #     for i in range(len(self.error_list)):
    #         data_x = self.data.loc[i]
    #         data_x = data_x.to_numpy()
    #         gradient_list.append(data_x * self.error_list[i])
    #
    #     gradient_batch = gradient_list[0]
    #     for i in range(1, len(gradient_list)):
    #         gradient_batch += gradient_list[i]
    #
    #     gradient_batch = gradient_batch / len(gradient_list)
    #     self.weights = self.weights - (config.learning_rate * gradient_batch)
    #
    #     bias_gradient = np.sum(self.error_list) / len(self.error_list)
    #     self.bias = self.bias - (config.learning_rate * bias_gradient)
    #
    #     self.reset_error_list()

    def get_error(self, error):
        self.error = error

    def reset(self):
        self.round = 0

    def create_shares(self, y_value, k_value, random_coef):
        # print(f'Client {self.name}')
        # print(f'y_value: {y_value}')
        if y_value >= 0:
            sum_values = random_coef * k_value + y_value

        else:
            y_value *= -1
            sum_values = random_coef * k_value + y_value
            sum_values *= -1

        share1 = secrets.randbelow(k_value - (k_value - 10) + 1) + (k_value - 10)
        share2 = sum_values - share1

        return share1, share2

