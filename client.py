import numpy as np
import secrets
import random

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
        self.batch_errors = []

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

    def give_data_for_round(self):
        data_for_round = self.data.loc[self.round]
        data_for_round = data_for_round.to_numpy()
        data_for_round = data_for_round.astype(float)

        self.round += 1

        return data_for_round

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

        self.weights = self.weights - (config.learning_rate * (gradients + config.regularization_rate * self.weights))
        self.bias = self.bias - (config.learning_rate * self.error)

    def update_weights_multi(self, error_list, number_of_classes):
        data_x = self.data.loc[self.round - 1]
        data_x = data_x.to_numpy()

        for i in range(number_of_classes):
            gradient = data_x * error_list[i]
            self.weights[i] = self.weights[i] - (config.learning_rate * (gradient + config.regularization_rate * self.weights[i]))
            self.bias[i] = self.bias[i] - (config.learning_rate * error_list[i])

    def update_weights_batch(self, batch_size):
        batch_data = self.data.loc[self.round - batch_size:self.round].to_numpy()

        gradients_list = []
        for i in range(batch_size):
            gradients = batch_data[i] * self.batch_errors[i]
            gradients_list.append(gradients)
        batch_gradient = 1/batch_size * np.sum(gradients_list, axis=0)

        self.weights = self.weights - (config.learning_rate * (batch_gradient + config.regularization_rate * self.weights))
        self.bias = self.bias - (config.learning_rate * self.error)

    def get_batch_error(self, error):
        self.batch_errors.append(error)

    def reset_batch_errors(self):
        self.batch_errors = []

    def send_shares(self, share1, share2):
        pass

    def get_error(self, error):
        self.error = error

    def reset(self):
        self.round = 0

    def create_shares(self, intermediate_output, k_value, random_coef):
        # if intermediate_output >= 0:
        #     sum_values = random_coef * k_value + intermediate_output
        #
        # else:
        #     intermediate_output *= -1
        #     sum_values = random_coef * k_value + intermediate_output
        #     sum_values *= -1
        #
        # share1 = secrets.randbelow(
        #     int(k_value) - (int(k_value) - 10) + 1) + (int(k_value) - 10) + random.uniform(-1, 1)
        # share2 = sum_values - share1
        #
        # return share1, share2

        if intermediate_output >= 0:
            temp = random_coef * k_value + intermediate_output
        else:
            intermediate_output *= -1
            temp = random_coef * k_value + intermediate_output
            temp *= -1

        share_list = []
        for i in range(config.n_servers - 1):
            share_list.append(np.int8(secrets.randbelow(
                int(k_value) - (int(k_value) - 10) + 1) + (int(k_value) - 10) + random.uniform(-1, 1)))

        share_list.append(temp - sum(share_list))
        share_list[-1] = np.float16(share_list[-1][0])

        return share_list

