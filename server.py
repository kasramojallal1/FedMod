import numpy as np
import pandas as pd


class Server:
    def __init__(self, name):
        self.name = name
        self.data = []

    def sum_data(self):
        return np.sum(self.data)

    def send_data(self):
        pass

    def get_from_client(self, data):
        self.data.append(data)

    def get_from_main_server(self, data):
        return data

    def reset(self):
        self.data = []
