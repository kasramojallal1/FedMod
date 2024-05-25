import preprocess
import nodes
import train_test
import config
import functions as func
import model_run

import os
import numpy as np
import pandas as pd
import random
import plotly.io as pio
import tensorflow as tf

pio.renderers.default = "browser"
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(43)
tf.random.set_seed(43)
random.seed(45)

dataset_address_1 = './datasets/boston_housing.csv'
dataset_address_2 = './datasets/heart.csv'
dataset_address_4 = './datasets/phishing.arff'
dataset_address_5 = './datasets/parkinson.arff'

dataset_1 = False
dataset_2 = False
dataset_3 = True
dataset_4 = False
dataset_5 = False

if __name__ == "__main__":

    if dataset_2:
        config.learning_rate = 0.01
        n_epochs = 40
        dataset_name = 'heart'
        problem_type = 'binary'
        n_classes = 2
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_2(dataset_address_2)

    elif dataset_3:
        config.learning_rate = 0.01
        n_epochs = 1
        dataset_name = 'ionosphere'
        problem_type = 'binary'
        n_classes = 2
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_3()

    elif dataset_4:
        config.learning_rate = 0.001
        n_epochs = 25
        dataset_name = 'phishing'
        problem_type = 'multi'
        n_classes = 3
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_4(dataset_address_4)

    elif dataset_5:
        config.learning_rate = 0.01
        n_epochs = 100
        dataset_name = 'parkinson'
        problem_type = 'binary'
        n_classes = 2
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_5(dataset_address_5)

    else:
        print("Dataset not found")
        exit()

    name_list = ['FedMod', 'Vanilla', 'FedV', 'HE', 'DP']
    n_sets = len(name_list)
    file_write_path_figures = f"results/{dataset_name}/figure-p{config.n_parties}"
    file_write_path_texts = f"results/{dataset_name}/report-p{config.n_parties}"

    for i in range(2, 3):
        config.n_parties = i

        party_sets, server_sets, main_server_sets = func.get_sets_of_entities(n_sets=n_sets,
                                                                              problem_type=problem_type,
                                                                              n_classes=n_classes, X_train=X_train,
                                                                              y_train=y_train)

        all_results = [
            model_run.run_fedmod(party_set=party_sets[0], server_set=server_sets[0],
                                 main_server_set=main_server_sets[0],
                                 X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_epochs=n_epochs,
                                 dataset_name=dataset_name, problem_type=problem_type, n_classes=n_classes),
            model_run.run_nosec(party_set=party_sets[1], server_set=server_sets[1],
                                main_server_set=main_server_sets[1],
                                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_epochs=n_epochs,
                                dataset_name=dataset_name, problem_type=problem_type, n_classes=n_classes),
            # model_run.run_baseline(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            #                        input_shape=config.nn_input_shape, n_epochs=n_epochs, dataset_name=dataset_name,
            #                        problem_type=problem_type, n_classes=n_classes),
            model_run.run_fe(party_set=party_sets[4], server_set=server_sets[4], main_server_set=main_server_sets[4],
                             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_epochs=n_epochs,
                             dataset_name=dataset_name),
            model_run.run_he(party_set=party_sets[2], server_set=server_sets[2], main_server_set=main_server_sets[2],
                             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_epochs=n_epochs,
                             dataset_name=dataset_name),
            model_run.run_dp(party_set=party_sets[3], server_set=server_sets[3], main_server_set=main_server_sets[3],
                             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_epochs=n_epochs,
                             dataset_name=dataset_name)
        ]

        train_loss_list = []
        train_accuracy_list = []
        test_loss_list = []
        test_accuracy_list = []
        test_precision_list = []
        test_recall_list = []
        size_transfer_list = []
        time_list = []

        for j in range(n_sets):
            train_loss_list.append(all_results[j][0])
            train_accuracy_list.append(all_results[j][1])
            test_loss_list.append(all_results[j][2])
            test_accuracy_list.append(all_results[j][3])
            test_precision_list.append(all_results[j][4])
            test_recall_list.append(all_results[j][5])
            size_transfer_list.append(all_results[j][6])
            time_list.append(all_results[j][7])

        func.draw_graphs(train_loss_list,
                         train_accuracy_list,
                         test_loss_list,
                         test_accuracy_list,
                         test_precision_list,
                         test_recall_list,
                         dataset_name,
                         name_list,
                         file_path=file_write_path_figures)

        func.print_results(name_list,
                           test_accuracy_list,
                           test_loss_list,
                           test_precision_list,
                           test_recall_list,
                           time_list,
                           size_transfer_list,
                           file_path=file_write_path_texts,
                           dataset_name=dataset_name)
