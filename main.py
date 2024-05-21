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
dataset_2 = True
dataset_3 = False
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
        n_epochs = 360
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
        config.learning_rate = 0.001
        n_epochs = 150
        dataset_name = 'parkinson'
        problem_type = 'binary'
        n_classes = 2
        X_train, X_test, y_train, y_test = preprocess.setup_dataframe_5(dataset_address_5)

    name_list = ['FedV', 'FedMod', 'Baseline']
    n_sets = len(name_list)
    file_write_path_figures = f"results/{dataset_name}/figure-p{config.n_parties}"
    file_write_path_texts = f"results/{dataset_name}/report-p{config.n_parties}"

    for i in range(2, 3):
        config.n_parties = i

        party_sets, server_sets, main_server_sets = func.get_sets_of_entities(n_sets=n_sets, problem_type=problem_type,
                                                                              n_classes=n_classes, X_train=X_train,
                                                                              y_train=y_train)

        # fe_results, fe_scores, fe_extra, fe_time = run_fe(party_sets[0], server_sets[0], main_server_sets[0],
        #                                                   X_train, y_train, X_test, y_test, n_epochs, dataset_name)

        # he_results, he_extra, he_time = run_he(party_sets[1], server_sets[1], main_server_sets[1],
        #                                        X_train, y_train, X_test, y_test, n_epochs, dataset_name)

        fed_results, fed_scores, fed_extra, fed_time = model_run.run_fedmod(party_set=party_sets[2],
                                                                            server_set=server_sets[2],
                                                                            main_server_set=main_server_sets[2],
                                                                            X_train=X_train, y_train=y_train,
                                                                            X_test=X_test, y_test=y_test,
                                                                            n_epochs=n_epochs,
                                                                            dataset_name=dataset_name,
                                                                            problem_type=problem_type,
                                                                            n_classes=n_classes)

        baseline_results, baseline_scores, baseline_extra, baseline_time = model_run.run_baseline(X_train=X_train,
                                                                                                  X_test=X_test,
                                                                                                  y_train=y_train,
                                                                                                  y_test=y_test,
                                                                                                  input_shape=fed_extra[
                                                                                                      0],
                                                                                                  n_epochs=n_epochs,
                                                                                                  dataset_name=dataset_name,
                                                                                                  problem_type=problem_type,
                                                                                                  n_classes=n_classes)

        fe_results = fed_results
        fe_scores = fed_scores
        fe_extra = fed_extra
        fe_time = fed_time

        train_loss_list = [fe_results[0], fed_results[0], baseline_results[0]]
        train_accuracy_list = [fe_results[1], fed_results[1], baseline_results[1]]

        test_loss_list = [fe_results[2], fed_results[2], baseline_results[2]]
        test_accuracy_list = [fe_results[3], fed_results[3], baseline_results[3]]

        test_precision_list = [fe_scores[0], fed_scores[0], baseline_scores[0]]
        test_recall_list = [fe_scores[1], fed_scores[1], baseline_scores[1]]

        time_list = [fe_time, fed_time, baseline_time]
        size_transfer_list = [fe_extra[1], fed_extra[1], 0]

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
