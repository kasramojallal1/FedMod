import time
import train_test


def run_fedmod(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name, problem_type, n_classes):
    start_time = time.time()

    if problem_type == 'binary':
        train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer, test_precision, test_recall = train_test.train_model_binary_classification(
            dataset_name=dataset_name, n_epochs=n_epochs,
            party_list=party_set, server_list=server_set, main_server=main_server_set,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    elif problem_type == 'multi':
        train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer, test_precision, test_recall = train_test.train_model_multi_classification(
            dataset_name=dataset_name, n_epochs=n_epochs,
            party_list=party_set, server_list=server_set, main_server=main_server_set,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, n_classes=n_classes)

    else:
        print("Problem type not supported")
        return

    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    algorithm_scores = [test_precision, test_recall]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, algorithm_scores, extra_results, time_taken


def run_he(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name):
    start_time = time.time()
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer = train_test.train_HE_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, extra_results, time_taken


def run_fe(party_set, server_set, main_server_set, X_train, y_train, X_test, y_test, n_epochs, dataset_name):
    start_time = time.time()
    train_loss, test_loss, train_accuracy, test_accuracy, input_shape, size_of_data_transfer, test_precision, test_recall = train_test.train_FE_binary_classification(
        dataset_name=dataset_name, n_epochs=n_epochs,
        party_list=party_set, server_list=server_set, main_server=main_server_set,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    algorithm_scores = [test_precision, test_recall]
    extra_results = [input_shape, size_of_data_transfer]
    time_taken = end_time - start_time

    return algorithm_results, algorithm_scores, extra_results, time_taken


def run_baseline(X_train, X_test, y_train, y_test, input_shape, n_epochs, dataset_name, problem_type, n_classes):
    start_time = time.time()

    if problem_type == 'binary':
        train_accuracy, test_accuracy, train_loss, test_loss, test_precision, test_recall = train_test.train_mlp_binary_baseline(
            n_epochs=n_epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            input_shape=input_shape, output_shape=1, dataset_name=dataset_name)
    elif problem_type == 'multi':
        train_accuracy, test_accuracy, train_loss, test_loss, test_precision, test_recall = train_test.train_mlp_multi_baseline(
            n_epochs=n_epochs, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            input_shape=input_shape, output_shape=n_classes, dataset_name=dataset_name, n_classes=n_classes)

    else:
        print("Problem type not supported")
        return

    end_time = time.time()

    algorithm_results = [train_loss, train_accuracy, test_loss, test_accuracy]
    algorithm_scores = [test_precision, test_recall]
    extra_results = [0, 0]
    time_taken = end_time - start_time

    return algorithm_results, algorithm_scores, extra_results, time_taken
