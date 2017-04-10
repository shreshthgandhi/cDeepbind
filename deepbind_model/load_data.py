import numpy as np


def load_data():
    with np.load("deepbind_data.npz") as inf:
        data_training, labels_training, data_validation, labels_validation = \
            inf["data_training"], inf["labels_training"], \
            inf["data_validation"], inf["labels_validation"]

    training_cases = data_training.shape[0]
    validation_cases = data_validation.shape[0]
    seq_length = data_training.shape[1]
    m = 16  # Tunable Motif length
    d = 10  # Number of tunable motifs
    m2 = 1  # Filter size for 2 conv net

    data_one_hot_training = np.zeros((training_cases, seq_length, 4))
    data_one_hot_validation = np.zeros((validation_cases, seq_length, 4))

    for i, case in enumerate(data_training):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                data_one_hot_training[i, j, 0] = 1
            elif nuc == 'G':
                data_one_hot_training[i, j, 1] = 1
            elif nuc == 'C':
                data_one_hot_training[i, j, 2] = 1
            elif nuc == 'T':
                data_one_hot_training[i, j, 3] = 1

    for i, case in enumerate(data_validation):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                data_one_hot_validation[i, j, 0] = 1
            elif nuc == 'G':
                data_one_hot_validation[i, j, 1] = 1
            elif nuc == 'C':
                data_one_hot_validation[i, j, 2] = 1
            elif nuc == 'T':
                data_one_hot_validation[i, j, 3] = 1

    data_one_hot_training = data_one_hot_training - 0.25
    data_one_hot_validation = data_one_hot_validation - 0.25
    return(data_one_hot_training, labels_training,
           data_one_hot_validation, labels_validation,training_cases,validation_cases,seq_length)


if __name__ == "__main__":
    load_data()
