import numpy as np


def load_data_rnac(target_id_list=None, fold_filter='A'):
    infile_seq = open('./data/rnac/sequences.tsv')
    infile_target = open('./data/rnac/targets.tsv')
    seq_train = []
    seq_test = []
    target_train = []
    target_test = []
    seq_len_train = 0
    seq_len_test = 0

    target_names = infile_target.readline().split()
    if target_id_list is None:
        target_id_list = target_names
    target_cols_idx = np.zeros(len(target_id_list), dtype=int)
    # target_cols_idx = target_names.index(target_id_list)

    for i in range(len(target_id_list)):
        target_cols_idx[i] = target_names.index(target_id_list[i])
    infile_seq.readline()
    for line_seq in infile_seq:
        seq = line_seq.split('\t')[2].strip()
        line_target = infile_target.readline()
        target = [line_target.split('\t')[i] for i in target_cols_idx]
        fold = line_seq.split('\t')[0].strip()
        target_np = np.array(target, dtype=float)
        if np.any(np.isnan(target_np)):
            continue
        if fold in fold_filter:
            seq_train.append(seq)
            target_train.append(target)
            seq_len_train = max(seq_len_train, len(seq))
        else:
            seq_test.append(seq)
            target_test.append(target)
            seq_len_test = max(seq_len_test, len(seq))

    seq_train_enc = []
    for i in range(len(target_id_list)):
        seq_enc = np.ones((len(seq_train), seq_len_train, 4)) * 0.25
        for i, case in enumerate(seq_train):
            for j, nuc in enumerate(case):
                if nuc == 'A':
                    seq_enc[i, j] = np.array([1, 0, 0, 0])
                elif nuc == 'G':
                    seq_enc[i, j] = np.array([0, 1, 0, 0])
                elif nuc == 'C':
                    seq_enc[i, j] = np.array([0, 0, 1, 0])
                elif nuc == 'U':
                    seq_enc[i, j] = np.array([0, 0, 0, 1])
        seq_enc = seq_enc - 0.25
        seq_train_enc.append(seq_enc)

    seq_test_enc = []
    for i in range(len(target_id_list)):
        seq_enc = np.ones((len(seq_train), seq_len_train, 4)) * 0.25
        for i, case in enumerate(seq_train):
            for j, nuc in enumerate(case):
                if nuc == 'A':
                    seq_enc[i, j] = np.array([1, 0, 0, 0])
                elif nuc == 'G':
                    seq_enc[i, j] = np.array([0, 1, 0, 0])
                elif nuc == 'C':
                    seq_enc[i, j] = np.array([0, 0, 1, 0])
                elif nuc == 'U':
                    seq_enc[i, j] = np.array([0, 0, 0, 1])
        seq_enc = seq_enc - 0.25
        seq_test_enc.append(seq_enc)
    data_one_hot_training = np.array(seq_train_enc[0])
    data_one_hot_test = np.array(seq_test_enc[0])
    labels_training = np.array([i[0] for i in target_train], dtype=float)
    labels_test = np.array([i[0] for i in target_test], dtype=float)
    training_cases = data_one_hot_training.shape[0]
    test_cases = data_one_hot_test.shape[0]
    seq_length = data_one_hot_training.shape[1]

    train_remove = np.round(0.02 * training_cases).astype(int)
    test_remove = np.round(0.02 * test_cases).astype(int)
    train_ind = np.argpartition(labels_training, -train_remove)[-train_remove:]
    test_ind = np.argpartition(labels_test, -test_remove)[-test_remove:]
    train_clamp = np.min(labels_training[train_ind])
    test_clamp = np.min(labels_test[test_ind])
    labels_training[train_ind] = train_clamp
    labels_test[test_ind] = test_clamp

    # return (data_one_hot_training, data_one_hot_test,
    #         labels_training, labels_test,
    #         training_cases, test_cases, seq_length)

    np.savez('deepbind_RNAC', data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_validation=data_one_hot_test,
             labels_validation=labels_test, training_cases=training_cases,
             validation_cases=test_cases,
             seq_length=seq_length)


if __name__ == '__main__':
    load_data_rnac(target_id_list=['RNCMPT00168'])
