import os
import numpy as np

def load_data_rnac_s(path):
    pos_file = open(os.path.join(path, 'input_seq.fa'))
    neg_file = open(os.path.join(path,'bg_input_seq.fa'))
    seq_pos = []
    seq_neg = []
    seq_lens_pos = []
    labels_pos = []
    labels_neg = []
    seq_lens_neg = []
    struct_pos = []
    struct_neg = []
    struct_pos_file = open(os.path.join(path, 'input_seq_combined.txt'))
    struct_neg_file = open(os.path.join(path, 'bg_input_seq_combined.txt'))
    num_struct_classes = 5

    for line in pos_file:
        seq = pos_file.next().strip()
        seq_pos.append(seq)
        seq_lens_pos.append(len(seq))
        labels_pos.append(1.0)
    for line in neg_file:
        seq = neg_file.next().strip()
        seq_neg.append(seq)
        seq_lens_neg.append(len(seq))
        labels_neg.append(0.0)
    assert len(seq_pos) == len(seq_neg)

    seq_len_pad_pos = np.max(seq_lens_pos)
    seq_len_pad_neg = np.max(seq_lens_neg)
    for line in struct_pos_file:
        probs = np.ones([num_struct_classes,seq_len_pad_pos]) * (1.0/ num_struct_classes)
        for i  in range(5):
            values_line = struct_pos_file.next().strip()
            values = np.array(map(np.float32, values_line.split('\t')))
            probs[i,0:values.shape[0]] = values
        struct_pos.append(probs)

    for line in struct_neg_file:
        probs = np.ones([num_struct_classes,seq_len_pad_pos]) * (1.0/ num_struct_classes)
        for i  in range(5):
            values_line = struct_neg_file.next().strip()
            values = np.array(map(np.float32, values_line.split('\t')))
            probs[i,0:values.shape[0]] = values
        struct_neg.append(probs)
        
    assert len(struct_neg) == len(struct_pos)
    assert seq_len_pad_pos == seq_len_pad_neg
    assert len(struct_neg) == len(seq_neg)

    seq_enc_pos = np.ones((len(seq_pos), seq_len_pad_pos, 4)) * 0.25
    for i, case in enumerate(seq_pos):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc_pos[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc_pos[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc_pos[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc_pos[i, j] = np.array([0, 0, 0, 1])

    seq_enc_neg = np.ones((len(seq_neg), seq_len_pad_neg, 4)) * 0.25
    for i, case in enumerate(seq_pos):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc_neg[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc_neg[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc_neg[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc_neg[i, j] = np.array([0, 0, 0, 1])
    seq_enc_pos -= 0.25
    seq_enc_neg -= 0.25
    struct_pos  = np.array(struct_pos, dtype=np.float32) - (1.0/num_struct_classes)
    struct_neg = np.array(struct_neg, dtype=np.float32) - (1.0/num_struct_classes)
    train_size = int(0.9 * len(seq_pos))
    test_size = len(seq_pos) - train_size
    
    data_one_hot_training = []
    data_one_hot_test = []
    labels_training = []
    labels_test = []
    structures_train = []
    structures_test = []
    seq_len_train = []
    seq_len_test = []
    
    for i in range(train_size):
        data_one_hot_training.append(seq_enc_pos[i])
        data_one_hot_training.append(seq_enc_neg[i])
        structures_train.append(struct_pos[i])
        structures_train.append(struct_neg[i])
        seq_len_train.append(seq_lens_pos[i])
        seq_len_train.append(seq_lens_neg[i])
        labels_training.append(labels_pos[i])
        labels_training.append(labels_neg[i])
        
    for i in range(train_size, len(seq_pos)):
        data_one_hot_test.append(seq_enc_pos[i])
        data_one_hot_test.append(seq_enc_neg[i])
        structures_test.append(struct_pos[i])
        structures_test.append(struct_neg[i])
        seq_len_test.append(seq_lens_pos[i])
        seq_len_test.append(seq_lens_neg[i])
        labels_test.append(labels_pos[i])
        labels_test.append(labels_neg[i])
                
    if not(os.path.exists(os.path.join(path,'npz_archives'))):
        os.makedirs(os.path.join(path,'npz_archives'))
    save_path = os.path.join(path,'npz_archives','SLBP_rnacs.npz')

    np.savez(save_path,
             data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_test=data_one_hot_test,
             labels_test=labels_test, training_cases=train_size,
             test_cases=test_size,
             structures_train=structures_train,
             structures_test=structures_test,
             seq_len_train=seq_len_pad_pos,
             seq_len_test=seq_len_pad_pos,
             seq_length=seq_len_pad_pos,
             )

load_data_rnac_s('../data/rnac_s')