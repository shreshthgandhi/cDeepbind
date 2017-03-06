
import argparse
import deepbind_model.utils as utils


# def load_data(target_id_list=None, fold_filter='A'):
#     # type: (object, object) -> object
#     infile_seq = open('../data/rnac/sequences.tsv')
#     infile_target = open('../data/rnac/targets.tsv')
#     seq_train = []
#     seq_test = []
#     target_train = []
#     target_test = []
#     exp_ids_train = []
#     exp_ids_test = []
#
#     infile_structA = open('../data/rnac/combined_profile_rnacA.txt')
#     infile_structB = open('../data/rnac/combined_profile_rnacB.txt')
#     structures_A = []
#     structures_B = []
#     seq_len_train = 41
#     num_struct_classes = 5
#
#     seq_len_train = 0
#     seq_len_test = 0
#
#     target_names = infile_target.readline().split()
#     if target_id_list is None:
#         target_id_list = target_names
#     target_cols_idx = np.zeros(len(target_id_list), dtype=int)
#     # target_cols_idx = target_names.index(target_id_list)
#
#     for i in range(len(target_id_list)):
#         target_cols_idx[i] = target_names.index(target_id_list[i])
#     infile_seq.readline()
#     for line_seq in infile_seq:
#         seq = line_seq.split('\t')[2].strip()
#         line_target = infile_target.readline()
#         target = [line_target.split('\t')[i] for i in target_cols_idx]
#         fold = line_seq.split('\t')[0].strip()
#         target_np = np.array(target, dtype=float)
#         if np.any(np.isnan(target_np)):
#             continue
#         if fold in fold_filter:
#             seq_train.append(seq)
#             target_train.append(target)
#             exp_ids_train.append(line_seq.split('\t')[1].strip())
#             seq_len_train = max(seq_len_train, len(seq))
#         else:
#             seq_test.append(seq)
#             target_test.append(target)
#             exp_ids_test.append(line_seq.split('\t')[1].strip())
#             seq_len_test = max(seq_len_test, len(seq))
#
#     iter_train = 0
#     seq_length = max(seq_len_test, seq_len_train)
#     iter_test = 0
#     for line_struct in infile_structA:
#         exp_id = line_struct.split('>')[1].strip()
#         exp_id_notnan = exp_ids_train[iter_train]
#         probs = np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes)
#         for i in range(5):
#             values_line = infile_structA.next().strip()
#             values = np.array(map(np.float32, values_line.split('\t')))
#             probs[i, 0:values.shape[0]] = values
#         if exp_id == exp_id_notnan:
#             structures_A.append(probs)
#             iter_train = iter_train + 1
#     if iter_train < len(exp_ids_train):
#         for i in range(iter_train, len(exp_ids_train)):
#             structures_A.append(np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes))
#
#     for line_struct in infile_structB:
#         exp_id = line_struct.split('>')[1].strip()
#         exp_id_notnan = exp_ids_test[iter_test]
#         probs = np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes)
#         for i in range(5):
#             values_line = infile_structB.next().strip()
#             values = np.array(map(np.float32, values_line.split('\t')))
#             probs[i, 0:values.shape[0]] = values
#         if exp_id == exp_id_notnan:
#             structures_B.append(probs)
#             iter_test = iter_test + 1
#     if iter_test < len(exp_ids_test):
#         for i in range(iter_test, len(exp_ids_test)):
#             structures_B.append(np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes))
#
#     seq_train_enc = []
#     for k in range(len(target_id_list)):
#         seq_enc = np.ones((len(seq_train), seq_length, 4)) * 0.25
#         for i, case in enumerate(seq_train):
#             for j, nuc in enumerate(case):
#                 if nuc == 'A':
#                     seq_enc[i, j] = np.array([1, 0, 0, 0])
#                 elif nuc == 'G':
#                     seq_enc[i, j] = np.array([0, 1, 0, 0])
#                 elif nuc == 'C':
#                     seq_enc[i, j] = np.array([0, 0, 1, 0])
#                 elif nuc == 'U':
#                     seq_enc[i, j] = np.array([0, 0, 0, 1])
#         seq_enc -= 0.25
#         seq_train_enc.append(seq_enc)
#
#     seq_test_enc = []
#     for k in range(len(target_id_list)):
#         seq_enc = np.ones((len(seq_test), seq_length, 4)) * 0.25
#         for i, case in enumerate(seq_test):
#             for j, nuc in enumerate(case):
#                 if nuc == 'A':
#                     seq_enc[i, j] = np.array([1, 0, 0, 0])
#                 elif nuc == 'G':
#                     seq_enc[i, j] = np.array([0, 1, 0, 0])
#                 elif nuc == 'C':
#                     seq_enc[i, j] = np.array([0, 0, 1, 0])
#                 elif nuc == 'U':
#                     seq_enc[i, j] = np.array([0, 0, 0, 1])
#         seq_enc = seq_enc - 0.25
#         seq_test_enc.append(seq_enc)
#     data_one_hot_training = np.array(seq_train_enc[0])
#     data_one_hot_test = np.array(seq_test_enc[0])
#     labels_training = np.array([i[0] for i in target_train], dtype=float)
#     labels_test = np.array([i[0] for i in target_test], dtype=float)
#     training_cases = data_one_hot_training.shape[0]
#     test_cases = data_one_hot_test.shape[0]
#     # seq_length = data_one_hot_training.shape[1]
#
#     structures_train = np.array(structures_A, dtype=np.float32)
#     structures_test = np.array(structures_B, dtype=np.float32)
#
#     train_remove = np.round(0.05 * training_cases).astype(int)
#     test_remove = np.round(0.05 * test_cases).astype(int)
#     train_ind = np.argpartition(labels_training, -train_remove)[-train_remove:]
#     test_ind = np.argpartition(labels_test, -test_remove)[-test_remove:]
#     train_clamp = np.min(labels_training[train_ind])
#     test_clamp = np.min(labels_test[test_ind])
#     labels_training[train_ind] = train_clamp
#     labels_test[test_ind] = test_clamp
#
#     # return (data_one_hot_training, data_one_hot_test,
#     #         labels_training, labels_test,
#     #         training_cases, test_cases, seq_length)
#     save_target = "../data/rnac/npz_archives/" +str(target_id_list[0])
#     np.savez(save_target, data_one_hot_training=data_one_hot_training,
#              labels_training=labels_training,
#              data_one_hot_test=data_one_hot_test,
#              labels_test=labels_test, training_cases=training_cases,
#              test_cases=test_cases,
#              structures_train=structures_train,
#              structures_test=structures_test,
#              seq_length=seq_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description= "Load data from RNA Compete experiments including structure")
    parser.add_argument("--all","-a", help = "load all proteins in respective files",
                        action="store_true")
    args = parser.parse_args()
    protein_list = ['RNCMPT00100',
                     'RNCMPT00101',
                     'RNCMPT00102',
                     'RNCMPT00103',
                     'RNCMPT00104',
                     'RNCMPT00105',
                     'RNCMPT00106',
                     'RNCMPT00107',
                     'RNCMPT00108',
                     'RNCMPT00109',
                     'RNCMPT00010',
                     'RNCMPT00110',
                     'RNCMPT00111',
                     'RNCMPT00112',
                     'RNCMPT00113',
                     'RNCMPT00114',
                     'RNCMPT00116',
                     'RNCMPT00117',
                     'RNCMPT00118',
                     'RNCMPT00119',
                     'RNCMPT00011',
                     'RNCMPT00120',
                     'RNCMPT00121',
                     'RNCMPT00122',
                     'RNCMPT00123',
                     'RNCMPT00124',
                     'RNCMPT00126',
                     'RNCMPT00127',
                     'RNCMPT00129',
                     'RNCMPT00012',
                     'RNCMPT00131',
                     'RNCMPT00132',
                     'RNCMPT00133',
                     'RNCMPT00134',
                     'RNCMPT00136',
                     'RNCMPT00137',
                     'RNCMPT00138',
                     'RNCMPT00139',
                     'RNCMPT00013',
                     'RNCMPT00140',
                     'RNCMPT00141',
                     'RNCMPT00142',
                     'RNCMPT00143',
                     'RNCMPT00144',
                     'RNCMPT00145',
                     'RNCMPT00146',
                     'RNCMPT00147',
                     'RNCMPT00148',
                     'RNCMPT00149',
                     'RNCMPT00014',
                     'RNCMPT00150',
                     'RNCMPT00151',
                     'RNCMPT00152',
                     'RNCMPT00153',
                     'RNCMPT00154',
                     'RNCMPT00155',
                     'RNCMPT00156',
                     'RNCMPT00157',
                     'RNCMPT00158',
                     'RNCMPT00159',
                     'RNCMPT00015',
                     'RNCMPT00160',
                     'RNCMPT00161',
                     'RNCMPT00162',
                     'RNCMPT00163',
                     'RNCMPT00164',
                     'RNCMPT00165',
                     'RNCMPT00166',
                     'RNCMPT00167',
                     'RNCMPT00168',
                     'RNCMPT00169',
                     'RNCMPT00016',
                     'RNCMPT00170',
                     'RNCMPT00171',
                     'RNCMPT00172',
                     'RNCMPT00173',
                     'RNCMPT00174',
                     'RNCMPT00175',
                     'RNCMPT00176',
                     'RNCMPT00177',
                     'RNCMPT00178',
                     'RNCMPT00179',
                     'RNCMPT00017',
                     'RNCMPT00180',
                     'RNCMPT00181',
                     'RNCMPT00182',
                     'RNCMPT00183',
                     'RNCMPT00184',
                     'RNCMPT00185',
                     'RNCMPT00186',
                     'RNCMPT00187',
                     'RNCMPT00018',
                     'RNCMPT00197',
                     'RNCMPT00199',
                     'RNCMPT00019',
                     'RNCMPT00001',
                     'RNCMPT00200',
                     'RNCMPT00202',
                     'RNCMPT00203',
                     'RNCMPT00205',
                     'RNCMPT00206',
                     'RNCMPT00209',
                     'RNCMPT00020',
                     'RNCMPT00212',
                     'RNCMPT00215',
                     'RNCMPT00216',
                     'RNCMPT00217',
                     'RNCMPT00218',
                     'RNCMPT00219',
                     'RNCMPT00021',
                     'RNCMPT00220',
                     'RNCMPT00223',
                     'RNCMPT00224',
                     'RNCMPT00225',
                     'RNCMPT00226',
                     'RNCMPT00228',
                     'RNCMPT00229',
                     'RNCMPT00022',
                     'RNCMPT00230',
                     'RNCMPT00232',
                     'RNCMPT00234',
                     'RNCMPT00235',
                     'RNCMPT00236',
                     'RNCMPT00237',
                     'RNCMPT00238',
                     'RNCMPT00239',
                     'RNCMPT00023',
                     'RNCMPT00240',
                     'RNCMPT00241',
                     'RNCMPT00245',
                     'RNCMPT00246',
                     'RNCMPT00248',
                     'RNCMPT00249',
                     'RNCMPT00024',
                     'RNCMPT00251',
                     'RNCMPT00252',
                     'RNCMPT00253',
                     'RNCMPT00254',
                     'RNCMPT00255',
                     'RNCMPT00256',
                     'RNCMPT00257',
                     'RNCMPT00258',
                     'RNCMPT00259',
                     'RNCMPT00025',
                     'RNCMPT00261',
                     'RNCMPT00262',
                     'RNCMPT00263',
                     'RNCMPT00265',
                     'RNCMPT00268',
                     'RNCMPT00269',
                     'RNCMPT00026',
                     'RNCMPT00270',
                     'RNCMPT00272',
                     'RNCMPT00273',
                     'RNCMPT00274',
                     'RNCMPT00278',
                     'RNCMPT00279',
                     'RNCMPT00027',
                     'RNCMPT00280',
                     'RNCMPT00281',
                     'RNCMPT00282',
                     'RNCMPT00283',
                     'RNCMPT00284',
                     'RNCMPT00285',
                     'RNCMPT00287',
                     'RNCMPT00288',
                     'RNCMPT00289',
                     'RNCMPT00028',
                     'RNCMPT00291',
                     'RNCMPT00029',
                     'RNCMPT00002',
                     'RNCMPT00031',
                     'RNCMPT00032',
                     'RNCMPT00033',
                     'RNCMPT00034',
                     'RNCMPT00035',
                     'RNCMPT00036',
                     'RNCMPT00037',
                     'RNCMPT00038',
                     'RNCMPT00039',
                     'RNCMPT00003',
                     'RNCMPT00040',
                     'RNCMPT00041',
                     'RNCMPT00042',
                     'RNCMPT00043',
                     'RNCMPT00044',
                     'RNCMPT00045',
                     'RNCMPT00046',
                     'RNCMPT00047',
                     'RNCMPT00049',
                     'RNCMPT00004',
                     'RNCMPT00050',
                     'RNCMPT00051',
                     'RNCMPT00052',
                     'RNCMPT00053',
                     'RNCMPT00054',
                     'RNCMPT00055',
                     'RNCMPT00056',
                     'RNCMPT00057',
                     'RNCMPT00058',
                     'RNCMPT00059',
                     'RNCMPT00005',
                     'RNCMPT00060',
                     'RNCMPT00061',
                     'RNCMPT00062',
                     'RNCMPT00063',
                     'RNCMPT00064',
                     'RNCMPT00065',
                     'RNCMPT00066',
                     'RNCMPT00067',
                     'RNCMPT00068',
                     'RNCMPT00069',
                     'RNCMPT00006',
                     'RNCMPT00070',
                     'RNCMPT00071',
                     'RNCMPT00072',
                     'RNCMPT00073',
                     'RNCMPT00074',
                     'RNCMPT00075',
                     'RNCMPT00076',
                     'RNCMPT00077',
                     'RNCMPT00078',
                     'RNCMPT00079',
                     'RNCMPT00007',
                     'RNCMPT00080',
                     'RNCMPT00081',
                     'RNCMPT00082',
                     'RNCMPT00083',
                     'RNCMPT00084',
                     'RNCMPT00085',
                     'RNCMPT00086',
                     'RNCMPT00087',
                     'RNCMPT00088',
                     'RNCMPT00089',
                     'RNCMPT00008',
                     'RNCMPT00090',
                     'RNCMPT00091',
                     'RNCMPT00093',
                     'RNCMPT00094',
                     'RNCMPT00095',
                     'RNCMPT00096',
                     'RNCMPT00097',
                     'RNCMPT00099',
                     'RNCMPT00009']
    if args.all:
        for protein in protein_list:
            print("Loading %s"%(protein))
            utils.load_data(target_id_list=[protein])
    else:
        utils.load_data(target_id_list=['RNCMPT00168'])
