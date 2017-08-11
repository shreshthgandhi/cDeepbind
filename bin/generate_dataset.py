import argparse
import os.path
import subprocess

from deepbind_model.utils import load_data_clipseq_shorter
from deepbind_model.utils import load_data_rnac2009


def clip_struct_compute():
    save_folder = '../data/GraphProt_CLIP_sequences/structure_annotations/'
    clip_experiments = ['PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_FUS',
                        'ICLIP_HNRNPC', 'PARCLIP_HUR', 'PARCLIP_IGF2BP123', 'PTBv1',
                        'PARCLIP_PUM2', 'PARCLIP_QKI', 'CLIPSEQ_SFRS1', 'ICLIP_TIA1']
    for clip_experiment_name in clip_experiments:
        if not (os.path.isdir(save_folder + clip_experiment_name)):
            os.makedirs(save_folder + clip_experiment_name)
            # clip_experiment = os.path.join('../data/GraphProt_CLIP_sequences/', clip_experiment_name)
            # W = '180'
            # L = '160'
            # positives_file = clip_experiment + '.train.positives.fa'
            # negatives_file = clip_experiment + '.train.negatives.fa'
            # print("[*] Starting RNAplfold for " + clip_experiment)
            # E_process_pos = subprocess.call([
            #                                      "../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold -W " + W + " -L " + L + " -u 1 <" + positives_file + " >" + os.path.join(save_folder,clip_experiment_name,"E_profile_pos.txt")],
            #                                  shell=True)
            # H_process_pos = subprocess.call([
            #                                      "../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold -W " + W + " -L " + L + "-u 1 <" + positives_file + " >" + os.path.join(save_folder,clip_experiment_name,"H_profile_pos.txt")],
            #                                  shell=True)
            # I_process_pos = subprocess.call([
            #     "../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold -W " + W + " -L " + L + "-u 1 <" + positives_file + " >" +os.path.join(save_folder,clip_experiment_name,"I_profile_pos.txt")],
            #     shell=True)
            # M_process_pos = subprocess.call([
            #     "../RNAplfold_scripts/RNAplfold_scripts/M_RNAplfold -W " + W + " -L " + L + "-u 1  <" + positives_file + " >" +os.path.join(save_folder,clip_experiment_name,"M_profile_pos.txt")],
            #     shell=True)
            #
            # # pos_processes = [E_process_pos,H_process_pos,I_process_pos,M_process_pos]
            # # for p in pos_processes:
            # #     p.wait()
            #
            # E_process_neg = subprocess.call([
            #     "../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold -W " + W + " -L " + L + "-u 1" + " <" + negatives_file + " >" + os.path.join(save_folder,clip_experiment_name ,"E_profile_neg.txt")],
            #     shell=True)
            # H_process_neg = subprocess.call([
            #     "../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold -W " + W + " -L " + L + "-u 1" + " <" + negatives_file + " >" + os.path.join(save_folder,clip_experiment_name ,"H_profile_neg.txt")],
            #     shell=True)
            # I_process_neg = subprocess.call([
            #     "../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold -W " + W + " -L " + L + "-u 1" + " <" + negatives_file + " >" + os.path.join(save_folder,clip_experiment_name ,"I_profile_neg.txt")],
            #     shell=True)
            # M_process_neg = subprocess.call([
            #     "../RNAplfold_scripts/RNAplfold_scripts/M_RNAplfold -W " + W + " -L " + L + "-u 1" + " <" + negatives_file + " >" + os.path.join(save_folder, clip_experiment_name,"M_profile_neg.txt")],
            #     shell=True)
            #
            # # processes = [ E_process_neg, H_process_neg,I_process_neg, M_process_neg]
            # # for p in processes:
            # #     p.wait()
            # print("[*] Finished structure profiles for " + clip_experiment)
            # # subprocess.call(["python ../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py " + \
            # #                  save_folder + "E_profile_pos.txt " + save_folder + "H_profile_pos.txt " + \
            # #                  save_folder + "I_profile_pos.txt " + save_folder + "M_profile_pos.txt 1 " + \
            # #                  clip_experiment + "_pos.txt"], shell=True)
            # # subprocess.call(["python ../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py " + \
            # #                  save_folder + "E_profile_neg.txt " + save_folder + "H_profile_neg.txt " + \
            # #                  save_folder + "I_profile_neg.txt ", save_folder + "M_profile_neg.txt 1 " + \
            # #                  clip_experiment + "_neg.txt"], shell=True)
            # print("[*] Combined letter profiles for " + clip_experiment)


def rnac_2009_struct_compute():
    fasta_folder = '../data/rnac_2009/full/fasta'
    text_folder = '../data/rnac_2009/full/'
    save_folder = '../data/rnac_2009/full/structure_annotations/'
    seq_files = []
    recompute_fasta = True
    for text_file in os.listdir(text_folder):
        if not ('AB' in text_file) and ('data' in text_file):
            seq_files.append(text_file)
    if recompute_fasta:
        for seq_file_name in seq_files:
            count = 0
            with open(os.path.join(text_folder, seq_file_name), 'r') as seq_file:
                with open(os.path.join(fasta_folder, seq_file_name.replace('.txt', '') + '.fasta'), 'w') as fasta_file:
                    for line in seq_file:
                        fasta_file.write(
                            '>' + seq_file_name.replace('.txt', '_') + str(count) + '\n' + line.split('\t')[1])
                        count += 1
            print("[*] Generated FASTA file for " + seq_file_name)

    rnac_experiment_list = [file.replace('.txt', '.fasta') for file in seq_files]
    for rnac_experiment in rnac_experiment_list:
        rnac_fasta_file = os.path.join(fasta_folder, rnac_experiment)
        W = '29'
        L = '29'
        print("[*] Starting secondary structure prediction for " + rnac_experiment)
        E_process = subprocess.Popen([
                                         '../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold -W ' + W + ' -L ' + L + ' -u 1 <' + rnac_fasta_file + ' >' + os.path.join(
                                             save_folder, 'E_profile.txt')], shell=True)
        H_process = subprocess.Popen([
                                         '../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold -W ' + W + ' -L ' + L + ' -u 1 <' + rnac_fasta_file + ' >' + os.path.join(
                                             save_folder, 'H_profile.txt')], shell=True)
        I_process = subprocess.Popen([
                                         '../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold -W ' + W + ' -L ' + L + ' -u 1 <' + rnac_fasta_file + ' >' + os.path.join(
                                             save_folder, 'I_profile.txt')], shell=True)
        M_process = subprocess.Popen([
                                         '../RNAplfold_scripts/RNAplfold_scripts/M_RNAplfold -W ' + W + ' -L ' + L + ' -u 1 <' + rnac_fasta_file + ' >' + os.path.join(
                                             save_folder, 'M_profile.txt')], shell=True)
        processes = [E_process, H_process, I_process, M_process]
        for p in processes:
            p.wait()

        subprocess.call([
                            "python ../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py " + save_folder + 'E_profile.txt ' + \
                            save_folder + 'H_profile.txt ' + save_folder + 'I_profile.txt ' + save_folder + 'M_profile.txt 1 ' + save_folder + rnac_experiment.replace(
                                '.fasta', '_profile')], shell=True)
        print("[*] Finished structure profiles for " + rnac_experiment)


def main(dataset_type, protein):
    if dataset_type == 'CLIP':
        clip_struct_compute()
    elif dataset_type == 'RNAC_2009':
        rnac_2009_struct_compute()
    elif dataset_type == 'RNAC_2009_numpy':
        load_data_rnac2009(protein_name=protein)
    elif dataset_type == 'CLIP_numpy':
        load_data_clipseq_shorter(protein)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default=None)
    parser.add_argument('--protein', nargs='+', default=None)
    args = parser.parse_args()
    for protein in args.protein:
        main(dataset_type=args.dataset_type,
             protein=protein)
