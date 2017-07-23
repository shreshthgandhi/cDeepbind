import os.path
import subprocess


def main():
    rnac_folder = '../data/rnac/npz_archives'
    clip_folder = '../data/GraphProt_CLIP_sequences'
    clip_experiments = ['CLIPSEQ_ELAVL1', 'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_FUS',
                        'ICLIP_HNRNPC', 'PARCLIP_HUR', 'PARCLIP_IGF2BP123', 'PTBv1',
                        'PARCLIP_PUM2', 'PARCLIP_QKI', 'CLIPSEQ_SFRS1', 'ICLIP_TIA1']
    clip_experiment = os.path.join(clip_folder, clip_experiments[0])
    W = '180'
    L = '120'
    positives_file = clip_experiment + '.ls.positives.fa'
    negatives_file = clip_experiment + '.ls.negatives.fa'
    E_process_pos = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + positives_file,
         ">E_profile_pos.txt"])
    H_process_pos = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + positives_file,
         ">H_profile_pos.txt"])
    I_process_pos = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + positives_file,
         ">I_profile_pos.txt"])
    M_process_pos = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + positives_file,
         ">M_profile_pos.txt"])
    E_process_neg = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + negatives_file,
         ">E_profile_neg.txt"])
    H_process_neg = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + negatives_file,
         ">H_profile_neg.txt"])
    I_process_neg = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + negatives_file,
         ">I_profile_neg.txt"])
    M_process_neg = subprocess.Popen(
        ["../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold", "-W " + W, "-L " + L, "-u 1", "<" + negatives_file,
         ">M_profile_neg.txt"])
    processes = [E_process_pos, H_process_pos, I_process_pos, M_process_pos, E_process_neg, H_process_neg,
                 I_process_neg, M_process_neg]
    for p in processes:
        p.wait()
    subprocess.call(["python", "../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py",
                     "E_profile_pos.txt", "H_profile_pos.txt", "I_profile_pos.txt", "M_profile_pos.txt", "1",
                     "combined_profile_pos.txt"])
    subprocess.call(["python", "../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py",
                     "E_profile_neg.txt", "H_profile_neg.txt", "I_profile_neg.txt", "M_profile_neg.txt", "1",
                     "combined_profile_neg.txt"])
    subprocess.call(
        ["cp", "../RNAplfold_scripts/RNAplfold_scripts/combined_profile_pos.txt", "../data/GraphProt_CLIP_sequences/"])
    subprocess.call(
        ["cp", "../RNAplfold_scripts/RNAplfold_scripts/combined_profile_neg.txt", "../data/GraphProt_CLIP_sequences/"])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_type', default=None)
    # args = parser.parse_args()
    main()
