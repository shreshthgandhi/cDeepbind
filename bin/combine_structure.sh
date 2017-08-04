#!/usr/bin/env bash
#for protein in PARCLIP_ELAVL1 PARCLIP_ELAVL1A PARCLIP_FUS ICLIP_HNRNPC PARCLIP_HUR PARCLIP_IGF2BP123 PTBv1 PARCLIP_PUM2 PARCLIP_QKI CLIPSEQ_SFRS1 ICLIP_TIA1;
for protein in ICLIP_HNRNPC PARCLIP_ELAVL1 PARCLIP_ELAVL1A PARCLIP_FUS PARCLIP_HUR PARCLIP_IGF2BP123 PTBv1
do
python ../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py ../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/E_profile_pos.txt \
../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/H_profile_pos.txt ../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/I_profile_pos.txt \
../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/M_profile_pos.txt 1 ../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/${protein}.train.positives_combined.txt
python ../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py ../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/E_profile_neg.txt \
../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/H_profile_neg.txt ../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/I_profile_neg.txt \
../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/M_profile_neg.txt 1 ../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/${protein}.train.negatives_combined.txt
done