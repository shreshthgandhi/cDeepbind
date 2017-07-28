#!/usr/bin/env bash
for protein in PARCLIP_ELAVL1 PARCLIP_ELAVL1A PARCLIP_FUS ICLIP_HNRNPC PARCLIP_HUR PARCLIP_IGF2BP123 PTBv1 PARCLIP_PUM2 PARCLIP_QKI CLIPSEQ_SFRS1 ICLIP_TIA1;
do
../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.positives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/E_profile_pos.txt&
../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.positives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/H_profile_pos.txt&
../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.positives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/I_profile_pos.txt&
../RNAplfold_scripts/RNAplfold_scripts/M_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.positives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/M_profile_pos.txt&
wait
../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.negatives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/E_profile_neg.txt&
../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.negatives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/H_profile_neg.txt&
../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.negatives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/I_profile_neg.txt&
../RNAplfold_scripts/RNAplfold_scripts/M_RNAplfold -W 180 -L 160 u 1 <../data/GraphProt_CLIP_sequences/${protein}.train.negatives.fa  >../data/GraphProt_CLIP_sequences/structure_annotations/${protein}/M_profile_neg.txt&
wait
done