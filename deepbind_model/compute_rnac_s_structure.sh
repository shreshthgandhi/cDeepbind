#!/usr/bin/env bash
for file in input_seq bg_input_seq;
do
../RNAplfold_scripts/RNAplfold_scripts/E_RNAplfold -W 40 -L 40 u 1 <../data/rnac_s/${file}.fa  >../data/rnac_s/${file}_E.txt&
../RNAplfold_scripts/RNAplfold_scripts/H_RNAplfold -W 40 -L 40 u 1 <../data/rnac_s/${file}.fa  >../data/rnac_s/${file}_H.txt&
../RNAplfold_scripts/RNAplfold_scripts/I_RNAplfold -W 40 -L 40 u 1 <../data/rnac_s/${file}.fa  >../data/rnac_s/${file}_I.txt&
../RNAplfold_scripts/RNAplfold_scripts/M_RNAplfold -W 40 -L 40 u 1 <../data/rnac_s/${file}.fa  >../data/rnac_s/${file}_M.txt&
wait
done

for file in input_seq bg_input_seq;
do
python ../RNAplfold_scripts/RNAplfold_scripts/combine_letter_profiles.py ../data/rnac_s/${file}_E.txt ../data/rnac_s/${file}_H.txt ../data/rnac_s/${file}_I.txt ../data/rnac_s/${file}_M.txt 1 ../data/rnac_s/${file}_combined.txt&
wait
rm ../data/rnac_s/${file}_E.txt ../data/rnac_s/${file}_H.txt ../data/rnac_s/${file}_I.txt ../data/rnac_s/${file}_M.txt
done

