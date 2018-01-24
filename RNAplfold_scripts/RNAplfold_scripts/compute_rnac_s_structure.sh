#!/usr/bin/env bash
for file in input_seq bg_input_seq;
do
./E_RNAplfold -W 40 -L 40 u 1 <${file}.fa  >${file}_E.txt
./H_RNAplfold -W 40 -L 40 u 1 <${file}.fa  >${file}_H.txt
./I_RNAplfold -W 40 -L 40 u 1 <${file}.fa  >${file}_I.txt
./M_RNAplfold -W 40 -L 40 u 1 <${file}.fa  >${file}_M.txt
done

for file in input_seq bg_input_seq;
do
python ./combine_letter_profiles.py ${file}_E.txt ${file}_H.txt ${file}_I.txt ${file}_M.txt 1 ${file}_combined.txt
done
