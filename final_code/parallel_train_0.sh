#!/usr/bin/env bash
declare -a protein_list=('RNCMPT00100'
                     'RNCMPT00101'
                     'RNCMPT00102'
                     'RNCMPT00103'
                     'RNCMPT00104'
                     'RNCMPT00105'
                     'RNCMPT00106'
                     'RNCMPT00107'
                     'RNCMPT00108'
                     'RNCMPT00109'
                     'RNCMPT00010'
                     'RNCMPT00110'
                     'RNCMPT00111'
                     'RNCMPT00112'
                     'RNCMPT00113'
                     'RNCMPT00114'
                     'RNCMPT00116'
                     'RNCMPT00117')
#declare -a protein_list=('RNCMPT00100')
declare GPUS=2

for i in "${protein_list[@]}"
do
#    for gpu in $(seq 0 $GPUS)
#    do
python Main_structure.py --gpus 0 --protein $i &&
echo $i
#    done
done