#!/usr/bin/env bash
declare -a protein_list=(
                     'RNCMPT00161'
'RNCMPT00278'
'RNCMPT00172'
'RNCMPT00283')
#declare -a protein_list=('RNCMPT00100')
declare GPUS=2

for i in "${protein_list[@]}"
do
#    for gpu in $(seq 0 $GPUS)
#    do
python Main_structure.py --gpus 2 --protein $i --model_type RNN_struct &&
echo $i
#    done
done