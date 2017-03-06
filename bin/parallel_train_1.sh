#!/usr/bin/env bash
declare -a protein_list=('RNCMPT00199'
'RNCMPT00209'
'RNCMPT00095'
'RNCMPT00146'
'RNCMPT00149')

#declare -a protein_list=('RNCMPT00289'
#'RNCMPT00158'
#'RNCMPT00258'
#'RNCMPT00081'
#'RNCMPT00147')
#declare -a protein_list=('RNCMPT00100')
declare GPUS=2

for i in "${protein_list[@]}"
do
#    for gpu in $(seq 0 $GPUS)
#    do
python Main_structure.py --gpus 1 --protein $i --model_type RNN_struct &&
echo $i
#    done
done