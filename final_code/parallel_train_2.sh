#!/usr/bin/env bash
declare -a protein_list=(
                     'RNCMPT00138'
                     'RNCMPT00139'
                     'RNCMPT00013'
                     'RNCMPT00140'
                     'RNCMPT00141'
                     'RNCMPT00142'
                     'RNCMPT00143'
                     'RNCMPT00144'
                     'RNCMPT00145'
                     'RNCMPT00146'
                     'RNCMPT00147'
                     'RNCMPT00148'
                     'RNCMPT00149'
                     'RNCMPT00014'
                     'RNCMPT00150'
                     'RNCMPT00151'
                     'RNCMPT00152')
#declare -a protein_list=('RNCMPT00100')
declare GPUS=2

for i in "${protein_list[@]}"
do
#    for gpu in $(seq 0 $GPUS)
#    do
python Main_structure.py --gpus 2 --protein $i &&
echo $i
#    done
done