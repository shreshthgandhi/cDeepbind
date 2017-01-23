#!/usr/bin/env bash
declare -a protein_list=('RNCMPT00118'
                     'RNCMPT00119'
                     'RNCMPT00011'
                     'RNCMPT00120'
                     'RNCMPT00121'
                     'RNCMPT00122'
                     'RNCMPT00123'
                     'RNCMPT00124'
                     'RNCMPT00126'
                     'RNCMPT00127'
                     'RNCMPT00129'
                     'RNCMPT00012'
                     'RNCMPT00131'
                     'RNCMPT00132'
                     'RNCMPT00133'
                     'RNCMPT00134'
                     'RNCMPT00136'
                     'RNCMPT00137')
#declare -a protein_list=('RNCMPT00100')
declare GPUS=2

for i in "${protein_list[@]}"
do
#    for gpu in $(seq 0 $GPUS)
#    do
python Main_structure.py --gpus 1 --protein $i &&
echo $i
#    done
done