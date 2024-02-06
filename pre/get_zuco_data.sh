#!/usr/bin/env bash
set -euxo pipefail

# this script only downloads, extracts, renames and preprocesses zuco 1
mkdir -p data/zuco

pip install osfclient

for zuco_data in q3zws
do
        for task in task1-\ SR task3\ -\ TSR
        do
            mkdir -p data/zuco/${task:0:5}
            for subj in ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH
            do
                osf -p $zuco_data fetch "osfstorage/$task/Matlab files/results${subj}_${task: -3:3}.mat" "data/zuco/${task:0:5}/Matlab_files/results${subj}_${task: -3:3}.mat"
            done
        done
done

python pre/zuco_preprocess.py --zuco_task task1
python pre/zuco_preprocess.py --zuco_task task3

python pre/fixation.py --task task1
python pre/fixation.py --task task3

