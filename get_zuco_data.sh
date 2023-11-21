#!/usr/bin/env bash
set -euxo pipefail

# this script only downloads, extracts, renames and preprocesses zuco 1
# if you wish to also download, extract, rename and preprocess zuco 2 uncomment everything below
mkdir -p data/zuco
#
# mkdir -p data/zuco2  # optional to also get zuco 2

pip install osfclient

for zuco_data in q3zws # 2urht  # optional to also get zuco2
do
    # if [ "$zuco_data" == "2urht" ];   # optional to also get zuco2then  task1-\ SR task2\ -\ NR
        for task in task3\ -\ TSR
        do
            mkdir -p data/zuco/${task:0:5}
            for subj in ZAB ZDM ZDN ZGW ZJM ZJN ZJS ZKB ZKH ZKW ZMG ZPH
            # Loop over each subject
            do
                # Fetch each file
                osf -p $zuco_data fetch "osfstorage/$task/Matlab files/results${subj}_${task: -3:3}.mat" "data/zuco/${task:0:5}/Matlab_files/results${subj}_${task: -3:3}.mat"
            done
        done
    # else  # optional to also get zuco2
        # for task in task1\ -\ NR task2\ -\ TSR  # optional to also get zuco2
        # do  # optional to also get zuco2
        #     mv $zuco_data"/osfstorage/$task" "data/zuco2/"${task:0:5}  # optional to also get zuco2
        #     mv "data/zuco2/${task:0:5}/Matlab files" "data/zuco2/${task:0:5}/Matlab_files"  # optional to also get zuco2
        # done  # optional to also get zuco2
    # fi  # optional to also get zuco2
done

python3 -m zuco_create_wordinfor_scanpath_files --zuco-task zuco11
python3 -m zuco_create_wordinfor_scanpath_files --zuco-task zuco12
# python3 -m scripts.zuco_create_wordinfor_scanpath_files --zuco-task zuco21  # optional to also get zuco2
