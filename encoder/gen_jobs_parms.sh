#!/bin/bash
datapath=../alldata/*
for assay in $datapath
#for a`ssay in 2217.csv.out.2
do
	assayname=$(basename "$assay")
	echo "$assayname generated"
	for fold_id in 0 1 2 3 4
	do



		mycase=${assayname}_${fold_id}

		job_path=jobs/${mycase}_encoding.job
        job_log=joblogs/encoder_${mycase}

        echo "/home/lifen/tools/anaconda3/envs/deeplearn/bin/python2.7 transform_try.py ${assayname} ${fold_id} 0.01 100 >> $job_log" > $job_path

		
	done
done
