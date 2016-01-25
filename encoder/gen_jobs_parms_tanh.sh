#!/bin/bash
#datapath=../alldata/*
#for assay in $datapath
#for a`ssay in 2217.csv.out.2

# 0 for sigmoid 1 for tanh
for encode_function in 1
do
	if [ $encode_function -eq 0 ]; then
		encode='_sigmoid'
	elif [ $encode_function -eq 1 ]; then
		encode='_tanh'
	fi

	for assayname in 733.csv.out.2
	do
		#assayname=$(basename "$assay")
		echo "$assayname generated"
		for fold_id in 0
		# try one fold first
		do
			for percent_encode in 0.01 0.004
			do
				for sample_method in 0 -1 
				# 0 default, -1 not sampling, 1 only sample non-zero values
				do
				if [ $sample_method -eq 0 ]; then
					sampling=''
				elif [ $sample_method -eq -1 ]; then
					sampling='_without_sampling'
				elif [ $sample_method -eq 1 ]; then
					sampling='_only_nz'
				fi
				mycase=${assayname}_${fold_id}_pct_${percent_encode}_epochs_100${sampling}${encode}

				job_path=jobs/${mycase}.job
				job_log=joblogs/${mycase}.log

				echo "/home/lifen/tools/anaconda3/envs/deeplearn/bin/python2.7 transform_try.py -a=${assayname} -f=${fold_id} -p=${percent_encode} -s=${sample_method} -e=${encode_function} >> $job_log" > $job_path
				done

			done
			
		done
	done
done
