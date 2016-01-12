#!/bin/bash
datapath=alldata/*
for assay in $datapath
#for a`ssay in 2217.csv.out.2
do
	assayname=$(basename "$assay")
	echo "$assayname generated"
	for i in 0 1 2 3 4
	do

		mycase=${assayname}_${i}

		# trainset and testset
		trainfile=traindata/${mycase}.train
		testfile=testdata/${mycase}.test

		model_path_rank=svm_rank_model/${mycase}.model
		pred_path_rank=svm_rank_pred/${mycase}.pred
		log_path_rank=svm_rank_log/${mycase}.log

		model_path_light=svm_light_model/${mycase}.model
		pred_path_light=svm_light_pred/${mycase}.pred
		log_path_light=svm_light_log/${mycase}.log

		job_path_rank=jobs/${mycase}_rank.job
		job_path_light=jobs/${mycase}_light.job

		c_parm=$(less $trainfile |awk '{print $2}'|uniq|wc -l)

		
		#generate the jobs for svm_rank
		echo "svm_rank_learn -c ${c_parm} -t 4 $trainfile $model_path_rank > $log_path_rank && svm_rank_classify ${testfile} ${model_path_rank} ${pred_path_rank} >> $log_path_rank" >$job_path_rank

		#generate the jobs for svm_light
		echo "svm_light_learn -z r -c 1 -t 4 $trainfile $model_path_light > $log_path_light && svm_light_classify ${testfile} ${model_path_light} ${pred_path_light} >> $log_path_light" >$job_path_light
	done
done
