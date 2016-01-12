#!/bin/bash
datapath=alldata/*

echo "this use the dA encoding, and linear kernel in svm_learn"
for assay in $datapath
#for a`ssay in 2217.csv.out.2
do
	assayname=$(basename "$assay")
	echo "$assayname generated"
	for i in 0 1 2 3 4
	do

		mycase=${assayname}_${i}

		# trainset and testset
		trainfile=traindata_encoded/${mycase}.train
		testfile=testdata_encoded/${mycase}.test

		model_path_rank=svm_rank_model_encoded/${mycase}.model
		pred_path_rank=svm_rank_pred_encoded/${mycase}.pred
		log_path_rank=svm_rank_log_encoded/${mycase}.log

		model_path_light=svm_light_model_encoded/${mycase}.model
		pred_path_light=svm_light_pred_encoded/${mycase}.pred
		log_path_light=svm_light_log_encoded/${mycase}.log

		job_path_rank=jobs_encoded/${mycase}_rank.job
		job_path_light=jobs_encoded/${mycase}_light.job

		c_parm=$(less $trainfile |awk '{print $2}'|uniq|wc -l)

		
		#generate the jobs for svm_rank
		echo "svm_rank_learn -c ${c_parm} -t 0 $trainfile $model_path_rank > $log_path_rank && svm_rank_classify ${testfile} ${model_path_rank} ${pred_path_rank} >> $log_path_rank" >$job_path_rank

		#generate the jobs for svm_light
		echo "svm_light_learn -z r -c 1 -t 0 $trainfile $model_path_light > $log_path_light && svm_light_classify ${testfile} ${model_path_light} ${pred_path_light} >> $log_path_light" >$job_path_light
	done
done
