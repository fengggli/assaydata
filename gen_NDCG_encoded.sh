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
		testfile=testdata_encoded/${mycase}.test

		pred_path_rank=svm_rank_pred_encoded/${mycase}.pred
		NDCG_rank=NDCG_rank/${mycase}_encoded.ndcg

		pred_path_light=svm_light_pred_encoded/${mycase}.pred
		NDCG_light=NDCG_light/${mycase}_encoded.ndcg

		get_DCG.sh $testfile $pred_path_rank  > $NDCG_rank
		get_DCG.sh $testfile $pred_path_light > $NDCG_light
		
	done
done
