#!/bin/bash

## sum the ndcg 10 of alll assays
datapath=alldata/*

rm -rf sum_NDCG_rank_encoded
rm -rf sum_NDCG_light_encoded

echo "assayname NDCG_num fold value" >> sum_NDCG_rank_encoded
echo "assayname NDCG_num fold value" >> sum_NDCG_light_encoded

for assay in $datapath
#for a`ssay in 2217.csv.out.2
do
	assayname=$(basename "$assay")
	echo "$assayname generated"
	for fold in 0 1 2 3 4
	do
		mycase=${assayname}_${fold}

		# trainset and testset
		testfile=testdata_encoded/${mycase}.test

		pred_path_rank=svm_rank_pred_encoded/${mycase}.pred
		NDCG_rank=NDCG_rank/${mycase}_encoded.ndcg

		pred_path_light=svm_light_pred_encoded/${mycase}.pred
		NDCG_light=NDCG_light/${mycase}_encoded.ndcg

		#echo "$my_ndcg_10_rank">> all_NDCG_rank
		echo $(less $NDCG_rank|wc)
		less $NDCG_rank |awk '{i++; if(i == 6) print "'$assayname' 5 '$fold' "$NF;}'>> sum_NDCG_rank_encoded
		less $NDCG_rank |awk '{i++; if(i == 11) print "'$assayname' 10 '$fold' "$NF;}'>> sum_NDCG_rank_encoded
		less $NDCG_rank |tail -1|awk '{print "'$assayname' all '$fold' "$NF;}'>> sum_NDCG_rank_encoded
		less $NDCG_light |awk '{i++; if(i == 6) print "'$assayname' 5 '$fold' "$NF;}'>> sum_NDCG_light_encoded
		less $NDCG_light |awk '{i++; if(i == 11) print "'$assayname' 10 '$fold' "$NF;}'>> sum_NDCG_light_encoded
		less $NDCG_light |tail -1|awk '{print "'$assayname' all '$fold' "$NF;}'>> sum_NDCG_light_encoded

		#get_DCG.sh $testfile $pred_path_rank  > $NDCG_rank
		#get_DCG.sh $testfile $pred_path_light > $NDCG_light
		
	done
done
