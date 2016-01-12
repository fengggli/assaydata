import math
import numpy
from os  import path



def get_ndcg(test_path, pred_path):
    test_rank = []
    pred_rank = []

    # all the return value
    ndcg10 = 0
    ndcg5 = 0
    ndcgall = 0

    '''
    
    if path.isfile(pred_path) == False:
        print 'false pred path', pred_path
        return [0, 0, 0, 0]
    
    '''
    with open(test_path) as fp:
        for line in fp:

            splits = line.split(' ')
            test_rank.append(float(splits[0]))
    with open(pred_path) as fp:
        for line in fp:
            pred_rank.append(float(line))

    #print("test rank:", test_rank)
    #print("prediction rank:", pred_rank)

    #get the index of the sorted list
    index_test = sorted(range(len(test_rank)), key=lambda k: test_rank[k], reverse =1)

    #get the index of the sorted list in prediction
    index_pred = sorted(range(len(pred_rank)), key=lambda k: pred_rank[k], reverse =1)

    #print("test index after sorted based score",index_test)
    #print("pred_index after sorted based score",index_pred)
    #print("length is ", len(index_pred))

    DCG = 0

    #this best DCG is for normalization
    best_DCG = 0

    current_rank = 0

    #this is index is for the best ranking
    if len(index_test)!=len(index_pred):
        print("prediction and test set should have the same length")
    #print("n DCG max_DCG NDCG")
    #this is the least and largest  CID score
    min_range =  test_rank[index_test[len(index_test)-1]]
    max_range =  test_rank[index_test[0]]
    #print("max_range:", max_range, "min_range", min_range)
    for iter in range(0, len(index_pred)):
        # a pointer to pred_set
        i = index_pred[iter]
        # a pointer to test_set
        j = index_test[iter]

        # actual score of this doc
	    # in the NDCG the score should normalized to  0~5, 0 for bad, 5 for exellent

        #print(iter,"'s interation, i(pred):,j(test): =",test_rank[i], test_rank[j])
        score = 5*(test_rank[i]-min_range)/(max_range-min_range)
        best_score = 5*(test_rank[j]-min_range)/(max_range-min_range)
	    #score_best = 5*(x-min_range)/(max_range-min_range)

        #score = (108.803-math.e**(-test_rank[i]))/20
        #best_score = (108.803-math.e**(-test_rank[j]))/20
        #print("score", score)
        #print("best score",best_score)

        Gain = 2**score-1
        best_Gain = 2**best_score-1
        #print("get gain:", Gain)
        CG = (1/math.log(iter+2, 2))*Gain
        best_CG = (1/math.log(iter+2, 2))*best_Gain

        #print("add CG :", CG)
        #print("add bestCG :", best_CG)
        DCG += CG
        best_DCG += best_CG
        #print("DCG is :", DCG)

        ndcg = DCG/best_DCG

        if iter == 9:
            ndcg10 = ndcg

        if iter == 4:
            ndcg5 = ndcg

        if iter == len(index_pred)-1:
            ndcgall = ndcg


        #print(iter+1, DCG, best_DCG, DCG/best_DCG)


    return [ndcg10, ndcg5, ndcgall, len(index_pred)]

if __name__ == "__main__":


    for rank_function in range(2):  # svm_rank:0 or svm_light:1

        if rank_function == 0:
            print 'for svm ranking \n'
            output_path = 'NDCG_result/svm_rank_sampled.result'

        else:
            print 'for svm light \n'
            output_path = 'NDCG_result/svm_light_sampled.result'


        ndcg_result = open(output_path, 'w')

        print 'test'
        print >> ndcg_result,'assayname NDCG_k avg variance length'

        
        f = open('assaylist', 'r')
        for line in f:

        # for each assay in the assay list
            assayname = line[:-1]
        
        # a minor test

        #assayname = '602235.csv.out.2'

            ndcg10_this_assay = []
            ndcg5_this_assay = []
            ndcgall_this_assay = []

            for fold_id in range(5):
                mycase = assayname + '_' +str(fold_id)
                mycasemore = mycase + '_pct_0.01_epochs_100' 


                test_path = 'encoder/testdata_sampled_encoded/' + mycasemore + '.test'

                if rank_function == 0:
                    pred_path = 'svm_rank_pred_sampled/' + mycase + '.pred'

                else:
                    pred_path = 'svm_light_pred_sampled/' + mycase + '.pred'


                [ndcg10,ndcg5,ndcgall, rank_length] = get_ndcg(test_path, pred_path)

                if ndcg5 != 0:
                    ndcg5_this_assay.append(ndcg5)
                else:
                    print 'empty path', test_path, pred_path
                if ndcg10 != 0:
                    ndcg10_this_assay.append(ndcg10)
                if ndcgall != 0:
                    ndcgall_this_assay.append(ndcgall)

            # average of the ndcg
            avg_ndcg10 = numpy.average(ndcg10_this_assay)
            avg_ndcg5 = numpy.average(ndcg5_this_assay)
            avg_ndcgall= numpy.average(ndcgall_this_assay)

            #variance of the ndcg
            var_ndcg10 = numpy.var(ndcg10_this_assay)
            var_ndcg5 = numpy.var(ndcg5_this_assay)
            var_ndcgall = numpy.var(ndcgall_this_assay)

            print >> ndcg_result, assayname + ' ' + str(rank_length)+ ' 5 ' + str(avg_ndcg5) + ' ' + str(var_ndcg5)
            print >> ndcg_result, assayname + ' ' + str(rank_length)+ ' 10 ' + str(avg_ndcg10) + ' ' + str(var_ndcg10)
            print >> ndcg_result, assayname + ' ' + str(rank_length)+ ' all ' + str(avg_ndcgall) + ' ' + str(var_ndcgall)

        ndcg_result.close()



