import math
import numpy
from os  import path


def get_active_dic(assay_name):
    dic_active = {}
    assay_path = 'alldata/'+ assay_name
    with open(assay_path) as fp:
        count = 0
        for line in fp:
            # don't want the first line
            if count == 0:
                count += 1
                continue
            splits = line[:-1].split(' ')
            if splits[1] == 'Active':
                dic_active[splits[0]] = 1
    return dic_active




def get_ndcg(test_path, pred_path, only_active=0):
    test_rank = []
    pred_rank = []

    # all the return value
    ndcg10 = 0
    ndcg5 = 0
    ndcgall = 0


    if only_active == 1:
        dic_active = get_active_dic('733.csv.out.2')

    if path.isfile(pred_path) == False:
        return [0, 0, 0, 0]


    '''
    with open(test_path) as fp:
        for line in fp:

            splits = line.split(' ')
            test_rank.append(float(splits[0]))
    with open(pred_path) as fp:
        for line in fp:
            pred_rank.append(float(line))

    '''
    myiter = 0
    fp_test = open(test_path, 'r')
    fp_pred = open(pred_path, 'r')
    line_count = 0

    all_lines_test = []
    all_lines_pred = []
    for line in fp_test:
        line_count += 1
        all_lines_test.append(line)

    for line in fp_pred:
        all_lines_pred.append(line[:-1])

    for myiter in range(line_count):
        splits = all_lines_test[myiter][:-1].split(' ')  # for ndcg_origin, -1 should replaced with -2, pay attention to the format of testfile
        label_with_sharp = splits[-1]
        mylabel = label_with_sharp[1:]
        if only_active == 0:
            test_rank.append(float(splits[0]))
            pred_rank.append(float(all_lines_pred[myiter]))
        else:
            if mylabel in dic_active:
                test_rank.append(float(splits[0]))
                pred_rank.append(float(all_lines_pred[myiter]))

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

        #print 'index', str(iter),'gain/bestgain = ', str(score), str(best_score) ,'cg = ', str(CG),', best cg = ', str(best_CG), ', dcg = ', str(DCG), ', best_dcg = ', str(best_DCG), ', ndcg = ', str(ndcg)

        if iter == 9:
            ndcg10 = ndcg

        if iter == 4:
            ndcg5 = ndcg

        if iter == len(index_pred)-1:
            ndcgall = ndcg


        #print(iter+1, DCG, best_DCG, DCG/best_DCG)


    return [ndcg10, ndcg5, ndcgall, len(index_pred)]


