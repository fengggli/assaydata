import math
import numpy
from os  import path
from cal_ndcg import get_ndcg




if __name__ == "__main__":


    for rank_function in range(2):  # svm_rank:0 or svm_light:1

        if rank_function == 0:
            print 'for svm ranking \n'
            output_path = 'NDCG_result/svm_rank_sampled_733_0_0.004.result'

        else:
            print 'for svm light \n'
            output_path = 'NDCG_result/svm_light_sampled_733_0_0.004.result'


        ndcg_result = open(output_path, 'w')

        print 'test'
        print >> ndcg_result, '{0:16}{1:>15}{2:>8}{3:>10}{4:>10}{5:>10}'.format('assayname', 'test_length','NDCG_k', 'avg', 'var','enc_rate')

        '''
        
        f = open('assaylist', 'r')
        for line in f:

        # for each assay in the assay list
            assayname = line[:-1]
        ''' 
        # a minor test

        assayname = '733.csv.out.2'

                
        #for fold_id in range(5):

        fold_id = 0
        for encoding_rate in [0.004]: 
            ndcg10_this_assay = []
            ndcg5_this_assay = []
            ndcgall_this_assay = []


            mycase = assayname + '_' +str(fold_id)
            mycasemore = mycase + '_pct_' + str(encoding_rate)+'_epochs_100_sigmoid' 


            test_path = 'encoder/testdata_sampled_encoded_new/' + mycasemore + '.test'

            if rank_function == 0:
                pred_path = 'svm_rank_pred_sampled_new/' + mycasemore + '.pred'

            else:
                pred_path = 'svm_light_pred_sampled_new/' + mycasemore + '.pred'


            [ndcg10,ndcg5,ndcgall, rank_length] = get_ndcg(test_path, pred_path,only_active=0)

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
            

            '''
            print >> ndcg_result, '{0:10}' % assayname + ' ' + '{0:5}' % rank_length+ ' 5 ' + '{0:8.5f}' % avg_ndcg5 + ' ' + '{0:8.5f}' % var_ndcg5 + ' ' + str(encoding_rate)
            print >> ndcg_result, assayname + ' ' + str(rank_length)+ ' 10 ' + '%.5f' % avg_ndcg10 + ' ' + '%.5f' % var_ndcg10 + ' ' + str(encoding_rate)
            print >> ndcg_result, assayname + ' ' + str(rank_length)+ ' all ' + '%.5f' % avg_ndcgall + ' ' + '%.5f' % var_ndcgall + ' ' + str(encoding_rate)
            '''
            print >> ndcg_result, '{0:16}{1:>15}{2:>8s}{3:>10.5f}{4:>10.4f}{5:>10.4f}'.format(assayname, rank_length,'5', avg_ndcg5, var_ndcg5, encoding_rate)
            print >> ndcg_result, '{0:16}{1:>15}{2:>8s}{3:>10.5f}{4:>10.4f}{5:>10.4f}'.format(assayname, rank_length,'10', avg_ndcg10, var_ndcg10, encoding_rate)
            print >> ndcg_result, '{0:16}{1:>15}{2:>8s}{3:>10.5f}{4:>10.4f}{5:>10.4f}'.format(assayname, rank_length,'all', avg_ndcgall, var_ndcgall, encoding_rate)


        ndcg_result.close()



