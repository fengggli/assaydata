import sys 
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy import spatial

import os
# from the svm_rank input file get the feature matrix
def file2matrix(file_path):

    cid_list = []
    with open(file_path) as f:
    # this is to store spase  CSR array
        row = []
        col = []
        data = []
        i = 0
        ret_left = []
        ret_right = []

        for line in f:
            entry = line[:-2].split(' ') # last space
            entry_len = len(entry)

            cid_list.append((entry[0], entry[1]))

            # store the feature matrix
            for j in range(2, entry_len):
                pair = entry[j].split(':')
                row.append(i)
                col.append(int(pair[0])-1)
                #data.append(float(pair[1]))

                # binary
                data.append(1.0)
            i += 1

    f.close()

    ret_matrix = csr_matrix( (data, (row, col))).toarray()


    return (ret_matrix, cid_list)


# caluate the similarity using Tanimoto kernel
def get_sim(matrix_x, cid_list):
    num_items = matrix_x.shape[0]

    sim_set_all = []
    sim_set_aa = []
    sim_set_ai = []
    sim_set_ii = []

    cid_pairs_all = []
    cid_pairs_aa = []
    cid_pairs_ai = []
    cid_pairs_ii = []

    for i in range(num_items):
        u = matrix_x[i]
        if np.linalg.norm(u) == 0:
            continue
        for j in range(i+1, num_items):
            v = matrix_x[j]
            if np.linalg.norm(v) == 0:
                continue
            '''
            a = np.dot(matrix_x[i], matrix_x[i])
            b = np.dot(matrix_x[j], matrix_x[j])
            c = np.dot(matrix_x[i], matrix_x[j])

            sim = c/ (a+b-c)
            '''
            #print str(len(matrix_x[i])), str(len(matrix_x[j]))
            sim = 1 - spatial.distance.cosine(u, v)

            #print a,b,c, sim
            if sim >= 0:
                sim_set_all.append(sim)
                cid_pairs_all.append((cid_list[i], cid_list[j]))
                # both of the compond is active
                if cid_list[i][1] == 'Active' and cid_list[j][1] == 'Active':
                    sim_set_aa.append(sim)
                    cid_pairs_aa.append((cid_list[i][0], cid_list[j][0]))

                # both is inactive
                elif cid_list[i][1] == 'Inactive' and cid_list[j][1] == 'Inactive':
                    sim_set_ii.append(sim)
                    cid_pairs_ii.append((cid_list[i][0], cid_list[j][0]))
                # active and inactive
                elif cid_list[i][1] == 'Inactive' and cid_list[j][1] == 'Active':
                    sim_set_ai.append(sim)
                    cid_pairs_ai.append((cid_list[i][0], cid_list[j][0]))

                elif cid_list[i][1] == 'Active' and cid_list[j][1] == 'Inactive':
                    sim_set_ai.append(sim)
                    cid_pairs_ai.append((cid_list[i][0], cid_list[j][0]))
            else:
                print 'error', 'simiarity = ',str(np.dot(u, v)),str(np.linalg.norm(u)), str(np.linalg.norm(v))
                break

    return  cid_pairs_all, sim_set_all, cid_pairs_aa, sim_set_aa, cid_pairs_ai, sim_set_ai, cid_pairs_ii, sim_set_ii


def test():

    matrix_x = np.array([[1.0,2],[3,4],[5.0,6]])
    cid_list = [(0,'Active'),(1, 'Active'),(2,'Active')]
    cid_pairs_all, sim_set_all, cid_pairs_aa, sim_set_aa, cid_pairs_ai, sim_set_ai, cid_pairs_ii, sim_set_ii = get_sim(matrix_x, cid_list)

    print sim_set_all



if __name__ == '__main__':
    print 'start'

    f_summary_result = open('summary.result' ,'w')
    f_assaylist = open('assaylist', 'r')

    print >> f_summary_result, 'assay avg std num_pairs note'
    for line in f_assaylist:
    #for line in [1]:


        assayname = line[:-1]
        assay_path = 'data_all/' + assayname
        '''
        assayname = 'test'
        assay_path = 'test/test.data'
        '''
        print assayname, 'start!'


        feature_matrix, cid_list = file2matrix(assay_path)

        #feature_matrix_active, cid_list_active = file2matrix(assay_path_active)

        cid_pairs_all, sim_set_all, cid_pairs_aa, sim_set_aa, cid_pairs_ai, sim_set_ai, cid_pairs_ii, sim_set_ii = get_sim(feature_matrix, cid_list)
        #cid_pairs_active, similarity_list_active = get_sim(feature_matrix_active, cid_list_active)

        #f_result_all = open('sim_result/'+assayname+'all.txt', 'w')

        f_result_all = open('sim_result/'+assayname+'all.txt', 'w')
        print >> f_result_all, 'Compound1', 'Compound2', 'Similarity'

        for myindex in range(len(sim_set_all)):
            print >> f_result_all, cid_pairs_all[myindex], sim_set_all[myindex]


        if len(sim_set_aa) != 0:
            sim_avg_aa = np.average(sim_set_aa)
            sim_std_aa = np.std(sim_set_aa)
        else:
            sim_avg_aa = 0
            sim_std_aa = 0
        print >>f_summary_result, assayname, sim_avg_aa, sim_std_aa, len(sim_set_aa),'active_active'

        if len(sim_set_ai) != 0:
            sim_avg_ai = np.average(sim_set_ai)
            sim_std_ai = np.std(sim_set_ai)
        else:
            sim_avg_ai = 0
            sim_std_ai = 0
        print >>f_summary_result, assayname, sim_avg_ai, sim_std_ai, len(sim_set_ai), 'inactive_active'

        if len(sim_set_ii) != 0:
            sim_avg_ii = np.average(sim_set_ii)
            sim_std_ii = np.std(sim_set_ii)
        else:
            sim_avg_ii = 0
            sim_std_ii = 0
        print >>f_summary_result, assayname, sim_avg_ii, sim_std_ii, len(sim_set_ii),'inactive_inactive'


        plt.hist([sim_set_all, sim_set_aa], color=['blue','red'], label=['all compond pairs','active componds_pairs'])
        plt.legend()
        plt.xlabel('Tanimoto similarity')
        plt.ylabel('number of pairs')

        plt.savefig('sim_result/' + assayname + 'hist.png')
        plt.close()

    
