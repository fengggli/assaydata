import os
import numpy
import scipy
import dA
import cPickle
import gzip

from scipy.sparse import csr_matrix


# from the svm_rank input file get the feature matrix
def file2matrix(file_path):
    with open(file_path) as f:
    # this is to store spase  CSR array
        row = []
        col = []
        data = []
        i = 0
        ret_left = []
        ret_right = []

        for line in f:
            entry = line[:-2].split(' ')
            entry_len = len(entry)

            # store the left part
            ret_left.append(entry[0] + ' ' +  entry[1])
            ret_right.append(entry[-1])

            # store the feature matrix
            for j in range(2, entry_len-1): # the last col is the cid name
                pair = entry[j].split(':')
                row.append(i)
                col.append(int(pair[0])-1)
                data.append(int(pair[1]))
            i += 1

    f.close()

    ret_matrix = csr_matrix( (data, (row, col))).toarray()


    return (ret_matrix, ret_left, ret_right)




    # get the numpy array

    # use picke to dump the numpy array

    # return the path of the pickle

def matrix2file(matrix_x, matrix_left, matrix_right ,file_path):
    #s_matrix_x = csr_matrix(matrix_x)
    #print s_matrix_x

    row_prev = -1

    all_rows = []

    num_row = 0
    num_col = 0
    for row in matrix_x:
        buffer = ''
        num_col = 0
        for col in row:

            if col > 0.5:
                buffer += str(num_col+1) + ':1 '

            num_col += 1

        all_rows.append(buffer)
        num_row += 1

    #print all_rows

    f = open(file_path, 'w')
    num_row = 0
    for row in all_rows:
        buffer = matrix_left[num_row] + ' ' + all_rows[num_row] + matrix_right[num_row]
        print >> f, buffer
        num_row += 1
    f.close()








# use the degnosing autocoder to transform the train and test data (format in numpy/format in the assay)
def do_transform(train_path, test_path, new_train_path, new_test_path, pickle_path, model_path, pecent_encode,training_epochs):
    # left is the left part besides the feature matrix
    matrix_train, train_left, train_right= file2matrix(train_path)
    matrix_test, test_left, test_right = file2matrix(test_path)





    dim_in = max(matrix_train.shape[1], matrix_test.shape[1])
    extra_col = 0

    # the dimension of two sparse matrix may be different
    if dim_in > matrix_train.shape[1]:
        extra_col = dim_in - matrix_train.shape[1]
        matrix_train = numpy.hstack((matrix_train, numpy.zeros((matrix_train.shape[0], extra_col), dtype=int)))

    if dim_in > matrix_test.shape[1]:
        extra_col = dim_in - matrix_test.shape[1]
        matrix_test = numpy.hstack((matrix_test, numpy.zeros((matrix_test.shape[0], extra_col), dtype= int)))

    # save the trainset and test set (which will be used by dA module)
    f = gzip.open(pickle_path, 'w')
    cPickle.dump([matrix_train, matrix_test], f)
    f.close()


    dim_out = int(dim_in*pecent_encode)
    print 'before mapping', matrix_train.shape, matrix_test.shape, 'mapped to dim_out = ', dim_out

    matrix_train_new, matrix_test_new = dA.test_dA(1, 0, dim_in, dim_out, learning_rate=0.1, training_epochs,
            dataset=pickle_path,
            batch_size=10, output_path=model_path)

    matrix2file(matrix_train_new, train_left, train_right, new_train_path)
    matrix2file(matrix_test_new, test_left, test_right, new_test_path)




if __name__ == '__main__':
    assay_directory = '../alldata'
    percent_encode = 0.01
    for file_name in os.listdir(assay_directory):
        for current_fold in range(5):
            mycase = file_name +'_' + str(current_fold)
            train_path = '../traindata/' + mycase +'.train'
            test_path = '../testdata/' + mycase +'.test'

            new_train_path = '../traindata_new/' + mycase +'.train'
            new_test_path = '../testdata_new/' + mycase +'.test'

            pickle_path = '../data/' + mycase +'.pkl.gz'  # this is for dA module

            model_path = '../W_b/' +mycase +'.pkl.gz' # this is the encode 's W and b
            print train_path, test_path, new_train_path, new_test_path

            do_transform(train_path, test_path, new_train_path, new_test_path, pickle_path, model_path, percent_encode)
            
'''
def test():
    matrix_train, train_left , train_right= file2matrix('../traindata/test.train')
    print matrix_train
    print train_left
    print train_right

    matrix2file(matrix_train, train_left, train_right , '../traindata_new/test.train')
'''
