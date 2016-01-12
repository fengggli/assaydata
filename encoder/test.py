import os
import sys
import timeit

import numpy as np


#def get_rand_loss(a, b):

if __name__ == '__main__':
    a = np.array([[8,0,0,0,0,0, 1],[2,0, 0,0,0,0, 2]])
    b = np.array([[1, 0,2,3,4,5,6],[1,2,3,4,5,6,7]])

    print a
    print b


    np.random.seed(1)

    m = np.array(np.sum(a, axis=0))
    
    print 'm = ', m, type(m)

    d = np.random.binomial(size = m.shape[0], n = 1, p = 0.4 )

    print 'd = ', d

    #my_add = theano.function([a,b], d)

    e = np.logical_or(d>0, m>0)
    print e

    # how to stack??
    sampled_index = []

    # stack 10 times, 10 is the batch size

    for i in range(a.shape[1]):
        if e[i] != 0:
            sampled_index.append(i)
    print sampled_index



    a_sampled = a[:,sampled_index]
    b_sampled = b[:,sampled_index]
    print a_sampled
    print b_sampled



'''

n, p = 10, .5
s = np.random.binomial(1, 0.1, 10)

print s

'''
