"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import sys
import timeit

import numpy
import gzip

import cPickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy.sparse import csr_matrix

#from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,


        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,

        filter_matrix=None,


        # filter for sampling, added by feng
        #sampling_filter = None

        # my type, added by feng

        s_type = 1,
        error_type = 0
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
 
            
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        #self.filter_matrix = numpy.random.randint(2, size=(mybatch_size, n_visible))
        #self.filter_matrix = numpy.zeros(mybatch_size, n_visible)
        self.myfilter_matrix = filter_matrix
        '''
        initial_filter = numpy.zeros(
                shape = input_shape,
                dtype=theano.config.floatX
            )
        self.sampling_filter = theano.shared(value=initial_filter, name='sampling_filter', borrow=True)

        '''

        '''

        for i in range(all_col_num):

                print

        p = all_zero / self.x.shape[1]

        # radom select 0.01 of all the features, this can overlap with the non-zero features
        d = self.theano_rng.binomial(size=m.shape, n = 1, p = 0.01)

        self.sampling_filter =  T.or_(d>0, m>0)
        '''







        self.params = [self.W, self.b, self.b_prime]

        self.s_type = s_type
        self.error_type = error_type


    '''
    def set_filter(self):
        # create the filter, sample based on the non-zero features and the sample the sample number of non-zero features
        #m = T.sum(a, axis=0)

        all_nz = []
        all_zero = []
        all_filter = []

        row = []
        col = []
        data = []

        # fix the random seed
        numpy.random.seed(5)


        #print self.input_shape.eval()

        all_row_num = self.input_shape[0].eval()
        all_col_num = self.input_shape[1].eval()


        x_instance = self.x.eval()

        #print type(all_col_num), all_col_num
        i = 0
        #for i in range(all_row_num):
        #while T.lt(i, all_row_num.eval()):
        while all_row_num > i:
            nz_this_row = []
            zeros_this_row = []
            j = 0
            #for j in range(all_col_num):
            while all_col_num > j:
            #while T.lt(j, all_col_num.eval()):
                elem = x_instance[i, j]
                if elem == 0:  # errors here elem is always nonzero
                    zeros_this_row.append(j)
                else:
                    nz_this_row.append(j)
                #print 'inside ', str(j), 'iteration in', str(all_col_num)

                j += 1

            len_sampled = len(nz_this_row)
            sampled_zeros_this_row = numpy.random.choice(zeros_this_row,size = len_sampled, replace=False)
            filtered_result_this_row = nz_this_row + sampled_zeros_this_row.tolist()


            # prepare to construct the sparse matrix
            for m in range(len(filtered_result_this_row)):
                row.append(i)
                col.append(filtered_result_this_row[m])
                data.append(float(1))


            # actually csr format
            all_zero.append(zeros_this_row) # all of the zero index in each row
            all_nz.append(nz_this_row) # all of the nz index in each row
            all_filter.append(nz_this_row + sampled_zeros_this_row)

            #print 'the ', str(i), 'iteration finished'
            i += 1

        tmp_filter_matrix = csr_matrix((data, (row, col))).toarray()
        extra_col = self.n_visible - tmp_filter_matrix.shape[1]
        if extra_col > 0:
            self.filter_matrix = numpy.hstack((tmp_filter_matrix, numpy.zeros((tmp_filter_matrix.shape[0], extra_col), dtype=tmp_filter_matrix.dtype)))
        else:
            self.filter_matrix = tmp_filter_matrix
        print 'sampling filter created'

    '''
    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


    # set the filter based on the input,(sample in each instance)
    #def set_sampling_filter(self, a):


    # get the sampled data using the filter, this will be called every time caculating the cost
    def get_sampled(self, a):
        #sampled_data = self.filter_matrix.multiply(a)



        #sampled_data = csr_matrix.multiply(self.filter_matrix, a)
        sampled_data = self.myfilter_matrix * a
        return sampled_data
        

    # not used ...
    '''
    # random sample features, default sample rate is 0.01
    def get_sampled(self, a, b):
        #theano_rng = RandomStreams()

        m = T.sum(a, axis=0)

        # radom select 0.01 of all the features, this can overlap with the non-zero features
        d = self.theano_rng.binomial(size=m.shape, n = 1, p = 0.01 )

        my_filter = T.or_(d>0, m>0)

        a_sampled = a*my_filter
        b_sampled = b*my_filter

        #return theano.function([a,b], [a_sampled, b_sampled])
        return [a_sampled, b_sampled]
    '''

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        sampled_x = self.get_sampled(tilde_x)
        sampled_z = self.get_sampled(z)

        #sampled_x = tilde_x
        #sampled_z = z
        # sampled in each update

        #[sampled_x, sampled_z] = self.get_sampled(tilde_x, z)

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch

        
        '''
        # the cross entropy loss
        L = T.fmatrices()
        if self.error_type == 1:
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        #square error, added by feng

        #print 'using'
        if self.error_type == 0:
            L = T.sum((self.x - z)**2, axis = 1)
        
        '''

        #sampled version
        L = T.fmatrices()
        if self.error_type == 1:
            L = - T.sum(sampled_x * T.log(sampled_z) + (1 - sampled_x) * T.log(1 - sampled_z), axis=1)

        #square error, added by feng

        #print 'using'
        if self.error_type == 0:
            L = T.sum((sampled_x - sampled_z)**2, axis = 1)

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost,  updates)



def test_dA(my_s_type, my_error_type, dim_in, dim_out, learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=10, output_path='dA_plots', sample_method=0):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)


    # fix the random see
    numpy.random.seed(5)


    # shared version and numpy version
    [train_set_x, test_set_x,train_set, test_set] = datasets

    index_k = T.lscalar()


    # jump out the epochs
    threshold = 5

    # get the filter matrix for each batch, put in one list

    np_filter_matrix = create_all_filter(train_set, sample_method)
    all_filters = theano.shared(np_filter_matrix, borrow=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    #index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    #x = theano.shared((numpy.zeros(batch_size, train_set_x.get_value(borrow=True).shape[0]), dtype=theano.config.floatX), borrow=True)

    '''
    x = theano.shared(
                value=numpy.zeros(
                    (batch_size, dim_in),
                    dtype=theano.config.floatX
                ),
                name='x',
                borrow=True
            )

    '''
    #shape_info = T.vector('shape_info')
    #shape_info = theano.shared(numpy.array([3,100]))
    # end-snippet-2


    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #
    my_filter_matrix = T.matrix('my_filter_matrix')

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,

        #n_visible=28 * 28,
        #n_hidden=500

        # added by Feng, dimension specifics
        n_visible=dim_in,
        n_hidden=dim_out,
        s_type = my_s_type,
        error_type = my_error_type,


        filter_matrix=my_filter_matrix
    )

    # param added by Feng
    cost,  updates = da.get_cost_updates(
        corruption_level=0,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index_k],
        # return params also, Added by Feng
        # cost,
        cost,
        updates=updates,
        givens={
            x: train_set_x[index_k * batch_size: (index_k + 1) * batch_size],
            my_filter_matrix: all_filters[index_k * batch_size: (index_k + 1) * batch_size]
        }
    )




    m = T.matrix('m')
    mapped  = da.get_hidden_values(m)
    get_mapped = theano.function(
        [m],
        mapped
    )




    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############
    #final_y = numpy.empty([,])

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        #yy = [] # for each epoch the y will update in each batch

        count = 0
        for batch_index in xrange(n_train_batches):
            #print 'batch No.', str(batch_index), 'started'


            # set the filter for this batch
            # testshape_info.set_value([5,100])



            c.append(train_da(batch_index))
            count += 1


        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

        # immediately flush
        sys.stdout.flush()

        if numpy.mean(c) < threshold:
            break

        # update the filter each time
        if sample_method == 0:
            np_filter_matrix = create_all_filter(train_set)
            all_filters.set_value(np_filter_matrix)



    end_time = timeit.default_timer()



    training_time = (end_time - start_time)

    print 'ran for %.2fm' % ((training_time) / 60.)
    print 'total iterations:', epoch




    mapped_train = get_mapped(train_set_x.get_value(borrow=True))
    print 'get the mapped train data'

    mapped_test = get_mapped(test_set_x.get_value(borrow = True))
    print 'get the mapped test data'
    '''

    f = gzip.open(output_path, 'w')

    cPickle.dump([da.W.eval(), da.b.eval(), da.W_prime.eval(), da.b_prime.eval()], f)
    f.close()
    '''

    '''
    w_path = output_path + '_W'
    b_path = output_path + 'b'
    w_prime_path = output_path + 'W_prime'
    b_prime_path = output_path + 'b_prime'

    write_csr(da.W.eval(), w_path)
    write_csr(da.b.eval(), b_path)
    write_csr(da.W_prime.eval(), w_prime_path)
    write_csr(da.b_prime(), b_prime_path)

    print '\nW and b W_prime, b_primeare saved in***' + output_path
    '''
    #print '\n***yyy'
    #print mapped_train
    #print mapped_test
    return (mapped_train, mapped_test)

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''


    #############
    # LOAD DATA #
    #############


    print '... loading data'

    # Load the dataset


    f = gzip.open(dataset, 'rb')
    train_set, test_set = cPickle.load(f)
    #type(train_set)
    f.close()


    numpy.savetxt('tmpfile', train_set[0:10])

    #print 'aready written'


    '''
    train_set = numpy.random.rand(20, 10)
    test_set = numpy.random.rand(10, 10)

    '''
    shared_train = theano.shared(numpy.asarray(train_set, dtype=theano.config.floatX), borrow=True)

    shared_test = theano.shared(numpy.asarray(test_set, dtype=theano.config.floatX), borrow=True)

    return (shared_train, shared_test, train_set, test_set)


# create the filter matrix for every batch, placed in one list
# sample method 0: sample nz and same number of zeros
# sample method 1: sample only nz
def create_all_filter(train_set, sample_method):
    # create the filter, sample based on the non-zero features and the sample the sample number of non-zero features
        #m = T.sum(a, axis=0)

        all_nz = []
        all_zero = []
        all_filter = []

        row = []
        col = []
        data = []

        # fix the random seed


        #print self.input_shape.eval()

        all_row_num = train_set.shape[0]
        all_col_num = train_set.shape[1]




        i = 0

        while all_row_num > i:
            nz_this_row = []
            zeros_this_row = []
            j = 0
            #for j in range(all_col_num):
            while all_col_num > j:
            #while T.lt(j, all_col_num.eval()):
                elem = train_set[i, j]
                if elem == 0:  # errors here elem is always nonzero
                    zeros_this_row.append(j)
                else:
                    nz_this_row.append(j)
                #print 'inside ', str(j), 'iteration in', str(all_col_num)

                j += 1

            if sample_method == 0:
                len_sampled = len(nz_this_row)
                sampled_zeros_this_row = numpy.random.choice(zeros_this_row, size = len_sampled, replace=False)
                filtered_result_this_row = nz_this_row + sampled_zeros_this_row.tolist()
            elif sample_method == 1:
                # only use the nz filter
                filtered_result_this_row = nz_this_row



            # prepare to construct the sparse matrix
            for m in range(len(filtered_result_this_row)):
                row.append(i)
                col.append(filtered_result_this_row[m])
                data.append(float(1))


            # actually csr format
            #all_zero.append(zeros_this_row) # all of the zero index in each row
            #all_nz.append(nz_this_row) # all of the nz index in each row
            #all_filter.append(nz_this_row + sampled_zeros_this_row)

            #print 'the ', str(i), 'iteration finished'
            i += 1

        tmp_filter_matrix = csr_matrix((data, (row, col))).toarray()
        extra_col = train_set.shape[0] - tmp_filter_matrix.shape[1]

        #all_in_one_filters = numpy.zeros(train_set.shape)
        if extra_col > 0:
            all_in_one_filtes = numpy.hstack((tmp_filter_matrix, numpy.zeros((tmp_filter_matrix.shape[0], extra_col), dtype=tmp_filter_matrix.dtype)))
        else:
            all_in_one_filters = tmp_filter_matrix

        print 'sampling filter created'

        return all_in_one_filters
        #return theano.shared(all_in_one_filters, borrow=True)

# write the matrix as the csr form
def write_csr(matrix_x, path):
    buffer = ''
    f = open(path, 'w')
    num_rows = matrix_x.shape[0]
    num_cols = matrix_x.shape[1]

    buffer += str(num_rows) + ' ' +str(num_cols) + '\n'
    for i in range(num_rows):

        for j in range(num_cols):
            if matrix_x[i, j] != 0:
                buffer += '(' + str(i) + ' ' + str(j) + ')' + ' ' +str(matrix_x[i,j]) + '\n'


    print >>f, buffer
    f.close()


if __name__ == '__main__':
    s_type = 1
    error_type = 0
    learning_rate = 0.1
    dim_in = 10 #dimensions of input
    dim_out = 5 # dimension of output
    mypath = '../data/mnist.pkl.gz'

    training_epochs= 15
    batch_size= 10
    output_path='../W_b/model.pkl'

    for arg in sys.argv[1:]:
        if arg.startswith('-s='):
            s_type = int(arg[len('-s='):])

        if arg.startswith('-e='):
            error_type = int(arg[len('-e='):])

        if arg.startswith('-r='):
            learning_rate = float(arg[len('-r='):])

        if arg.startswith('-d='):
            dim_in = int(arg[len('-d='):])

        if arg.startswith('-o='):
            dim_out = int(arg[len('-o='):])

        if arg.startswith('-p='):
            mypath = arg[len('-p='):]

        if arg.startswith('-b='):
            batch_size = int(arg[len('-b='):])


    print 'configuration:'

    if s_type == 0:
        print '\tusing linear '
    elif s_type == 1:
        print '\tusing sigmoid'
    elif s_type == 2:
        print '\tusing tanh'

    if error_type == 0:
        print '\tusing Square error '
    elif error_type == 1:
        print '\tusing cross entropy error'

    print '\tlearning rate is', learning_rate

    print '\t#dimension of input is ', dim_in, ', #dimension of output is ', dim_out


    print '\tpath of raw file is', mypath

    print '\tbatch size = ', batch_size

    print '\ttraining epochs = ', training_epochs

    print '\n********************************\nlearning begin!'

    test_dA(s_type, error_type, dim_in, dim_out, learning_rate, training_epochs, mypath, batch_size, output_path)



    #test_dA(s_type, error_type, learning_rate, mydim ,mypath)
