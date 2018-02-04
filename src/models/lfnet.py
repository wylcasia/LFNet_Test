"""
--------LFNet-----------
Author: Yunlong Wang
Date: 2018.01
------------------------
"""
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet import conv2d
from collections import OrderedDict

__all__ = ['LFNet','load_model','config']

"""
a random number generator used to initialize weights
"""
SEED = 123
rng = np.random.RandomState(SEED)


class ConvLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape, std = 1e-3):
        """
        Allocate a c with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type name: str
        :param name: given a special name for the ConvPoolLayer
        """

        # self.filter_shape = filter_shape
        # self.image_shape = image_shape
        # self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
        #            np.prod(poolsize))
        # initialize weights with random weights

        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.normal(0, std, size=filter_shape),
                dtype=config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(np.zeros(filter_shape[0]).astype(config.floatX), borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def conv(self, input):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            border_mode='half'
        )
        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        return output

def prefix_p(prefix, params):
    tp = OrderedDict()
    for kk, pp in params.items():
        tp['%s_%s' % (prefix, kk)] = params[kk]
    return tp

class LFNet(object):
    """LFNet"""

    def __init__(self, options):
        self.options = options

    def build_net(self, model):
        options = self.options
        # forward & backward data flow
        x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='x')

        # forward net
        IMsF_layer_f, IMsF_params_f = self._init_IMsF(options)
        x_f = self._build_IMsF(x, options, IMsF_layer_f, IMsF_params_f)
        brcn_layers_f, brcn_params_f = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        proj_f = self._build_model(x_f, options, brcn_layers_f, brcn_params_f, go_backwards=False)
        params_f = dict(IMsF_params_f, **brcn_params_f)

        # backward net
        IMsF_layer_b, IMsF_params_b = self._init_IMsF(options)
        x_b = self._build_IMsF(x, options, IMsF_layer_b, IMsF_params_b)
        brcn_layers_b, brcn_params_b = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        proj_b = self._build_model(x_b, options, brcn_layers_b, brcn_params_b, go_backwards=True)
        params_b = dict(IMsF_params_b, **brcn_params_b)

        params = dict(prefix_p('f', params_f), **(prefix_p('b', params_b)))

        if model is not None:
            for k in params.iterkeys():
                params[k].set_value(model[k])

        proj = (proj_f + proj_b[::-1])/2.0

        f_x = theano.function([x], proj, name='f_proj')

        return f_x


    def _init_IMsF(self, options):

        layers = OrderedDict()
        params = OrderedDict()

        IMsF_shape = options['IMsF_shape']

        for i in range(len(IMsF_shape)):
            # print('IMsF_'+str(i))
            layers['IMsF_' + str(i)] = ConvLayer(IMsF_shape[i],1e-1)
            params['IMsF_' + str(i) + '_w'] = layers['IMsF_' + str(i)].params[0]
            params['IMsF_' + str(i) + '_b'] = layers['IMsF_' + str(i)].params[1]
            params['IMsF_' + str(i) + '_rescale'] = theano.shared(np.ones(IMsF_shape[i][0]).astype(config.floatX),
                                                                  borrow=True)

        return layers, params

    def _init_layer(self, filter_shape, rec_filter_size):
        """
        Global (net) parameter. For the convolution and regular opt.
        """
        layers = OrderedDict()
        params = OrderedDict()


        for i in range(len(filter_shape)):
            layers['conv_' + str(i) + '_v'] = ConvLayer(filter_shape[i])
            layers['conv_' + str(i) + '_t'] = ConvLayer(filter_shape[i])
            params['conv_' + str(i) + '_v_w'] = layers['conv_' + str(i) + '_v'].params[0]
            params['conv_' + str(i) + '_v_b'] = layers['conv_' + str(i) + '_v'].params[1]
            params['conv_' + str(i) + '_t_w'] = layers['conv_' + str(i) + '_t'].params[0]
            params['conv_' + str(i) + '_t_b'] = layers['conv_' + str(i) + '_t'].params[1]

            if i < len(rec_filter_size):
                layers['conv_' + str(i) + '_r'] = ConvLayer(rec_filter_size[i])
                params['conv_' + str(i) + '_r_w'] = layers['conv_' + str(i) + '_r'].params[0]
                params['conv_' + str(i) + '_r_b'] = layers['conv_' + str(i) + '_r'].params[1]

            params['b_' + str(i)] = theano.shared(np.zeros(filter_shape[i][0]).astype(config.floatX), name='b_' + str(i), borrow=True)


        return layers, params

    def _build_IMsF(self, input, options, layers, params):

        def _step(x_,layer_):
            layer_ = str(layer_.data)
            # print(layer_)
            h_ = layers['IMsF_'+str(layer_)].conv(x_)
            h_ = tensor.nnet.relu(h_)
            return h_

        rval = input
        _rval = 0.0

        for i in range(len(options['IMsF_shape'])):
            rval, _ = theano.scan(_step, sequences=[rval],
                                  non_sequences=[i],
                                  name='IMsF_layers_' + str(i))
            _rval += rval \
                     * params['IMsF_' + str(i) + '_rescale'].dimshuffle('x','x',0,'x','x')

        proj = _rval

        return proj

    def _build_model(self, input, options, layers, params, go_backwards=False):

        def _step1(x_, t_, layer_):
            layer_ = str(layer_.data)
            v = layers['conv_' + layer_ + '_v'].conv(x_)
            t = layers['conv_' + layer_ + '_t'].conv(t_)
            h = v + t

            return x_, h

        def _step2(h, r_, layer_):
            layer_ = str(layer_.data)
            o = h + params['b_' + layer_].dimshuffle('x', 0, 'x', 'x')
            if layer_ != str(len(options['filter_shape']) - 1):
                r = layers['conv_' + layer_ + '_r'].conv(r_)
                o = tensor.nnet.relu(o + r)
            return o

        rval = input
        if go_backwards:
            rval = rval[::-1]

        for i in range(len(options['filter_shape'])):
            rval, _ = theano.scan(_step1, sequences=[rval],
                                  outputs_info=[rval[0], None],
                                  non_sequences=[i],
                                  name='rnn_layers_k_' + str(i))
            rval = rval[1]
            rval, _ = theano.scan(_step2, sequences=[rval],
                                  outputs_info=[rval[-1]],
                                  non_sequences=[i],
                                  name='rnn_layers_q_' + str(i))
        # diff = options['padding']
        proj = rval \
               # + input[:,:,:,diff:-diff,diff:-diff]

        return proj

def pred_error(f_pred,data,target):

    x = data
    y = target
    pred = f_pred(x)

    pred = np.round(pred * 255.0)
    y = np.round(y * 255.0)

    z = np.mean((y - pred) ** 2)
    #
    # z /= x.shape[0] * x.shape[3] * x.shape[4]
    rmse = np.sqrt(z)
    # print('RMSE: ',rmse.eval())
    psnr = 20 * np.log10(255.0 / rmse)
    # psnr = tensor.sum(psnr)

    return psnr


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def load_model(path):
    npy = np.load(path)
    return npy.all()






