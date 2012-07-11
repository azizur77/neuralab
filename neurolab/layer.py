# -*- coding: utf-8 -*-
"""
The module contains the basic layers architectures

"""
from core import Layer
import init
import trans

import numpy as np


class Perceptron(Layer):
    """
    Perceptron Layer class
    
    :Parameters:
        ci: int
            Number of input
        cn: int
            Number of neurons
        transf: callable
            Transfer function
    
    :Example:
        >>> import neurolab as nl
        >>> # create layer with 2 inputs and 4 outputs(neurons)
        >>> l = Perceptron(2, 4, nl.trans.PureLin())
    
    """
    
    def __init__(self, ci, cn, transf):

        Layer.__init__(self, ci, cn, cn, {'w':(cn, ci), 'b': cn})
        
        self.transf = transf
        if not hasattr(transf, 'out_minmax'):
            Inf = 1e100
            self.out_minmax = np.array([(self.transf(-Inf), self.transf(Inf))] * self.co)
        else:
            self.out_minmax = np.array([np.asfarray(transf.out_minmax)] * self.co)
        
        # default init function
        self.initf = init.initwb_reg
        self.s = np.zeros(self.cn)

    def _step(self, inp):
        self.s = np.sum(self.np['w'] * inp, axis=1)
        self.s += self.np['b']
        return self.transf(self.s)


class Convolution(Layer):
    """
    Convolutional Perceptron layer

    :Parameter
        ci: int
            Number of input
        cb: int
            Number of inputs in each batch
        cbo: int
            Number of outputs of each batch
        shift: int
            Number of neurons shifted after each batch
        transf: callable
            Transfer function

        :Example:
            >>> import neurolab as nl
            >>> # Create a convolutional layer with 10 inputs, 3 output, 3
                inputs in each batch.
            >>> l = Convolution(10, 3, 3, nl.trans.PureLin())
            
    """
    def __init__(self, ci, cb, transf, cbo=1, shift=1):
        
        Layer.__init__(self, ci, cn, cn, {'w': (cn, cb), 'b': cn})
        self.cb = cb
        self.shift = shift
        self.transf = transf

    def _step(self, inp):
        slices = self.slice(inp, self.cb, self.shift)
        self.s = np.sum(self.np["w"] * slices, xaxis=1)
        self.s += self.np["b"]
        return self.transf(self.s)

    @classmethod
    def slice(inp, cb, shift):
        assert len(inp.shape) == 1
        left = 0
        result = []
        while left <= inp.size - cb:
            result.append(inp[left:left + cb])
            left += shift
        return np.concatenate(result)
            
>>>>>>> a07f0b8d73a474e9a694cc7855293237c05d9bc7

class Competitive(Layer):
    """ 
    Competitive Layer class
    
    :Parameters:
        ci: int
            Number of input
        cn: int
            Number of neurons
        distf: callable default(euclidean)
            Distance function
    
    """
    
    def __init__(self, ci, cn, distf=None):
        Layer.__init__(self, ci, cn, cn, {'w': (cn, ci), 'conscience': cn})
        self.transf = trans.Competitive()
        self.initf = init.midpoint
        self.out_minmax[:] = np.array([self.transf.out_minmax] * cn)
        self.np['conscience'].fill(1.0)
        def euclidean(A, B):
            """
            Euclidean distance function.
            See scipi.spatial.distance.cdist()
            
            :Example:
                >>> import numpy as np
                >>> euclidean(np.array([0,0]), np.array([[0,1], [0, 5.5]])).tolist()
                [1.0, 5.5]
                
            """
            return np.sqrt(np.sum(np.square(A-B) ,axis=1))
        
        self.distf = euclidean
    
    def _step(self, inp):

        d = self.distf(self.np['w'], inp.reshape([1,len(inp)]))
        self.last_dist = d
        out = self.transf(self.np['conscience'] * d)
<<<<<<< HEAD
        return out
=======
        return out
>>>>>>> a07f0b8d73a474e9a694cc7855293237c05d9bc7
