import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

import numpy as np

from .baseclasses import BaseNoise

'''
script to compute the Fisher information matrix for a given likelihood function. for the moment, we will only consider the instrumental noise case.
'''

class FisherMatrix(BaseNoise):

    def __init__(self,  
                 asdTM=2.4e-15, 
                 asdOMS=7.9e-12, 
                 fkneeTM=0.4e-3, 
                 fkneeOMS=2e-3, 
                 equal_arms=False, 
                 fs=None, 
                 Ncov=None, 
                 channels='AET', 
                 use_gpu=False, 
                 units='strain', 
                 interpkwargs=dict(kind='akima', axis=1),
                 likelihood='whittle'
                 ):
        '''
        Initialize the Fisher matrix object.
        '''
        super().__init__(asdTM=asdTM, 
                         asdOMS=asdOMS, 
                         fkneeTM=fkneeTM, 
                         fkneeOMS=fkneeOMS, 
                         equal_arms=equal_arms, 
                         fs=fs, 
                         Ncov=Ncov, 
                         channels=channels, 
                         use_gpu=use_gpu, 
                         units=units, 
                         interpkwargs=interpkwargs
                         )
        

        self.likelihood = likelihood #! not sure if it is the same for the wishart and whittle likelihoods
        if self.likelihood != 'whittle':
            raise ValueError('Only the Whittle likelihood is currently supported.')

        self.derivatives_TM = [jax.vmap(jax.grad(self.available_functions[channel], argnums=0), in_axes=(None, None, 0)) for channel in self.channels]
        self.derivatives_OMS = [jax.vmap(jax.grad(self.available_functions[channel], argnums=1), in_axes=(None, None, 0)) for channel in self.channels]

        #todo: covariance matrix
        #todo: summation


    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, freqs, nparams=2):
        '''
        Compute the Fisher information matrix for a given set of frequencies.
        '''

        Fim = jnp.zeros((nparams, nparams))


        for i in range(nparams):
            for j in range(nparams):
                sum_channels = 0.0
                for k, channel in enumerate(self.channels):
                    dC_dthetai = jax.vmap(jax.grad(self.available_functions[channel], argnums=i), in_axes=(None, None, 0))(self.asdTM, self.asdOMS, freqs)
                    dC_dthetaj = jax.vmap(jax.grad(self.available_functions[channel], argnums=j), in_axes=(None, None, 0))(self.asdTM, self.asdOMS, freqs)

                    inv_C = 1.0 / self.psd[:, k]

                    sum_channels += jnp.sum(inv_C**2 * dC_dthetai * dC_dthetaj)

                Fim = Fim.at[i, j].set(sum_channels)

        return Fim


