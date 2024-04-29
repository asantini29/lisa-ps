import jax
import jax.numpy as jnp
from functools import partial
import warnings

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
                 nparams=2
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
        
        self.nparams = nparams

    @partial(jax.jit, static_argnums=(0,))
    def compute_fisher_matrix(self, freqs, nu=1):
        '''
        Compute the Fisher information matrix for a given set of frequencies.
        '''

        Fim = jnp.zeros((self.nparams, self.nparams))
        self.psd = self.get_PSDS(self.asdTM, self.asdOMS, freqs, squeeze=True)


        for i in range(self.nparams):
            for j in range(self.nparams):
                sum_channels = 0.0
                for k, channel in enumerate(self.channels):
                    dC_dthetai = jax.vmap(jax.grad(self.available_functions[channel], argnums=i), in_axes=(None, None, 0))(self.asdTM, self.asdOMS, freqs)
                    dC_dthetaj = jax.vmap(jax.grad(self.available_functions[channel], argnums=j), in_axes=(None, None, 0))(self.asdTM, self.asdOMS, freqs)

                    inv_C = 1.0 / self.psd[:, k]

                    sum_channels += jnp.sum(inv_C**2 * dC_dthetai * dC_dthetaj * nu)

                Fim = Fim.at[i, j].set(sum_channels)

        return Fim
    
    def compute_covariance_matrix(self, freqs, nu=1):
        '''
        Compute the covariance matrix for a given set of frequencies.
        '''
        if nu == 1:
            warnings.warn('Calculating the Fisher matrix for nu=1. This only applies to the Whittle likelihood function, if using the Whishart likelihood, provide the effective number of DoFs per frequency bin.')
        FIM = self.compute_fisher_matrix(freqs, nu)
        cov = jnp.linalg.inv(FIM)
        return cov


