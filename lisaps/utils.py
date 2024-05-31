import jax
import jax.numpy as jnp
from functools import partial
import warnings
import h5py
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

import numpy as np

from .baseclasses import BaseNoise, GPUobject
from .stochasticbackgrounds import StochasticContribution

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



class DataGenerator(GPUobject):
    
    def __init__(self, psd_fn, domain='frequency', use_gpu=False):
        '''
        Initialize the Utils class.

        Args:
            psd_fn (callable): A callable to generate the desired PSD given an array of frequencies.
            domain (str, optional): The domain of the data. Must be either "frequency" or "time". Defaults to "frequency".
            use_gpu (bool, optional): Flag indicating whether to use GPU acceleration. Defaults to False.
        '''
        super().__init__(use_gpu=use_gpu)
        self.psd_fn = psd_fn

        assert domain in ['frequency', 'time'], 'Domain must be either "frequency" or "time".'
        self.domain = domain

    def __call__(self, 
                 psd_args=[], 
                 psd_kwargs={},
                 freqs=None, 
                 N=None,
                 dt=None,
                 normalize=True, 
                 return_jax=False
                 ):
        '''
        Generate samples from the given PSD function.

        Parameters:
        - psd_args (list): List of positional arguments to be passed to the PSD function.
        - psd_kwargs (dict): Dictionary of keyword arguments to be passed to the PSD function.
        - freqs (array-like, optional): Array of frequencies. If not provided, it will be calculated based on N and dt.
        - N (int, optional): Number of samples. Required if freqs is not provided.
        - dt (float, optional): Sampling rate. Required if freqs is not provided.
        - normalize (bool, optional): Whether to normalize the generated samples. Default is True.
        - return_jax (bool, optional): Whether to return the samples as a JAX array. Default is False.

        Returns:
        - samples (array-like): Generated samples.

        Raises:
        - ValueError: If neither freqs nor N and dt are provided.
        '''

        if freqs is None and N is None and dt is None:
            raise ValueError('Either provide the frequencies or the number of samples and the sampling rate.')
        if freqs is None:
            freqs = self.xp.fft.rfftfreq(N, dt)
            df = self.xp.diff(freqs)[0]
            freqs[0] = freqs[1]
        else:
            df = self.xp.diff(freqs)[0]

        psd_kwargs['freqs'] = freqs 
 
        psd = self.psd_fn(*psd_args, **psd_kwargs).astype(self.xp.float64) #! working only with AET

        if len(psd.shape) > 2:
            psd = psd[0]

        if normalize:
            norm = 1.0 / (4.0 * df)
        else:
            norm = 1.0
        
        std_dev = self.xp.sqrt( psd * norm)

        if self.domain == 'frequency':
            samples = [self.xp.random.normal(0, std_dev[:, i], len(std_dev[:, i]))+ 1j * self.xp.random.normal(0, std_dev[:, i], len(std_dev[:, i]))  for i in range(std_dev.shape[1])]
            samples = self.xp.array(samples).T
            samples = self.xp.concatenate((freqs[:, None], samples), axis=1)

        else:
            samples = [self.xp.fft.irfft(self.xp.random.normal(0, std_dev[:, i], len(std_dev[:, i]))+ 1j * self.xp.random.normal(0, std_dev[:, i], len(std_dev[:, i]))) for i in range(std_dev.shape[1])]
            samples = self.xp.array(samples).T
            times = self.xp.arange(0, N*dt, dt)
            samples = self.xp.concatenate((times[:, None], samples), axis=1)

        if return_jax:
            samples = jnp.asarray(samples)
        
        return samples

    def generate_datasets(self, 
                          filename,
                          n_datasets=1,
                          extension='h5',
                          channels='AET',
                          skip=0,
                          args=[],
                          kwargs={}):
        '''
        Generate data and save it to a file.

        Parameters:
        - filename (str): The base name of the file to be saved.
        - n_datasets (int): The number of datasets to generate.
        - extension (str): The file extension to use for saving the data. Default is 'h5'.
        - channels (str): The type of channels to generate. Must be either 'AET' or 'XYZ'. Default is 'AET'.
        - skip (int): The number of letters to skip when naming the datasets. Default is 0.
        - args, kwargs: Additional arguments to be passed to the data generation function.

        Raises:
        - ValueError: If channels is neither 'AET' nor 'XYZ'.

        Returns:
        - None
        '''

        if channels == 'AET':
            keys = ['A2', 'E2', 'T2']
        elif channels == 'XYZ':
            keys = ['X2', 'Y2', 'Z2']
        else:
            raise ValueError('Channels must be either "AET" or "XYZ".')
        
        first_letter = int(97 + skip)
        
        for i in tqdm(range(n_datasets)):
            savename = f'{filename}' + chr(first_letter + i) + f'.{extension}'
            samples = self.__call__(*args, **kwargs)
            
            if extension == 'npy':
                np.save(savename, samples)

            elif extension == 'npz':
                np.savez(savename, TDIs=samples)

            elif extension == 'h5':
                with h5py.File(savename, 'w') as hdf:
                    if self.domain == 'frequency':
                        hdf['freqs'] = samples[:, 0]
                    else:
                        hdf['t'] = samples[:, 0]

                    for j, key in enumerate(keys):
                        hdf[key] = samples[:, j+1]
                    
                    hdf.close()


class NoiseGenerator(DataGenerator):
    def __init__(self, noise_kwargs, domain='frequency', use_gpu=False):
        '''
        Initialize the NoiseGenerator class.

        Args:
        - noise_kwargs (dict): Dictionary of keyword arguments to be passed to the Noise class.
        - domain (str, optional): The domain of the data. Must be either "frequency" or "time". Defaults to "frequency".
        - use_gpu (bool, optional): Flag indicating whether to use GPU acceleration. Defaults to False.
        '''
        
        psd_fn = BaseNoise(**noise_kwargs).set_PSDS

        super().__init__(psd_fn, domain=domain, use_gpu=use_gpu)


class SignalGenerator(DataGenerator):
    def __init__(self, signal_kwargs, domain='frequency', use_gpu=False):
        '''
        Initialize the SignalGenerator class.

        Args:
        - signal_kwargs (dict): Dictionary of keyword arguments to be passed to the StochasticContribution class.
        - domain (str, optional): The domain of the data. Must be either "frequency" or "time". Defaults to "frequency".
        - use_gpu (bool, optional): Flag indicating whether to use GPU acceleration. Defaults to False.
        '''
        signal_fn = StochasticContribution(**signal_kwargs).TDI_background#.backgrounds_fn[0]

        super().__init__(signal_fn, domain=domain, use_gpu=use_gpu)








# helper functions
@jax.vmap
@jax.vmap
@jax.jit
def fill_diagonal(x):
    return jnp.diag(x[:3])
            
@jax.vmap
@jax.vmap
@jax.jit
def fill_lower_triangle(x):
    '''
    Fill the lower triangle of a 3x3 matrix with the given values.
    '''
    x11, x22, x33, x12, x13, x23 = x

    l11 = jnp.sqrt(x11)
    l21 = x12 / l11
    l31 = x13 / l11
    l22 = jnp.sqrt(x22 - l21**2)
    l32 = (x23 - l21 * l31) / l22
    l33 = jnp.sqrt(x33 - l31**2 - l32**2)

    return jnp.array([[l11, 0, 0], [l21, l22, 0], [l31, l32, l33]])

@jax.vmap
@jax.vmap
@jax.jit
def fill_upper_triangle(x):
    '''
    Fill the upper triangle of a 3x3 matrix with the given values.
    '''
    x11, x22, x33, x12, x13, x23 = x

    u11 = jnp.sqrt(x11)
    u12 = x12 / u11
    u13 = x13 / u11
    u22 = jnp.sqrt(x22 - u12**2)
    u23 = (x23 - u12 * u13) / u22
    u33 = jnp.sqrt(x33 - u13**2 - u23**2)

    return jnp.array([[u11, u12, u13], [0, u22, u23], [0, 0, u33]])

def get_matrix_determinant(x, hermitian=True, return_mat=False):
    '''
    Calculate the parts of the covariance matrix useful for the likelihood evaluation. These are the input of the `solve` method chosen and the determinant of the covariance matrix.

    Parameters:
    - x: The input matrix.
    - hermitian: A boolean value indicating whether the matrix is Hermitian. Default is True.

    Returns:
    If hermitian is True:
    - (l, True): a tuple containing the lower triangle of the matrix and `True`, on the same line of `scipy.linalg.cho_factor()`.
    - det: The determinant of the covariance matrix. 

    If hermitian is False:
    - cov: The covariance matrix obtained from x.
    - det: The determinant of the covariance matrix.
    '''

    if hermitian:
        l = fill_lower_triangle(x)
    
        logdet = logdet_from_triangle(l)

        if return_mat:
            u = fill_upper_triangle(x)
            out = l @ u

        else:
            out = (l, True)

        return out, logdet

    else:
        cov = fill_covmat(x)
        logdet = jnp.log(jnp.linalg.det(cov))

        return cov, logdet
    

@jax.vmap
@jax.vmap
@jax.jit
def fill_covmat(x):
    '''
    Fill a 3x3 generic covariance matrix with the given values.

    Args:
    - x (array-like): The values to fill the covariance matrix with.

    Returns:
    - covmat (array-like): The filled covariance matrix.
    '''
    x11, x22, x33, x12, x13, x23, x21, x31, x32 = x

    return jnp.array([[x11, x12, x13], [x21, x22, x23], [x31, x32, x33]])

@jax.vmap
@jax.vmap
@jax.jit
def logdet_from_triangle(x):
    
    """
    Compute the determinant of a 3x3 matrix given its lower triangle. 

    Args:
    - x (array-like): The values of the lower triangle of the matrix.

    Returns:
    - det (float): The determinant of the matrix.
    """

    diag = jnp.diag(x)
    det = 2 * jnp.sum(jnp.log(diag))

    return det


