# -*- coding: utf-8 -*-
from typing import Any, Callable
import warnings
import time

import numpy as np
try:
    import cupy as xp
except:
    import numpy as xp

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

from .utils import get_matrix_determinant
from .baseclasses import GPUobject

class Likelihood(GPUobject):
    """
    A class to represent the likelihood for a given dataset and model. This interface is designed specifically to work with the `Eryn` sampler (https://mikekatz04.github.io/Eryn/html/index.html).
    Attributes:
        data (DataContainer): Container for the data and related parameters.
        use_gpu (bool): Flag to indicate whether to use GPU for computations.
        xp (module): Numpy or CuPy module depending on the use_gpu flag.
        return_gpu (bool): Flag to indicate whether to return results on GPU.
        nchannels (int): Number of channels in the data.
        fmin (float): Minimum frequency for the analysis.
        fmax (float): Maximum frequency for the analysis.
        fullmatrix (bool): Flag to indicate whether to use full matrix operations.
        hermitian (bool): Flag to indicate whether the matrix is Hermitian.
        solve (function): Function to solve linear equations.
        nsource_wf_gen (int): Number of source waveform generators.
        psd_fn (function): Function to compute the power spectral density.
        noisekeys (list): List of keys for noise components.
        backgroundkeys (list): List of keys for background components.
        foregroundkeys (list): List of keys for foreground components.
        inf (float): Value to use for infinity in log likelihood calculations.
        rj (bool): Flag for reversible jump MCMC.
        nsubset (int): Number of subsets for parallel computation.
        tc_container (list): Container for `Eryn` parameter transforms.
    """
    def __init__(self, 
                    psd_fn,
                    data_container=None, 
                    source_wf_gen=None,
                    hermitian=True,
                    fullmatrix=False,
                    noisekeys=[],
                    backgroundkeys=[],
                    foregroundkeys=[],
                    inf=1e300,
                    use_gpu=True,
                    return_gpu=False,
                    nsubset=1,
                    rj=False,
                    **kwargs
                    ):
        
        """
        Initialize the Likelihood class.
        Parameters:
        -----------
        psd_fn : function
            Power spectral density function.
        t : array-like, optional
            Time array.
        d : array-like, optional
            Data array.
        freqs : array-like, optional
            Frequency array.
        dtilde : array-like, optional
            Fourier transformed data.
        fmin : float, optional
            Minimum frequency, default is 1e-4.
        fmax : float, optional
            Maximum frequency, default is 2.9e-2.
        weights : array-like, optional
            Weights for the data channels.
        source_wf_gen : function or list of functions, optional
            Source waveform generator(s).
        nchannels : int, optional
            Number of data channels, default is 3.
        average : bool, optional
            Whether to average the likelihood, default is False.
        hermitian : bool, optional
            Whether the covariance matrix is hermitian, default is True.
        fullmatrix : bool, optional
            Whether to use full matrix, default is False.
        f_segments : float, optional
            Frequency segments, default is 1e-5.
        window : tuple, optional
            Window function and its parameter, default is ('kaiser', 30).
        noisekeys : list, optional
            Keys for noise components.
        backgroundkeys : list, optional
            Keys for background components.
        foregroundkeys : list, optional
            Keys for foreground components.
        inf : float, optional
            Infinity value, default is 1e14.
        use_gpu : bool, optional
            Whether to use GPU, default is True.
        return_gpu : bool, optional
            Whether to return GPU arrays, default is False.
        nsubset : int, optional
            Number of subsets, default is 1.
        rj : bool, optional
            Whether to use RJMCMC, default is False.
        **kwargs : dict
            A
            dditional keyword arguments.
        """
        self.data = data_container

        self.use_gpu = use_gpu
        GPUobject.__init__(self, use_gpu=use_gpu)
        self.return_gpu = return_gpu

        self.fullmatrix = fullmatrix
        self.hermitian = hermitian 
        if self.hermitian:
            self.solve = jax.scipy.linalg.cho_solve
        else:
            self.solve = jnp.linalg.solve

        self.get_data_information(data_container)

        if source_wf_gen is None: # fit only for the psd (noise, stochastic components)    
            self.nsource_wf_gen = 0

        else:  # with deterministic signals we have to use the whittle likelihood
            if not isinstance(source_wf_gen, list):
                source_wf_gen = [source_wf_gen]

            self.nsource_wf_gen = len(source_wf_gen)

            if self.fullmatrix: 
                self.compute_logl = self.whittle_logl_full
            else:
                self.compute_logl = self.whittle_logl_diagonal

        self.psd_fn = psd_fn

        self.noisekeys = noisekeys
        self.backgroundkeys = backgroundkeys
        self.foregroundkeys = foregroundkeys

        self.setup_indeces()
        
        self.psd_fn.setup_frequency_dependences(self.data.freqs)

        self.inf = inf
        self.rj = rj
        self.nsubset = nsubset

        @property
        def tc_container(self):
            return self._tc_container
        
        @tc_container.setter
        def tc_container(self, tc_container=[None, None, None, None]):
            self._tc_container = tc_container

    def __call__(self, args, groups=None, **kwargs):
        """
        Evaluate the likelihood function.
        This method evaluates the likelihood function given the input arguments.
        Args:
            args (list or any): The input arguments. The order that `args` has to follow is 
                                [(templates), (noise), (backgrounds), (foregrounds)].
            groups (list or None, optional): The groups for the input arguments. Defaults to None.
            **kwargs: Additional keyword arguments.
        Returns:
            numpy.ndarray: The evaluated log-likelihood values.
        """
        if not isinstance(args, list):
            args = [args]

        if groups is None:
            groups = [np.arange(np.atleast_2d(args[0]).shape[0])]

        if not isinstance(groups, list):
            groups = [groups]

        unique_groups = np.unique(np.concatenate([groups_i for groups_i in groups]))
        ngroups = unique_groups.max() + 1 if unique_groups.shape[0] > 0 else 0
        
        wf_args_all, noise_args_all, background_args_all, foreground_args_all = self.unpack_args(args)
        wf_groups_all, noise_groups_all, background_groups_all, foreground_groups_all = self.unpack_groups(groups)

        logl_all = []

        subset = int(ngroups / self.nsubset) if ngroups > self.nsubset else ngroups

        inds_all = np.arange(0, ngroups + 1, subset)

        if inds_all[-1] < ngroups:
            inds_all = np.concatenate([inds_all, np.array([ngroups])])

        for i in range(len(inds_all) - 1):
            wf_args, noise_args, background_args, foreground_args = [], [], [], []
            wf_groups, noise_groups, background_groups, foreground_groups = [], [], [], []

            for j in range(self.nsource_wf_gen):
                inds = np.where((wf_groups_all[j] >= inds_all[i]) & (wf_groups_all[j] < inds_all[i + 1]))
                wf_args += [wf_args_all[j][inds]]
                wf_groups += [wf_groups_all[j][inds]]

            for j in range(len(self.noisekeys)):
                inds = np.where((noise_groups_all[j] >= inds_all[i]) & (noise_groups_all[j] < inds_all[i + 1]))
                noise_args += [noise_args_all[j][inds]]
                noise_groups += [noise_groups_all[j][inds]]

            for j in range(len(self.backgroundkeys)):
                inds = np.where((background_groups_all[j] >= inds_all[i]) & (background_groups_all[j] < inds_all[i + 1]))
                background_args += [background_args_all[j][inds]]
                background_groups += [background_groups_all[j][inds]]

            for j in range(len(self.foregroundkeys)):
                inds = np.where((foreground_groups_all[j] >= inds_all[i]) & (foreground_groups_all[j] < inds_all[i + 1]))
                foreground_args += [foreground_args_all[j][inds]]
                foreground_groups += [foreground_groups_all[j][inds]]

            psd = self.psd_fn(self.data.freqs,
                                   noise_args,
                                   background_args,
                                   foreground_args,
                                   noise_groups,
                                   background_groups,
                                   foreground_groups,
                                   **kwargs)
            
            
            if self.nsource_wf_gen > 0:
                h = self.xp.zeros(shape=(self.freqs[0]))

                for source, wf_args_i in zip(self.source_wf_gen, wf_args):

                    h += source(wf_args_i)

                n = self.d - h
                ntilde = self.data.get_Xtilde(n)
                ntildentilde = self.data.get_XtildeXtilde(ntilde)
                
            else:
                ntilde = self.data.dtilde[self.xp.newaxis, :, :]
                ntildentilde = self.data.dtildedtilde

            if self.use_gpu:
                mempool = xp.get_default_memory_pool()
                mempool.free_all_blocks()

            logl = self.compute_logl(psd, ntilde, ntildentilde).real
            logl_all.append(logl)

        logl_out = np.concatenate(logl_all)
        
        # check if all the values are finite
        logl_out[~np.isfinite(logl_out)] = -self.inf

        # if np.any(np.isnan(logl_out)):
        #     breakpoint()
        if self.return_gpu:
            return logl_out
        else:
            try:
                return logl_out.get()
            except:
                return logl_out
            
    def get_data_information(self, data_container):
        """
        Get the data information from the data container.

        Args:
            data_container (DataContainer): The data container.
        """
        self.nchannels = data_container.nchannels
        self.fmin = data_container.fmin
        self.fmax = data_container.fmax

        if data_container.averaged: # without signal we can use an averaged likelihood
            if self.fullmatrix:
                self.compute_logl = self.wishart_logl_full
            else:
                self.compute_logl = self.wishart_logl_diagonal
            
        else:
            if self.fullmatrix:
                self.compute_logl = self.whittle_logl_full
            else:
                self.compute_logl = self.whittle_logl_diagonal


        
    def setup_indeces(self):
        """
        Set up the indeces for the different components.
        """

        self.idx_wf = 0
        self.idx_noise = self.nsource_wf_gen
        self.idx_background = self.idx_noise + len(self.noisekeys)
        self.idx_foreground = self.idx_background + len(self.backgroundkeys)
        self.indeces = [self.idx_wf, self.idx_noise, self.idx_background, self.idx_foreground]
        
    def unpack_args(self, args):
        """
        Unpacks the arguments into separate components. Transforms the parameters if necessary.

        Args:
            args (list): The list of arguments to be unpacked.

        Returns:
            tuple: A tuple containing the unpacked components: wf_args, noise_args, background_args, foreground_args.
        """
        wf_args, noise_args, background_args, foreground_args = [], [], [], []
        components = [wf_args, noise_args, background_args, foreground_args]
        indeces = self.indeces + [len(args)]
        
        for i in range(len(components)):
            if self.tc_container[i] is not None:
                components[i] += [self.tc_container[i][j].both_transforms(arg) for j,arg in enumerate(args[indeces[i] : indeces[i+1]])]
            else:
                components[i] += args[indeces[i] : indeces[i+1]]

        return wf_args, noise_args, background_args, foreground_args
    
    def unpack_groups(self, groups):
        wf_groups, noise_groups, background_groups, foreground_groups = [], [], [], []
        components = [wf_groups, noise_groups, background_groups, foreground_groups]
        indeces = self.indeces + [len(groups)]
        
        for i in range(len(components)):
            components[i] += groups[indeces[i] : indeces[i+1]]

        return wf_groups, noise_groups, background_groups, foreground_groups
    

    @partial(jax.jit, static_argnums=(0,))
    def lognormal_logl_full(self, psd, ntilde, ntildentilde):
        """
        Compute the log likelihood for the log-normal likelihood.

        Args:
            psd (array): The power spectral density.
            ntilde (array): The residual data in the frequency domain.
            ntildentilde (array): The residual data in the frequency domain times its complex conjugate traspose.

        Returns:
            array: The log likelihood.
        """
        cov = psd
        logl = - jnp.sum( self.data.weights * (ntildentilde / cov + jnp.log(cov)),  axis = (1, 2))

        return logl

    @partial(jax.jit, static_argnums=(0,))
    def whittle_logl_full(self, psd, ntilde, ntildentilde):
        """
        Compute the log likelihood for the Whittle likelihood.

        Args:
            psd (array): The power spectral density.
            ntilde (array): The residual data in the frequency domain.
            ntildentilde (array): The residual data in the frequency domain times its complex conjugate traspose.

        Returns:
            array: The log likelihood.
        """

        to_solve, logdet = get_matrix_determinant(psd, hermitian=self.hermitian, return_mat=False)

        ntilde_rep = jnp.repeat(ntilde, psd.shape[0], axis=0)
        ntildeconj_invcov = self.solve(to_solve, jnp.conj(ntilde_rep)[:, :, :])
    
        ntildentilde = jnp.einsum('ijk,ijk -> ij', ntildeconj_invcov, ntilde)
        logl = - jnp.sum(ntildentilde + logdet, axis=-1)

        return logl

    @partial(jax.jit, static_argnums=(0,))
    def whittle_logl_diagonal(self, psd, ntilde, ntildentilde):
        """
        Compute the log likelihood for the Whittle likelihood.

        Args:
            psd (array): The power spectral density.
            ntilde (array): The residual data in the frequency domain.
            ntildentilde (array): The residual data in the frequency domain times its complex conjugate traspose.

        Returns:
            array: The log likelihood.
        """

        cov = psd
        logl = - jnp.sum( self.data.weights * (ntildentilde / cov + jnp.log(cov)),  axis = (1, 2))

        return logl
    

    @partial(jax.jit, static_argnums=(0,))
    def wishart_logl_full(self, psd, *args, **kwargs) :
        """
        Compute the log likelihood for the Wishart likelihood. 
        Since the Wishart likelihood is based on averaging over data segments, it cannot be used when including deterministic signals.

        Args:
            psd (array): The power spectral density.
        
        Returns:
            array: The log likelihood.
        """

        cov, logdet = get_matrix_determinant(psd, hermitian=self.hermitian, return_mat=True)
        invcov = jnp.linalg.inv(cov)
        del cov

        return -  jnp.sum(jnp.einsum('...ii', jnp.einsum('...ij, ...jk->...ik', invcov, self.data.Y)) + self.data.nu * logdet, axis=-1) 


    @partial(jax.jit, static_argnums=(0,))
    def wishart_logl_diagonal(self, psd, *args, **kwargs):
        """
        Compute the log likelihood for the Wishart likelihood. 
        Since the Wishart likelihood is based on averaging over data segments, it cannot be used when including deterministic signals.

        Args:
            psd (array): The power spectral density.
        
        Returns:
            array: The log likelihood.
        """
        
        cov = psd
        return - jnp.sum(self.data.Y / cov + self.data.nu[None, :, None] * jnp.log(cov), axis=(1,2)) #+ self.norm

