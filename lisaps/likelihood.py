# -*- coding: utf-8 -*-
from pysco import utils
from typing import Any, Callable
import warnings

import numpy as np
from scipy import signal
try:
    import cupy as xp
except:
    import numpy as xp

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

from .baseclasses import DataContainer
from .utils import get_matrix_determinant

class Likelihood:

    def __init__(self, 
                    psd_fn, 
                    t=None,
                    d=None,
                    freqs=None,
                    dtilde=None,
                    fmin=1e-4,
                    fmax=2.9e-2,
                    weights=None,
                    source_wf_gen=None,
                    nchannels=3,
                    average=False,
                    hermitian=True,
                    fullmatrix=False,
                    f_segments=1e-5,
                    window=('kaiser', 30),
                    noisekeys=[],
                    backgroundkeys=[],
                    foregroundkeys=[],
                    inf=1e14,
                    use_gpu=True,
                    return_gpu=False,
                    nsubset=1,
                    rj=False,
                    **kwargs
                    ):
        '''
        TODO: write docstring
        '''

        self.data = DataContainer(
            t=t,
            d=d,
            freqs=freqs,
            dtilde=dtilde,
            weights=weights,
            nchannels=nchannels,
            fmin=fmin,
            fmax=fmax,
            average=average,
            f_segments=f_segments,
            fullmatrix=fullmatrix,
            window=window,
            use_gpu=use_gpu
        )

        self.use_gpu = use_gpu
        self.xp = xp if self.use_gpu else np
        self.return_gpu = return_gpu

        self.nchannels = nchannels
        self.fmin = fmin
        self.fmax = fmax

        self.fullmatrix = fullmatrix
        self.hermitian = hermitian 
        if self.hermitian:
            self.solve = jax.scipy.linalg.cho_solve
        else:
            self.solve = jnp.linalg.solve

        if source_wf_gen is None: # fit only for the psd (noise, stochastic components)
            
            self.nsource_wf_gen = 0

            if average: # without signal we can use an averaged likelihood
                #breakpoint()
                self.compute_logl = self.wishart_logl
            else:
                self.compute_logl = self.whittle_logl

        else:                
                
            if not isinstance(source_wf_gen, list):
                source_wf_gen = [source_wf_gen]

            self.nsource_wf_gen = len(source_wf_gen)

        self.psd_fn = psd_fn

        self.noisekeys = noisekeys
        self.backgroundkeys = backgroundkeys
        self.foregroundkeys = foregroundkeys

        self.setup_indeces()

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
        '''
        TODO 
        -) check vectorization
        -) complete custom CUDA Kernel for spline interpolation
        -) check factors in front
        -) add response for individual sources

        #* The order that `args` has to follow is [(templates), (noise), (backgrounds), (foregrounds)]

        
        '''

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
                logl_args = [ntilde]
                
            else:
                ntilde = self.data.dtilde[self.xp.newaxis, :, :]
                ntildentilde = self.data.dtildedtilde
                logl_args = []

            #breakpoint()
            if self.use_gpu:
                mempool = xp.get_default_memory_pool()
                mempool.free_all_blocks()
            #breakpoint()
            logl = self.compute_logl(psd, ntilde, ntildentilde).real
            # logl = - self.xp.sum( self.xp.sum(ntildentilde / cov, axis = -1) + self.nu * xp.sum(self.xp.log(cov), axis = -1) , axis = -1)
            logl_all.append(logl)

        logl_out = np.concatenate(logl_all)
        logl_out[~np.isfinite(logl_out)] = -self.inf

        if not self.return_gpu:
            #TODO write this in a more elegant way
            try:
                return logl_out.get()
            except:
                return logl_out
        else:
            return logl_out
        
    def setup_indeces(self):
        self.idx_wf = 0
        self.idx_noise = self.nsource_wf_gen
        self.idx_background = self.idx_noise + len(self.noisekeys)
        self.idx_foreground = self.idx_background + len(self.backgroundkeys)
        self.indeces = [self.idx_wf, self.idx_noise, self.idx_background, self.idx_foreground]
        
    def unpack_args(self, args):
        """
        Unpacks the arguments into separate components.

        Args:
            args (list): The list of arguments to be unpacked.

        Returns:
            tuple: A tuple containing the unpacked components: wf_args, noise_args, background_args, foreground_args.
        """
        wf_args, noise_args, background_args, foreground_args = [], [], [], []
        components = [wf_args, noise_args, background_args, foreground_args]
        indeces = self.indeces + [len(args)]
        #breakpoint()
        
        for i in range(len(components)):
            if self.tc_container[i] is not None:
                components[i] += [self.tc_container[i][j].transform_base_parameters(arg) for j,arg in enumerate(args[indeces[i] : indeces[i+1]])]
            else:
                components[i] += args[indeces[i] : indeces[i+1]]

        return wf_args, noise_args, background_args, foreground_args
    
    def unpack_groups(self, groups):
        wf_groups, noise_groups, background_groups, foreground_groups = [], [], [], []
        components = [wf_groups, noise_groups, background_groups, foreground_groups]
        indeces = self.indeces + [len(groups)]
        #breakpoint()
        
        for i in range(len(components)):
            components[i] += groups[indeces[i] : indeces[i+1]]

        return wf_groups, noise_groups, background_groups, foreground_groups


    @partial(jax.jit, static_argnums=(0,))
    def whittle_logl(self, psd, ntilde, ntildentilde):
        """
        Compute the log likelihood for the Whittle likelihood.

        Args:
            psd (array): The power spectral density.
            ntilde (array): The residual data in the frequency domain.
            ntildentilde (array): The residual data in the frequency domain times its complex conjugate traspose.

        Returns:
            array: The log likelihood.
        """

        if self.fullmatrix:
            to_solve, logdet = get_matrix_determinant(psd, hermitian=self.hermitian, return_mat=False)

            ntilde_rep = jnp.repeat(ntilde, psd.shape[0], axis=0)
            ntildeconj_invcov = self.solve(to_solve, jnp.conj(ntilde_rep)[:, :, :])
        
            ntildentilde = jnp.einsum('ijk,ijk -> ij', ntildeconj_invcov, ntilde)
            logl = - jnp.sum(ntildentilde + logdet, axis=-1)

        else:
            cov = psd
            logl = - jnp.sum( self.data.weights * (ntildentilde / cov + jnp.log(cov)),  axis = (1, 2))

        return logl
    

    @partial(jax.jit, static_argnums=(0,))
    def wishart_logl(self, psd, *args, **kwargs) :
        """
        Compute the log likelihood for the Wishart likelihood. 
        Since the Wishart likelihood is based on averaging over data segments, it cannot be used when including deterministic signals.

        Args:
            psd (array): The power spectral density.
        
        Returns:
            array: The log likelihood.
        """

        if self.fullmatrix:
            cov, logdet = get_matrix_determinant(psd, hermitian=self.hermitian, return_mat=True)
            invcov = jnp.linalg.inv(cov)
            del cov

            return -  jnp.sum(jnp.einsum('...ii', jnp.einsum('...ij, ...jk->...ik', invcov, self.data.Y)) + self.data.nu * logdet, axis=-1) 

        else:
            cov = psd
            return - jnp.sum(self.data.Y / cov + self.data.nu[None, :, None] * jnp.log(cov), axis=(1,2)) #+ self.norm

