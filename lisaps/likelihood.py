# -*- coding: utf-8 -*-
from pysco import performance
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

class Likelihood:

    def __init__(self, 
                    psd_fn, 
                    t=None,
                    d=None,
                    freqs=None,
                    dtilde=None,
                    fmin=1e-4,
                    fmax=2.9e-2,
                    source_wf_gen=None,
                    nchannels=3,
                    average=False,
                    correlated=False,
                    fullmatrix=False,
                    Nbins=1000,
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
            nchannels=nchannels,
            fmin=fmin,
            fmax=fmax,
            average=average,
            Nbins=Nbins,
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

        if source_wf_gen is None: # fit only for the psd (noise, stochastic components)
            
            self.nsource_wf_gen = 0

            self.correlated = correlated
            self.fullmatrix = fullmatrix

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
        # if self.rj:
        #     self.nsubset = 1
        # else:     
        self.nsubset = nsubset

        @property
        def tc_container(self):
            return self._tc_container
        
        @tc_container.setter
        def tc_container(self, tc_container=[None, None, None, None]):
            self._tc_container = tc_container

    def __call__(self, args, groups=None, tc_container=None,**kwargs):
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
                logl_args = []

            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()

            logl = self.compute_logl(psd, *logl_args)
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
    def whittle_logl(self, psd, *args, **kwargs):
        
        if self.fullmatrix:
            ntilde = args[0]
            cov = self.get_covariance(psd)
            ntilde_rep = self.xp.repeat(ntilde, cov.shape[0], axis=0)
            ntildeconj_invcov = self.xp.linalg.solve(cov, self.xp.conj(ntilde_rep)[:, :, :])
            detcov = self.xp.linalg.det(cov)
            del cov
            ntildentilde = self.xp.einsum('ijk,ijk -> ij', ntildeconj_invcov, ntilde)
            logl = - self.xp.sum(ntildentilde + self.xp.log(detcov), axis=-1)

        else:
            cov = psd
            #ntildentilde = self.get_XtildeXtilde()
            #logl = - self.xp.sum( self.data.dtildedtilde / cov + self.xp.log(cov),  axis = (1, 2))
            logl = - jnp.sum( self.data.dtildedtilde / cov + jnp.log(cov),  axis = (1, 2))

        return logl
    
    @partial(jax.jit, static_argnums=(0,))
    def wishart_logl(self, psd, *args, **kwargs) :

        if self.fullmatrix:
            cov = self.get_covariance(psd)
            invcov = self.xp.linalg.inv(cov)
            detcov = self.xp.linalg.det(cov)
            del cov

            return -  self.xp.sum(self.xp.einsum('...ii', self.xp.einsum('...ij, ...jk->...ik', invcov, self.data.Y)) + self.data.nu * self.xp.log(detcov), axis=-1)

        else:
            cov = psd
            #return - self.xp.sum(self.data.Y / cov + self.data.nu[None, :, None] * self.xp.log(cov), axis=(1,2)) #+ self.norm
            return - jnp.sum(self.data.Y / cov + self.data.nu[None, :, None] * jnp.log(cov), axis=(1,2)) #+ self.norm

    
    def get_covariance(self, psd):
        #TODO need a way to write this in a JAX compatible way

        nin, nfreqs = psd.shape[0], psd.shape[1]
        covariance = self.xp.zeros(shape=(nin, nfreqs, self.nchannels, self.nchannels))
        for i in range(self.nchannels):
            covariance[:,:,i,i] = psd[:,:,i]

        if self.correlated:
            covariance[:,:,0,1] = psd[:,:,3]  
            covariance[:,:,0,2] = psd[:,:,4]  
            covariance[:,:,1,2] = psd[:,:,5]

            covariance[:,:,1,0] = self.xp.conj(psd[:,:,3])
            covariance[:,:,2,0] = self.xp.conj(psd[:,:,4])
            covariance[:,:,2,1] = self.xp.conj(psd[:,:,5])

        return covariance