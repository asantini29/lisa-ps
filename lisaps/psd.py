# -*- coding: utf-8 -*-

from .baseclasses import BaseNoise, SciRDv1, TDIresponse
from .stochasticbackgrounds import StochasticContribution
from typing import Any, Callable
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from pysco import utils

jax.config.update("jax_enable_x64", True)

import warnings

class Psd(SciRDv1, BaseNoise, StochasticContribution):
    """
    Psd class represents the power spectral density (PSD) model for noise and stochastic backgrounds.

    Args:
        asdTM (float): Amplitude spectral density (ASD) for TM channel. Default is 2.4e-15.
        asdOMS (float): ASD for OMS channel. Default is 7.9e-12.
        fmin (float): Minimum frequency. Default is 1e-4.
        fmax (float): Maximum frequency. Default is 2.5e-2.
        fs (float): Sampling frequency. Default is None.
        equal_arms (bool): Flag indicating whether the arms have equal lengths. Default is False.
        Ncov (int): Number of covariance matrices. Default is None.
        channels (list): List of channel names. Default is None.
        use_gpu (bool): Flag indicating whether to use GPU. Default is False.
        units (str): Frequency units. Default is 'hertz'.
        noiseless (bool): Flag indicating whether to consider noiseless case. Default is False.
        splineperturbation (dict): Dictionary containing spline perturbation information. Default is {}.
        interpkwargs (dict): Additional keyword arguments for interpolation. Default is None.
        fitASDs (bool): Flag indicating whether to fit for ASDs. Default is False.
        backgrounds (list): List of background models. Default is [].
        background_kwargs (dict): Additional keyword arguments for background models. Default is {}.
        foregrounds (list): List of foreground models. Default is [].
        foreground_kwargs (dict): Additional keyword arguments for foreground models. Default is {}.
        isotropicresponse (None): Isotropic response. Default is None.
        GBresponse (None): GB response. Default is None.
        ftol (float): Tolerance for spline interpolation. Default is 0.1.
        **kwargs: Additional keyword arguments.

    Attributes:
        PSDS_design (None): Design PSDs.
        logfmin (float): Logarithm of the minimum frequency.
        logfmax (float): Logarithm of the maximum frequency.
        noiseperturbation (dict): Dictionary containing noise perturbation information.
        backgroundperturbation (dict): Dictionary containing background perturbation information.
        fitASDs (bool): Flag indicating whether to fit for ASDs.
        ftol (float): Tolerance for spline interpolation.

    Methods:
        set_noisefn: Sets the noise function based on the perturbation type.
        constmod: Applies constant modification to the input PSDs.
        splinemod: Applies spline modification to the input PSDs.
        logperturbation: Calculates the spline perturbation in logarithmic space.
        logperturbation_numba: Calculates the spline perturbation using a numba cuda kernel.
        prepare_interp_input_numba: Prepares the input for spline interpolation using a numba cuda kernel.
    """
    def __init__(self, 
                 asdTM=2.4e-15, 
                 asdOMS=7.9e-12, 
                 fmin=1e-4, 
                 fmax=2.9e-2, 
                 fs=None,
                 equal_arms=False,
                 custom_armlength=None,
                 Ncov=None, 
                 channels=None, 
                 use_gpu=False, 
                 units='hertz', 
                 noiseless=False, 
                 splineperturbation={}, 
                 interpkwargs=None, 
                 fitASDs=False, 
                 backgrounds=[],
                 background_kwargs={},
                 foregrounds=[],
                 foreground_kwargs={},
                 isotropicresponse=None, 
                 GBresponse=None,
                 correct_sagnac=True,
                 ftol=0.1,
                 scirdv1=False,
                 **kwargs
                 ):
        
        if noiseless:
            asdTM = 0.
            asdOMS = 0.

        if scirdv1:
            SciRDv1.__init__(self, fs=fs, Ncov=Ncov, channels=channels, use_gpu=use_gpu, units=units, interpkwargs=interpkwargs)
        else:
            BaseNoise.__init__(self, asdTM=asdTM, asdOMS=asdOMS, equal_arms=equal_arms, custom_armlength=custom_armlength, fs=fs, Ncov=Ncov, channels=channels, use_gpu=use_gpu, units=units, interpkwargs=interpkwargs)

        if not isinstance(backgrounds, list):
            backgrounds = [backgrounds]
        if not isinstance(foregrounds, list):
            foregrounds = [foregrounds]

        StochasticContribution.__init__(self, 
                                       backgrounds=backgrounds, 
                                       background_kwargs=background_kwargs, 
                                       foregrounds=foregrounds, 
                                       foreground_kwargs=foreground_kwargs, 
                                       TDIsetup=self.TDIsetup,
                                       equal_arms=equal_arms,
                                       isotropicresponse=isotropicresponse, 
                                       GBresponse=GBresponse, 
                                       channels=self.channels, 
                                       units=units,
                                       use_gpu=use_gpu,
                                       correct_sagnac=correct_sagnac
                                       )

        self.PSDS_design = None

        self.fmin = fmin
        self.fmax = fmax

        self.logfmin = self.xp.log10(self.fmin)
        self.logfmax = self.xp.log10(self.fmax)

        # freqs = freqs

        self.fitASDs = fitASDs
        self.update_perturbation(splineperturbation)

        if (self.fitASDs) and (self.noiseperturbation):
            warnings.warn('Fitting both for the noise ASDs and perturbation. This will affect convergence')

        self.ftol = ftol

    @property
    def noiseperturbation(self):
        return self._noiseperturbation
    
    @noiseperturbation.setter
    def noiseperturbation(self, value):
        self._noiseperturbation = value

    @property
    def backgroundperturbation(self):
        return self._backgroundperturbation
    
    @backgroundperturbation.setter
    def backgroundperturbation(self, value):
        self._backgroundperturbation = value

    def update_perturbation(self, splineperturbation):
        '''
        Update the spline perturbation.
        '''
        assert isinstance(splineperturbation, dict), '`splineperturbation` must be a dictionary'
        self.noiseperturbation = splineperturbation['noise']
        self.backgroundperturbation = splineperturbation['background']
        self.set_noisefn()
        print('Using ' + self.noisefn.__name__)

    def set_noisefn(self):

        if self.noiseperturbation:
            self.noisefn = self.splinemod

        else:
            if self.fitASDs:
                self.noisefn = self.constmod
            else:
                self.noisefn = self.set_PSDS                 


    def constmod(self, freqs,  args, **kwargs):

        freqs = jnp.asarray(freqs)
        args = args[0]
        args = jnp.atleast_2d(args)

        asdTM, asdOMS = jnp.asarray(args[:, 0:1]), jnp.asarray(args[:, 1:2])

        if (args.shape[1] == 2):
            asdTM, asdOMS = jnp.asarray(args[:, 0:1]), jnp.asarray(args[:, 1:2])

            PSDS = self.get_PSDS(asdTM, asdOMS, freqs, squeeze=False)

        else:
            asdTM, asdOMS = jnp.asarray(args[:, 0::2]).reshape(-1, 2), jnp.asarray(args[:, 1::2]).reshape(-1, 2)
            PSDS = self.get_PSDS(asdTM, asdOMS, freqs, squeeze=False).reshape(-1, len(freqs), self.Ncov)

        return PSDS
    
    def splinemod(self, freqs, args, groups, knots=None, **kwargs):
        '''
        Applies spline modification to the input power spectral densities (PSDs).

        Parameters:
        - freqs (array-like): Frequencies at which the PSDs are evaluated.
        - args (array-like or list): Arguments of the PSDs evaluation. if `self.fitASDs` is True the first argument is the ASDs, while the rest are the spline coefficients. 
        - groups (array-like or list): Group indices for the ASDs and the spline coefficients.
        - knots (array-like, optional): Knots for the spline interpolation. If not provided, use reversible jump.
        - **kwargs: Additional keyword arguments.

        Returns:
        - PSDS (ndarray): Modified PSDs with shape (n_in, len(freqs), Ncov).
        '''
        
        if self.fitASDs:
            PSDS = self.constmod(freqs, args[:1])    
            args = args[1:]  
            groups = groups[1:]

        else:
            PSDS = self.set_PSDS(freqs, out=True)   

        if not isinstance(args, list):
            args = [args]
        
        if not isinstance(groups, list):
            groups = [groups]
            
        nin = min([arg.shape[0] for arg in args])
        #PSDS = self.xp.empty((nin, len(freqs), self.Ncov))

        knots, weights = self.prepare_interp_input_numba(args=args, groups=groups)
        # breakpoint()
        # ftol_mask = self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol, axis=-1)
        # ftol_mask = self.xp.broadcast_to(ftol_mask, (freqs.shape[0], self.Ncov, nin)).transpose(2, 0, 1)

        ftol_mask = self.xp.any(self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol, axis=-1), axis=0)
        
        logperturbation = self.logperturbation_numba(freqs=freqs, knots=knots, weights=weights)
        perturbation = 10**logperturbation
        perturbation[ftol_mask] = self.xp.nan

        perturbation = jnp.asarray(perturbation)        

        PSDS = PSDS * perturbation
         

        return PSDS
    
    def logperturbation(self, freqs, knots, weights):
        '''
        Spline perturbation.
        '''
        interp = self.interp(knots, weights, **self.interpkwargs)
        logperturbation = (interp(self.xp.log10(freqs))).reshape(-1, len(freqs), weights.shape[0]) #put it in the same shape of PSDS
        
        return logperturbation
    
    def logperturbation_numba(self, freqs, knots, weights):
        '''
        Spline perturbation with numba cuda kernel.
        '''
        logperturbation = self.interp(self.xp.log10(freqs), knots, weights)

        return logperturbation.transpose(1, 2, 0)
    
    def prepare_interp_input_numba(self, args, groups):
        '''
        here args is a list of the fashion [ (edges), (knots_i) ]
        I want the output to be (knots position, knots weights)
        '''

        if (args[1].shape[-1] == self.Ncov + 1) or (len(args) == 2): # this means all the weights are together or there is only one spline
            edges_weights, knots_full = self.xp.asarray(args[0]), self.xp.asarray(args[1]) #always work along the `1` axis for frequency operations # TODO may have to change this, maybe (Ncov, Nin, Nfreq) is better
            groups_knots = groups[1]           
    
            leftedge_full = edges_weights[:, 0::2]
            rightedge_full = edges_weights[:, 1::2]

            group_unique, group_index, group_inverse, group_count = self.xp.unique(groups_knots, return_index=True, return_counts=True, return_inverse=True)

            diff_temp = self.xp.ones_like(group_inverse)
            diff_temp[1:] = (~self.xp.diff(group_inverse).astype(bool)).astype(int)

            inds_per_group = (self.xp.cumsum(diff_temp) - 1)
            inds_group_subtract = inds_per_group[group_index][group_inverse]
            inds_per_group = inds_per_group - inds_group_subtract

            ngroups = (group_unique.max().item() - group_unique.min().item() ) + 1
            maxgroups = group_count.max().item()

            knots_full_nans = self.xp.full((ngroups, maxgroups, knots_full.shape[-1]), self.xp.nan)
            #breakpoint()
            leftedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmin), leftedge_full), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmax), rightedge_full), axis=1)[:, None, :]

            knots_full_nans[(groups_knots, inds_per_group)] = knots_full
            # for i, g in enumerate(groups_unique):
            #     #breakpoint()
            #     knots_full_nans[i, :groups_count[i]] = knots_full[groups_knots == g]

            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)

            positions = knots_full_nans[:,:,:1]
            weights = knots_full_nans[:,:,1:]

            ii = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, ii, axis=1)

            if (args[1].shape[-1] == self.Ncov + 1):
                sortedpositions = self.xp.repeat(sortedpositions, self.Ncov, axis = -1).transpose(2,0,1)
            else:
                sortedpositions = sortedpositions.transpose(2,0,1)

            sortedweights = self.xp.take_along_axis(weights, ii, axis=1).transpose(2,0,1)

        else:
            edges_weights = self.xp.asarray(args[0])
            leftedge_full = edges_weights[:, 0::2]
            rightedge_full = edges_weights[:, 1::2]

            args_knots = [self.xp.asarray(arg) for arg in args[1:]] #still a list
            groups_knots = groups[1:] #still a list

            groups_unique = np.unique(groups[0])
            ngroups = (groups_unique.max().item() - groups_unique.min().item()) + 1

            try:
                maxgroups = self.Nknotsmax  
            except:
                maxgroups = 0
                for g in groups_knots:
                    _, counts = np.unique(g, return_counts=True)
                    maxgroups = counts.max().item() if counts.max().item() > maxgroups else maxgroups
                self.Nknotsmax = maxgroups
            
            knots_full_nans = self.xp.full((ngroups, maxgroups, 2*self.Ncov), self.xp.nan)
            leftedge_full = self.xp.concatenate((self.xp.full((ngroups, self.Ncov), self.logfmin), leftedge_full), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups, self.Ncov), self.logfmax), rightedge_full), axis=1)[:, None, :]

            for j, (arg, group) in enumerate(zip(args_knots, groups_knots)):
                group = self.xp.asarray(group)
                group_unique, group_index, group_inverse, group_count = self.xp.unique(group, return_index=True, return_counts=True, return_inverse=True)

                diff_temp = self.xp.ones_like(group_inverse)
                diff_temp[1:] = (~self.xp.diff(group_inverse).astype(bool)).astype(int)

                inds_per_group = (self.xp.cumsum(diff_temp) - 1)
                inds_group_subtract = inds_per_group[group_index][group_inverse]
                inds_per_group = inds_per_group - inds_group_subtract

                knots_full_nans[:,:, j][(group, inds_per_group)] = arg[:, 0]
                knots_full_nans[:,:, self.Ncov + j][(group, inds_per_group)] = arg[:, 1]

            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)

            positions = knots_full_nans[:,:,:self.Ncov]
            weights = knots_full_nans[:,:,self.Ncov:]

            ii = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, ii, axis=1).transpose(2,0,1)
            sortedweights = self.xp.take_along_axis(weights, ii, axis=1).transpose(2,0,1)

        return sortedpositions, sortedweights
                

    def __call__(self, freqs, noiseargs=[], backargs=[], foreargs=[], noisegroups=[], backgroups=[], foregroups=[], **kwargs):
        '''
        Compute the total PSD in each channel.

        Parameters:
        - freqs (array-like): Frequencies at which to compute the PSD.
        - noiseargs (list, optional): Arguments for the noise function. Default is an empty list.
        - backargs (list, optional): Arguments for the background function. Default is an empty list.
        - foreargs (list, optional): Arguments for the foreground function. Default is an empty list.
        - noisegroups (list, optional): Groups for the noise function. Default is an empty list.
        - backgroups (list, optional): Groups for the background function. Default is an empty list.
        - foregroups (list, optional): Groups for the foreground function. Default is an empty list.
        - **kwargs (dict): Additional keyword arguments.

        Returns:
        - PSDS (array-like): The total PSD in each channel.

        '''
        PSDS = self.noisefn(freqs=freqs, args=noiseargs, groups=noisegroups, **kwargs['noise'])

        #breakpoint()  
        
        if self.nbackgrounds > 0:
            
            try:
                response = self.isotropicresponse[jnp.newaxis, :, :]
            except:
                self.set_isotropicresponse(freqs)
                response = self.isotropicresponse[jnp.newaxis, :, :]

            sgwbs_all = jnp.zeros_like(PSDS)

            for i in range(self.nbackgrounds):
                
                if len(backargs[i]) > 0:
                    back = self.backgrounds[i]
                    h2omega = self.backgrounds_fn[i](freqs, backargs[i], **kwargs[back])[:,:,None]         

                    if self.backgroundperturbation:

                        bknots, bweights = self.prepare_interp_input_numba(backargs[self.nbackgrounds+2*i:self.nbackgrounds+2*(i+1)], backgroups[self.nbackgrounds+2*i:self.nbackgrounds+2*(i+1)])

                        ftol_mask = self.xp.any(self.xp.any(self.xp.abs(self.xp.diff(bknots)) < self.ftol, axis=-1), axis=0)
                        
                        logperturbation = self.logperturbation_numba(freqs=freqs, knots=bknots, weights=bweights)
                       
                        perturbation = 10**logperturbation
                        perturbation[ftol_mask] = self.xp.nan
                        perturbation = jnp.asarray(perturbation)        

                        h2omega = h2omega * perturbation

                    #h2omega = jnp.repeat(h2omega, self.Ncov, axis=-1)
                    #breakpoint()
                    Shs = self.convert_to_psd(freqs, h2omega)
                    #Sh = [self.convert_to_psd(freqs, h2omega) for i in range(self.Ncov)]
                    #Shs = self.xp.array(Sh).transpose(1, 2, 0)
                
                sgwbs_all = sgwbs_all + Shs   

            PSDS = PSDS + sgwbs_all * response 
        
        if self.nforegrounds > 0:

            try:
                response = self.GBresponse[jnp.newaxis, :, :]
            except:
                self.set_GBresponse(freqs)
                response = self.GBresponse[jnp.newaxis, :, :]

            sgwfs_all = jnp.zeros_like(PSDS)

            for i in range(self.nforegrounds):
                
                if len(foreargs[i]) > 0:
                    fore = self.foregrounds[i]
                    h2omega = self.foregrounds_fn[i](freqs, foreargs[i], **kwargs[fore])[:,:,None] 

                    # Shs = self.xp.array(Sh).transpose(1, 2, 0)
                    # response = self.GBresponse[self.xp.newaxis, :, :]
                    # sgwbs_all += Shs * response 

                    if self.foregroundperturbation:

                        fknots, fweights = self.prepare_interp_input_numba(backargs[self.nforegrounds+2*i:self.nforegrounds+2*(i+1)], foregroups[self.nforegrounds+2*i:self.nforegrounds+2*(i+1)])

                        ftol_mask = self.xp.any(self.xp.any(self.xp.abs(self.xp.diff(fknots)) < self.ftol, axis=-1), axis=0)
                        
                        logperturbation = self.logperturbation_numba(freqs=freqs, knots=fknots, weights=fweights)
                       
                        perturbation = 10**logperturbation
                        perturbation[ftol_mask] = self.xp.nan
                        perturbation = jnp.asarray(perturbation)        

                        h2omega = h2omega * perturbation

                    Shs = self.convert_to_psd(freqs, h2omega)
                    #Sh = [self.convert_to_psd(freqs, h2omega) for i in range(self.Ncov)]
                    #Shs = self.xp.array(Sh).transpose(1, 2, 0)
                    sgwfs_all = sgwfs_all + Shs  

                PSDS = PSDS + sgwfs_all * response 

        return PSDS
