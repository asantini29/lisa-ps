# -*- coding: utf-8 -*-

from .baseclasses import BaseNoise
from .stochasticbackgrounds import StochasticContribution
from typing import Any, Callable
import numpy as np

import time

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

import warnings

#todo: clean and document the gaussian bump functions

class Psd(BaseNoise, StochasticContribution):
    """
    A class to represent the Power Spectral Density (PSD) of a noise model.
    Inherits from:
        BaseNoise: Base class for noise models.
        StochasticContribution: Base class for stochastic contributions.
    Attributes:
        kwargs (dict): Additional keyword arguments.
        PSDS_design (None): Placeholder for the PSD design.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        logfmin (float): Logarithm of the minimum frequency.
        logfmax (float): Logarithm of the maximum frequency.
        fitASDs (bool): Flag to fit Amplitude Spectral Densities (ASDs).
        ftol (float): Tolerance for frequency differences.
    Methods:
        noiseperturbation: Property to get/set noise perturbation.
        backgroundperturbation: Property to get/set background perturbation.
        foregroundperturbation: Property to get/set foreground perturbation.
        update_perturbation(perturbation): Update the spline perturbation.
        set_noisefn(): Set the noise function based on perturbation and fitting flags.
        constmod(freqs, args, **kwargs): Compute constant model PSDs.
        splinemod(freqs, args, groups, knots=None, **kwargs): Apply spline modification to the input PSDs.
        logperturbation(freqs, knots, weights): Compute spline perturbation.
        logperturbation_numba(freqs, knots, weights): Compute spline perturbation with numba cuda kernel.
        prepare_interp_input_numba(args, groups): Prepare input for spline interpolation with numba.
        gaussian_bump(freq, A, f_center, width): Compute Gaussian bump.
        gaussian_bump_sum(freq, A, f_center, width): Compute sum of Gaussian bumps.
        gaussian_bump_rj(freq, args, groups): Compute Gaussian bump with reversible jump.
        log_gaussian_bump(freq, A, f_center, width): Compute log-Gaussian bump.
        log_gaussian_bump_sum(freq, A, f_center, width): Compute sum of log-Gaussian bumps.
        log_gaussian_bump_rj(freq, args, groups, nin): Compute log-Gaussian bump with reversible jump.
        __call__(freqs, noiseargs=[], backargs=[], foreargs=[], noisegroups=[], backgroups=[], foregroups=[], **kwargs): Compute the total PSD in each channel.
    """
    
    def __init__(self, 
                 asdTM=2.4e-15, 
                 asdOMS=7.9e-12, 
                 fmin=1e-4, 
                 fmax=2.9e-2, 
                 T=1.0,
                 fs=None,
                 custom_armlength=None,
                 Ncov=None, 
                 channels=None, 
                 use_gpu=False, 
                 units='hertz', 
                 noiseless=False, 
                 perturbation={'noise':False, 'background':False, 'foreground':False}, 
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
                 filtered=True,
                 **kwargs
                 ):
        """
        Initialize the PSD class.
        Parameters:
        -----------
        asdTM : float, optional
            Amplitude spectral density for test mass noise (default is 2.4e-15).
        asdOMS : float, optional
            Amplitude spectral density for optical metrology system noise (default is 7.9e-12).
        fmin : float, optional
            Minimum frequency (default is 1e-4).
        fmax : float, optional
            Maximum frequency (default is 2.9e-2).
        T : float, optional
            Observation time in years (default is 1.0).
        fs : float or None, optional
            Sampling frequency (default is None).
        equal_arms : bool, optional
            If True, assumes equal arm lengths (default is False).
        custom_armlength : float or None, optional
            Custom arm length (default is None).
        Ncov : int or None, optional
            Number of covariance matrices (default is None).
        channels : list or None, optional
            List of channels (default is None).
        use_gpu : bool, optional
            If True, use GPU for computations (default is False).
        units : str, optional
            Units for frequency ('hertz' or 'radians') (default is 'hertz').
        noiseless : bool, optional
            If True, assumes noiseless data (default is False).
        perturbation : dict, optional
            Dictionary specifying perturbation types (default is {'noise': False, 'background': False, 'foreground': False}).
        interpkwargs : dict or None, optional
            Keyword arguments for interpolation (default is None).
        fitASDs : bool, optional
            If True, fit amplitude spectral densities (default is False).
        backgrounds : list, optional
            List of background noise sources (default is []).
        background_kwargs : dict, optional
            Keyword arguments for background noise sources (default is {}).
        foregrounds : list, optional
            List of foreground noise sources (default is []).
        foreground_kwargs : dict, optional
            Keyword arguments for foreground noise sources (default is {}).
        isotropicresponse : callable or None, optional
            Function for isotropic response (default is None).
        GBresponse : callable or None, optional
            Function for galactic binary response (default is None).
        correct_sagnac : bool, optional
            If True, correct for Sagnac effect (default is True).
        ftol : float, optional
            Tolerance for fitting (default is 0.1).
        scirdv1 : bool, optional
            If True, use SCIRDV1 (default is False).
        filtered : bool, optional
            If True, use filtered testmass and oms noises (default is True).
        **kwargs : dict
            Additional keyword arguments.
        """
        self.kwargs = kwargs.copy()
        if 'perturbation_type' not in kwargs:
            self.kwargs['perturbation_type'] = 'spline'
        else:
            perturbation_type = kwargs.pop('perturbation_type')

        BaseNoise.__init__(self, asdTM=asdTM, asdOMS=asdOMS, custom_armlength=custom_armlength, T=T, fs=fs, Ncov=Ncov, channels=channels, use_gpu=use_gpu, units=units, interpkwargs=interpkwargs, scirdv1=scirdv1,filtered=filtered, **kwargs)

        if noiseless:
            self.asdTM = 0.
            self.asdOMS = 0.

        if not isinstance(backgrounds, list):
            backgrounds = [backgrounds]
        if not isinstance(foregrounds, list):
            foregrounds = [foregrounds]

        StochasticContribution.__init__(self, 
                                       backgrounds=backgrounds, 
                                       background_kwargs=background_kwargs, 
                                       foregrounds=foregrounds, 
                                       foreground_kwargs=foreground_kwargs, 
                                       custom_armlength=custom_armlength,
                                       TDIsetup=self.TDIsetup,
                                       isotropicresponse=isotropicresponse, 
                                       GBresponse=GBresponse, 
                                       channels=self.channels, 
                                       units=units,
                                       use_gpu=use_gpu,
                                       interpkwargs=interpkwargs,
                                       correct_sagnac=correct_sagnac,
                                       )

        self.PSDS_design = None

        self.fmin = fmin
        self.fmax = fmax

        self.logfmin = self.xp.log10(self.fmin)
        self.logfmax = self.xp.log10(self.fmax)

        self.fitASDs = fitASDs
        self.update_perturbation(perturbation)

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

    @property
    def foregroundperturbation(self):
        return self._foregroundperturbation
    
    @foregroundperturbation.setter
    def foregroundperturbation(self, value):
        self._foregroundperturbation = value

    def update_perturbation(self, perturbation):
        """
        Update the psd perturbation.

        Parameters:
        - perturbation (dict): Dictionary specifying the perturbation types.

        #todo document the extra keywords
        """

        assert isinstance(perturbation, dict), '`perturbation` must be a dictionary'
        self.noiseperturbation = perturbation['noise']
        self.backgroundperturbation = perturbation['background']
        self.foregroundperturbation = perturbation['foreground']

        if 'Nknotsmax' in perturbation.keys():
            self.Nknotsmax = perturbation['Nknotsmax']

        if 'fill_edges' in perturbation.keys() and perturbation['fill_edges'] is not None:
            if isinstance(perturbation['fill_edges'], (list, tuple, np.ndarray)):
                self.leftedge = perturbation['fill_edges'][0]
                self.rightedge = perturbation['fill_edges'][1]
            else:
                self.leftedge = perturbation['fill_edges']
                self.rightedge = perturbation['fill_edges']

                self.prepare_interp_input = self.prepare_interp_input_fixed_edges
        
        else:
            self.prepare_interp_input = self.prepare_interp_input_with_edges

        self.set_noisefn()

    def set_noisefn(self):
        """
        Sets the noise function for the instance based on the current configuration.
        This method determines which noise function to use based on the attributes
        `noiseperturbation` and `fitASDs`. The possible noise functions that can be
        set are `splinemod`, `constmod`, or `set_PSDS`.
        - If `noiseperturbation` is True, the noise function is set to `splinemod`.
        - If `noiseperturbation` is False and `fitASDs` is True, the noise function
          is set to `constmod`.
        - If both `noiseperturbation` and `fitASDs` are False, the noise function
          is set to `set_PSDS`.
        """

        if self.noiseperturbation:
            self.noisefn = self.splinemod

        else:
            if self.fitASDs:
                self.noisefn = self.constmod
            else:
                self.noisefn = self.set_PSDS                 


    def constmod(self, freqs,  args, **kwargs):
        """
        Compute the Power Spectral Density (PSD) for given frequencies and arguments.
        Parameters:
        -----------
        freqs : array-like
            Array of frequency values.
        args : array-like
            Array of arguments where the first dimension corresponds to different sets of parameters.
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        PSDS : jnp.ndarray
            Computed PSD values. The shape of the output depends on the shape of the input arguments.
        """

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
        """
        Applies spline modification to the input power spectral densities (PSDs).

        Parameters:
        - freqs (array-like): Frequencies at which the PSDs are evaluated.
        - args (array-like or list): Arguments of the PSDs evaluation. if `self.fitASDs` is True the first argument is the ASDs, while the rest are the spline coefficients. 
        - groups (array-like or list): Group indices for the ASDs and the spline coefficients.
        - knots (array-like, optional): Knots for the spline interpolation. If not provided, use reversible jump.
        - **kwargs: Additional keyword arguments.

        Returns:
        - PSDS (ndarray): Modified PSDs with shape (n_in, len(freqs), Ncov).
        """
        
        if self.fitASDs:
            PSDS = self.constmod(freqs, args[:1])    
            ngroups = groups[0].shape[0]
            args = args[1:]  
            groups = groups[1:]
        else:
            PSDS = self.set_PSDS(freqs, out=True)   
            ngroups = 0

        # if self.xp.any(self.xp.isnan(PSDS)):
        #     warnings.warn('Some noise PSDs are NaN')
        #     breakpoint()

        if not isinstance(args, list):
            args = [args]
        
        if not isinstance(groups, list):
            groups = [groups]
            
        nin = min([arg.shape[0] for arg in args])

        knots, weights = self.prepare_interp_input(args=args, groups=groups, ngroups=ngroups)

        ftol_mask = self.xp.any(self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol, axis=-1), axis=0)
        
        logperturbation = self.logperturbation_numba(freqs=freqs, knots=knots, weights=weights)
        perturbation = 10**logperturbation
        perturbation[ftol_mask] = self.xp.nan 
        
        perturbation = jnp.asarray(perturbation)        

        PSDS = PSDS * perturbation
         
        return PSDS
    
    def logperturbation(self, freqs, knots, weights):
        """
        Apply a log perturbation to the given frequencies using the specified knots and weights.
        Parameters:
        -----------
        freqs : array-like
            The frequencies at which to apply the log perturbation.
        knots : array-like
            The knot points for the interpolation.
        weights : array-like
            The weights for the interpolation.
        Returns:
        --------
        logperturbation : ndarray
            The log perturbation applied to the frequencies, reshaped to match the shape of the PSDs.
        """
        
        interp = self.interp(knots, weights, **self.interpkwargs)
        logperturbation = (interp(self.xp.log10(freqs))).reshape(-1, len(freqs), weights.shape[0]) #put it in the same shape of PSDS
        
        return logperturbation
    
    def logperturbation_numba(self, freqs, knots, weights):
        """
        Compute the log perturbation using numba (or numba CUDA if possible) for performance optimization.
        Parameters:
        freqs (array-like): Array of frequency values.
        knots (array-like): Array of knot points for interpolation.
        weights (array-like): Array of weights for interpolation.
        Returns:
        numpy.ndarray: Transposed log perturbation array with shape (n_knots, n_weights, n_freqs).
        """
        
        logperturbation = self.interp(self.xp.log10(freqs), knots, weights)

        return logperturbation.transpose(1, 2, 0)
    
    def prepare_interp_input_with_edges(self, args, groups, ngroups=0):
        """
        Prepares the input data for interpolation using Numba.
        Parameters:
        -----------
        args : list
            A list of arrays where the first element contains the edges and weights, 
            and the subsequent elements contain the knots. The shape of the arrays 
            determines the processing path.
        groups : list
            A list of arrays where the first element contains the group indices for 
            the edges and weights, and the subsequent elements contain the group 
            indices for the knots.
        ngroups : int, optional
            Number of groups (default is 0).
        Returns:
        --------
        sortedpositions : ndarray
            An array of sorted positions for interpolation.
        sortedweights : ndarray
            An array of sorted weights corresponding to the sorted positions.
        """

        if (args[1].shape[-1] == self.Ncov + 1) or (len(args) == 2): # this means all the weights are together or there is only one spline
            edges_weights, knots_full = self.xp.asarray(args[0]), self.xp.asarray(args[1]) #always work along the `1` axis for frequency operations # TODO may have to change this, maybe (Ncov, Nin, Nfreq) is better
            groups_knots = groups[1]           
    
            leftedge_full = edges_weights[:, 0::2]
            rightedge_full = edges_weights[:, 1::2]

            #? group_unique, group_index, group_inverse, group_count = self.xp.unique(groups_knots, return_index=True, return_counts=True, return_inverse=True)
            group_unique, group_index, group_inverse = self.xp.unique(groups_knots, return_index=True, return_counts=False, return_inverse=True)

            diff_temp = self.xp.ones_like(group_inverse)
            diff_temp[1:] = (~self.xp.diff(group_inverse).astype(bool)).astype(int)

            inds_per_group = (self.xp.cumsum(diff_temp) - 1)
            inds_group_subtract = inds_per_group[group_index][group_inverse]
            inds_per_group = inds_per_group - inds_group_subtract

            ngroups = leftedge_full.shape[0] #? (group_unique.max().item() - group_unique.min().item() ) + 1
            maxgroups = self.Nknotsmax #? group_count.max().item() if group_count.shape[0] > 0 else 0

            knots_full_nans = self.xp.full((ngroups, maxgroups, knots_full.shape[-1]), self.xp.nan)
            if leftedge_full.shape[0] != ngroups:
                breakpoint()

            leftedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmin), leftedge_full), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmax), rightedge_full), axis=1)[:, None, :]

            knots_full_nans[(groups_knots, inds_per_group)] = knots_full
        
            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)

            positions = knots_full_nans[:,:,:1]
            weights = knots_full_nans[:,:,1:]

            order = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, order, axis=1)

            if (args[1].shape[-1] == self.Ncov + 1):
                sortedpositions = self.xp.repeat(sortedpositions, self.Ncov, axis = -1).transpose(2,0,1)
            else:
                sortedpositions = sortedpositions.transpose(2,0,1)

            sortedweights = self.xp.take_along_axis(weights, order, axis=1).transpose(2,0,1)

        else:
            edges_weights = self.xp.asarray(args[0])
            leftedge_full = edges_weights[:, 0::2]
            rightedge_full = edges_weights[:, 1::2]

            args_knots = [self.xp.asarray(arg) for arg in args[1:]] #still a list
            groups_knots = groups[1:] #still a list

            # groups_unique = np.unique(groups[0])
            # ngroups = (groups_unique.max().item() - groups_unique.min().item()) + 1

            ngroups = groups[0].shape[0] # RJ is not used in this branch, since it is the one handling the edges. there will be only one leaf per group

            maxgroups = self.Nknotsmax  
            
            knots_full_nans = self.xp.full((ngroups, maxgroups, 2*self.Ncov), self.xp.nan)
            leftedge_full = self.xp.concatenate((self.xp.full((ngroups, self.Ncov), self.logfmin), leftedge_full), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups, self.Ncov), self.logfmax), rightedge_full), axis=1)[:, None, :]

            for j, (arg, group) in enumerate(zip(args_knots, groups_knots)):
                group = self.xp.asarray(group)
                #? group_unique, group_index, group_inverse, group_count = self.xp.unique(group, return_index=True, return_counts=True, return_inverse=True)
                group_unique, group_index, group_inverse = self.xp.unique(group, return_index=True, return_counts=False, return_inverse=True)

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

            order = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, order, axis=1).transpose(2,0,1)
            sortedweights = self.xp.take_along_axis(weights, order, axis=1).transpose(2,0,1)

        return sortedpositions, sortedweights
    
    def prepare_interp_input_fixed_edges(self, args, groups, ngroups=0):
        """
        Prepares the input data for interpolation using Numba.
        Parameters:
        -----------
        args : list
            A list of arrays where the first element contains the edges and weights, 
            and the subsequent elements contain the knots. The shape of the arrays 
            determines the processing path.
        groups : list
            A list of arrays where the first element contains the group indices for 
            the edges and weights, and the subsequent elements contain the group 
            indices for the knots.
        ngroups : int, optional
            Number of groups (default is 0).
        Returns:
        --------
        sortedpositions : ndarray
            An array of sorted positions for interpolation.
        sortedweights : ndarray
            An array of sorted weights corresponding to the sorted positions.
        """

        if len(args) == 1: # this means all the weights are together or there is only one spline
            knots_full = self.xp.asarray(args[0]) #always work along the `1` axis for frequency operations # TODO may have to change this, maybe (Ncov, Nin, Nfreq) is better
            groups_knots = groups[0]           
    
            #leftedge_full = edges_weights[:, 0::2]
            #rightedge_full = edges_weights[:, 1::2]

            #? group_unique, group_index, group_inverse, group_count = self.xp.unique(groups_knots, return_index=True, return_counts=True, return_inverse=True)
            group_unique, group_index, group_inverse = self.xp.unique(groups_knots, return_index=True, return_counts=False, return_inverse=True)

            diff_temp = self.xp.ones_like(group_inverse)
            diff_temp[1:] = (~self.xp.diff(group_inverse).astype(bool)).astype(int)

            inds_per_group = (self.xp.cumsum(diff_temp) - 1)
            inds_group_subtract = inds_per_group[group_index][group_inverse]
            inds_per_group = inds_per_group - inds_group_subtract

            if group_unique.shape[0] > 1:
                ngroups = max(ngroups, (group_unique.max().item() - group_unique.min().item() ) + 1)

            maxgroups = self.Nknotsmax #? group_count.max().item() if group_count.shape[0] > 0 else 0

            knots_full_nans = self.xp.full((ngroups, maxgroups, knots_full.shape[-1]), self.xp.nan)

            leftedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmin), self.xp.full((ngroups,1), self.leftedge)), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmax), self.xp.full((ngroups,1), self.rightedge)), axis=1)[:, None, :]

            knots_full_nans[(groups_knots, inds_per_group)] = knots_full
        
            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)

            positions = knots_full_nans[:,:,:1]
            weights = knots_full_nans[:,:,1:]

            order = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, order, axis=1)

            # if (args[1].shape[-1] == self.Ncov + 1):
            #     sortedpositions = self.xp.repeat(sortedpositions, self.Ncov, axis = -1).transpose(2,0,1)
            # else:
            #     sortedpositions = sortedpositions.transpose(2,0,1)
            
            sortedpositions = sortedpositions.transpose(2,0,1)
            sortedweights = self.xp.take_along_axis(weights, order, axis=1).transpose(2,0,1)

        else:
            args_knots = [self.xp.asarray(arg) for arg in args] #still a list
            groups_knots = groups#[1:] #still a list

            groups_unique, groups_index, groups_inverse = [], [], []

            for i in range(len(groups_knots)):
                tmp_unique, tmp_index, tmp_inverse = self.xp.unique(groups_knots[i], return_index=True, return_counts=False, return_inverse=True)

                groups_unique.append(tmp_unique)
                groups_index.append(tmp_index)
                groups_inverse.append(tmp_inverse)

                if tmp_unique.shape[0] > 0:
                    ngroups = max(ngroups, (tmp_unique.max().item() - tmp_unique.min().item()) + 1)            

            maxgroups = self.Nknotsmax  
            
            knots_full_nans = self.xp.full((ngroups, maxgroups, 2*self.Ncov), self.xp.nan)
            leftedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmin), self.xp.full((ngroups,1), self.leftedge)), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmax), self.xp.full((ngroups,1), self.rightedge)), axis=1)[:, None, :]

            for j, (arg, group) in enumerate(zip(args_knots, groups_knots)):
                group = self.xp.asarray(group)
                #? group_unique, group_index, group_inverse, group_count = self.xp.unique(group, return_index=True, return_counts=True, return_inverse=True)
                group_unique, group_index, group_inverse = groups_unique[j], groups_index[j], groups_inverse[j]

                diff_temp = self.xp.ones_like(group_inverse)
                diff_temp[1:] = (~self.xp.diff(group_inverse).astype(bool)).astype(int)

                inds_per_group = (self.xp.cumsum(diff_temp) - 1)
                inds_group_subtract = inds_per_group[group_index][group_inverse]
                inds_per_group = inds_per_group - inds_group_subtract

                knots_full_nans[:,:, j][(group, inds_per_group)] = arg[:, 0]
                knots_full_nans[:,:, self.Ncov + j][(group, inds_per_group)] = arg[:, 1]

            leftedge_full = self.xp.repeat(leftedge_full, self.Ncov, axis = 2)
            rightedge_full = self.xp.repeat(rightedge_full, self.Ncov, axis = 2)

            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)

            positions = knots_full_nans[:,:,:self.Ncov]
            weights = knots_full_nans[:,:,self.Ncov:]

            order = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, order, axis=1).transpose(2,0,1)
            sortedweights = self.xp.take_along_axis(weights, order, axis=1).transpose(2,0,1)

        return sortedpositions, sortedweights
    

    @partial(jax.jit, static_argnums=(0,))
    def gaussian_bump(self, freq, A, f_center, width):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        return A * jnp.exp(-((freq - f_center)**2) / (2 * width**2))

    @partial(jax.jit, static_argnums=(0,))
    def gaussian_bump_sum(self, freq, A, f_center, width):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        return jnp.sum(self.gaussian_bump(freq, A, f_center, width), axis=1)

    def gaussian_bump_rj(self, freq, args, groups):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        A = self.xp.array(args[:, 0])[:, jnp.newaxis]
        f_center = self.xp.array(args[:, 1])[:, jnp.newaxis]
        width = self.xp.array(args[:, 2])[:, jnp.newaxis]

        group_unique, group_index, group_inverse, group_count = self.xp.unique(groups, return_index=True, return_counts=True, return_inverse=True)

        nleavesmax = group_count.max().item()
        A_full = self.xp.zeros((len(group_unique), nleavesmax, freq.shape[0]))
        f_center_full = self.xp.zeros((len(group_unique), nleavesmax, freq.shape[0]))
        width_full = self.xp.ones((len(group_unique), nleavesmax, freq.shape[0]))

        for i, group in enumerate(group_unique):
            idxs = self.xp.where(group_inverse == i)[0]
            A_full[i, :len(idxs)] = A[idxs]
            f_center_full[i, :len(idxs)] = f_center[idxs]
            width_full[i, :len(idxs)] = width[idxs]
        
        A_full = jnp.array(A_full)
        f_center_full = jnp.array(f_center_full)
        width_full = jnp.array(width_full)

        bump = self.gaussian_bump_sum(freq, A_full, f_center_full, width_full)
        
        return bump

    @partial(jax.jit, static_argnums=(0,))
    def log_gaussian_bump(self, freq, A, f_center, width):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        #breakpoint()    
        return A * jnp.exp(-(jnp.log(freq / f_center)**2) / (2 * width**2))

    @partial(jax.jit, static_argnums=(0,))
    def log_gaussian_bump_sum(self, freq, A, f_center, width):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        return jnp.sum(self.log_gaussian_bump(freq, A, f_center, width), axis=1)

    def log_gaussian_bump_rj(self, freq, args, groups, nin):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        A = self.xp.array(args[:, 0])[:, jnp.newaxis]
        f_center = self.xp.array(args[:, 1])[:, jnp.newaxis]
        width = self.xp.array(args[:, 2])[:, jnp.newaxis]

        group_unique, group_index, group_inverse, group_count = self.xp.unique(groups, return_index=True, return_counts=True, return_inverse=True)

        nleavesmax = group_count.max().item()
        #nin = group_unique.max().item() + 1
        # A_full = self.xp.zeros((len(group_unique), nleavesmax, freq.shape[0]))
        # f_center_full = self.xp.zeros((len(group_unique), nleavesmax, freq.shape[0]))
        # width_full = self.xp.ones((len(group_unique), nleavesmax, freq.shape[0]))

        A_full = self.xp.full(shape=(nin, nleavesmax, 1), fill_value=0.0)#self.xp.nan)        
        f_center_full = self.xp.full(shape=(nin, nleavesmax, 1), fill_value= 0.0)#self.xp.nan)
        width_full = self.xp.full(shape=(nin, nleavesmax, 1), fill_value=0.0)#self.xp.nan)

        for i, group in enumerate(group_unique):
            idxs = self.xp.where(group_inverse == i)[0]
            A_full[i, :len(idxs)] = A[idxs]
            f_center_full[i, :len(idxs)] = f_center[idxs]
            width_full[i, :len(idxs)] = width[idxs]
            
        finite_mask = self.xp.isfinite(f_center_full)

        #breakpoint()

        A_full = jnp.array(A_full)#[finite_mask])
        f_center_full = jnp.array(f_center_full)#[finite_mask])
        width_full = jnp.array(width_full)#[finite_mask])

        bump = self.log_gaussian_bump_sum(freq, A_full, f_center_full, width_full)
        
        return bump
                

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

        if self.nbackgrounds > 0:

            sgwbs_all = jnp.zeros_like(PSDS)

            for i in range(self.nbackgrounds):
                
                if len(backargs[i]) > 0:
                    back = self.backgrounds[i]
                    h2omega = self.backgrounds_fn[i](freqs, backargs[i], **kwargs[back])#[:,:,None]         

                    if self.backgroundperturbation:

                        bknots, bweights = self.prepare_interp_input(backargs[self.nbackgrounds+2*i:self.nbackgrounds+2*(i+1)], backgroups[self.nbackgrounds+2*i:self.nbackgrounds+2*(i+1)], ngroups=backargs[i].shape[0])

                        ftol_mask = self.xp.any(self.xp.any(self.xp.abs(self.xp.diff(bknots)) < self.ftol, axis=-1), axis=0)
                        
                        logperturbation = self.logperturbation_numba(freqs=freqs, knots=bknots, weights=bweights)

                        perturbation = 10**logperturbation
                        perturbation[ftol_mask] = self.xp.nan

                        perturbation = jnp.asarray(perturbation)        

                        h2omega = h2omega * perturbation

                    Shs = self.convert_to_psd(freqs, h2omega)
                    Shs = self.convert_units(Shs)
                
                sgwbs_all = sgwbs_all + Shs   

            PSDS = PSDS + sgwbs_all * self.isotropicresponse 

        if self.nforegrounds > 0:
            
            for j in range(self.nforegrounds): #compatibility with the idea of having multiple foregrounds
                fore = self.foregrounds[j]
                
                Shs = self.foregrounds_fn[j](freqs, foreargs[j], **kwargs[fore])#[:,:,None] 
                nin = Shs.shape[0]

                if self.foregroundperturbation:

                    if self.kwargs['perturbation_type'] == 'spline':

                        fknots, fweights = self.prepare_interp_input(foreargs[self.nforegrounds+2*j:self.nforegrounds+2*(j+1)], foregroups[self.nforegrounds+2*j:self.nforegrounds+2*(j+1)], ngroups=foreargs[j].shape[0])
                        ftol_mask = self.xp.any(self.xp.any(self.xp.abs(self.xp.diff(fknots)) < self.ftol, axis=-1), axis=0)
                        logperturbation = self.logperturbation_numba(freqs=freqs, knots=fknots, weights=fweights)
                        
                        perturbation = 10**logperturbation
                        perturbation[ftol_mask] = self.xp.nan
                        perturbation = jnp.asarray(perturbation)        

                        Shs = Shs * perturbation
                        Shs = self.convert_units(Shs)

                    elif self.kwargs['perturbation_type'] == 'bump':
                        bump_args, bump_groups = foreargs[self.nforegrounds+j:self.nforegrounds+(j+1)][0], foregroups[self.nforegrounds+j:self.nforegrounds+(j+1)][0]
                        bumps = self.log_gaussian_bump_rj(freqs, bump_args, bump_groups, nin) 
                        Shs = Shs + bumps[:,:,None]

                    else:
                        raise ValueError('Invalid perturbation type')

                PSDS = PSDS + Shs * self.GBresponse 
        

        return PSDS
