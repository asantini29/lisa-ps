from .baseclasses import BaseNoise, TDIresponse
from .stochasticbackgrounds import StochasticBackgrounds
from typing import Any, Callable
import numpy as np

from cupyx.scipy.interpolate import Akima1DInterpolator as cupy_Akima1DInterpolator

import warnings

class Psd(BaseNoise, StochasticBackgrounds):

    def __init__(self, 
                 asdTM=2.4e-15, 
                 asdOMS=7.9e-12, 
                 fmin=1e-4, 
                 fmax=2.5e-2, 
                 freqs=None,
                 equal_arms=False,
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
                 ftol=0.1,
                 **kwargs
                 ):
        
        if noiseless:
            asdTM = 0.
            asdOMS = 0.

        BaseNoise.__init__(self, asdTM=asdTM, asdOMS=asdOMS, equal_arms=equal_arms, Ncov=Ncov, channels=channels, use_gpu=use_gpu, units=units, interpkwargs=interpkwargs)

        if not isinstance(backgrounds, list):
            backgrounds = [backgrounds]
        if not isinstance(foregrounds, list):
            foregrounds = [foregrounds]

        StochasticBackgrounds.__init__(self, backgrounds=backgrounds, background_kwargs=background_kwargs, foregrounds=foregrounds, foreground_kwargs=foreground_kwargs, isotropicresponse=isotropicresponse, GBresponse=GBresponse, channels=self.channels, units=units, use_gpu=use_gpu)

        self.PSDS_design = None

        self.fmin = fmin
        self.fmax = fmax

        self.logfmin = self.xp.log10(self.fmin)
        self.logfmax = self.xp.log10(self.fmax)

        # freqs = freqs

        assert isinstance(splineperturbation, dict)
        self.noiseperturbation = splineperturbation['noise']
        self.backgroundperturbation = splineperturbation['background']

        self.fitASDs = fitASDs
        self.set_noisefn()

        if (self.fitASDs) and (self.noiseperturbation):
            warnings.warn('Fitting both for the noise ASDs and perturbation. This will affect convergence')

        self.ftol = ftol


    def set_noisefn(self):

        if self.noiseperturbation:
            self.noisefn = self.splinemod

        else:
            if self.fitASDs:
                self.noisefn = self.constmod
            else:
                self.noisefn = self.get_PSDS                 


    def constmod(self, freqs,  args, **kwargs):

        args = args[0]
        PSDS = self.xp.empty((args.shape[0], len(freqs), self.Ncov))
        args = np.atleast_2d(args)

        asdTM, asdOMS = self.xp.asarray(args[:, 0:1]), self.xp.asarray(args[:, 1:2])
        self.asdTM = asdTM
        self.asdOMS = asdOMS

        for i, channel in enumerate(self.channels):

            if (args.shape[1] > 2) and (i > 0):
                asdTM, asdOMS = self.xp.asarray(args[:, 2*i:2*i+1]), self.xp.asarray(args[:, 2*i+1:2*i+2])
                self.asdTM = asdTM
                self.asdOMS = asdOMS

            PSDS[:, :, i] = self.available_functions[channel](freqs)  

        return PSDS
    

    def splinemod(self, freqs, args, groups, knots=None, **kwargs):
        '''
        args -> spline 
        ASDs -> TM and OMS ASDs, shape: (n_in, 2)
        '''
        #breakpoint()
        if self.fitASDs:
            self.PSDS_design = self.constmod(freqs, args[:1])    
            args = args[1:]  
            groups = groups[1:]

        else:
            if self.PSDS_design is None:
                self.set_PSDS(freqs)   

        if not isinstance(args, list):
            args = [args]
        
        if not isinstance(groups, list):
            groups = [groups]
            
        #splinepert = cupy_Akima1DInterpolator(x=self.xp.array([-4., -3.6, -3.1, -2.2, np.log10(2.5e-2)]), y=self.xp.array([0.0, -0.2, 0.3, -0.7, 0.0]))

        nin = min([arg.shape[0] for arg in args])
        PSDS = self.xp.empty((nin, len(freqs), self.Ncov))

        #knots, weights = self.prepare_interp_input(args=args, groups=groups)
        knots, weights = self.prepare_interp_input_numba(args=args, groups=groups)
        
        #breakpoint()
        ftol_mask = self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol, axis=-1)
        ftol_mask = self.xp.broadcast_to(ftol_mask, (freqs.shape[0], self.Ncov, nin)).transpose(2, 0, 1)
        PSDS[ftol_mask] = self.xp.nan
        
        logperturbation = self.logperturbation_numba(freqs=freqs, knots=knots, weights=weights)

        #breakpoint()
        # for j in range(logperturbation.shape[-1]):
        #     PSDS[:, :, j] = self.PSDS_design[:, :, j] * 10**(logperturbation[:, :, j]) #* 10**(splinepert(self.xp.log10(freqs)))[None, None, :]
        PSDS = self.PSDS_design * 10**(logperturbation)

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

        return logperturbation.reshape(self.Ncov, -1, freqs.shape[0]).transpose(1, 2, 0)
    

    def prepare_interp_input_numba(self, args, groups):
        '''
        here args is a list, probably of the fashion [ (edges), (knots_1), (knots_2), (knots_3) ]
        I want the output to be (knots position, knots weights)
        '''
        if (args[1].shape[-1] == self.Ncov + 1) or (len(args) == 2): # this means all the weights are together or there is only one spline
            edges_weights, knots_full = self.xp.asarray(args[0]), self.xp.asarray(args[1]) #always work along the `1` axis for frequency operations # TODO may have to change this, maybe (Ncov, Nin, Nfreq) is better
            groups_knots = groups[1]           
    
            leftedge_full = edges_weights[:, 0::2]
            rightedge_full = edges_weights[:, 1::2]

            groups_unique, groups_count = np.unique(groups_knots, return_counts=True)
            ngroups = groups_unique.max().item() + 1
            maxgroups = groups_count.max().item()

            knots_full_nans = self.xp.full((ngroups, maxgroups, knots_full.shape[-1]), self.xp.nan)
            #breakpoint()
            leftedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmin), leftedge_full), axis=1)[:, None, :]
            rightedge_full = self.xp.concatenate((self.xp.full((ngroups,1), self.logfmax), rightedge_full), axis=1)[:, None, :]

            for i, g in enumerate(groups_unique):
                #breakpoint()
                knots_full_nans[i, :groups_count[i]] = knots_full[groups_knots == g]

            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)
            
            positions = knots_full_nans[:,:,:1]
            weights = knots_full_nans[:,:,1:]

            ii = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, ii, axis=1)
            sortedpositions = self.xp.repeat(sortedpositions, self.Ncov, axis = -1).transpose(2,0,1)

            sortedweights = self.xp.take_along_axis(weights, ii, axis=1).transpose(2,0,1)

        else:

            edges_weights = self.xp.asarray(args[0])
            leftedge_full = edges_weights[:, 0::2]
            rightedge_full = edges_weights[:, 1::2]
            
            args_knots = [self.xp.asarray(arg) for arg in args[1:]] #still a list
            groups_knots = groups[1:] #still a list

            groups_unique = np.unique(groups[0])
            ngroups = groups_unique.max().item() + 1

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

                # for i, g in enumerate(group_unique):
                #     knots_full_nans[i, :group_count[i], j] = arg[:, 0][group == g]
                #     knots_full_nans[i, :group_count[i], self.Ncov + j] = arg[:, 1][group == g]
            
            knots_full_nans = self.xp.concatenate((leftedge_full, knots_full_nans, rightedge_full), axis = 1)

            positions = knots_full_nans[:,:,:self.Ncov]
            weights = knots_full_nans[:,:,self.Ncov:]

            ii = self.xp.argsort(positions, axis = 1)
            sortedpositions = self.xp.take_along_axis(positions, ii, axis=1).transpose(2,0,1)
            sortedweights = self.xp.take_along_axis(weights, ii, axis=1).transpose(2,0,1)

        return sortedpositions, sortedweights
                

    def prepare_interp_input(self, args):
        '''
        here args is a list, probably of the fashion [ (edges), (knots) ] for a single channel.
        I'd like to move away from dictionaries.
        I want the output to be (knots position, knots weights)
        '''
        #breakpoint()

        if len(args) == 1: # * if args is not a list it means that it is an array of weights, so it's fine
            return self.knots, args[0]    # * if not using rj provide the knots positions to the constructor
        
        else:
            edges_weights, knots_full = self.xp.atleast_2d(self.xp.asarray(args[0])), self.xp.atleast_2d(self.xp.asarray(args[1])) #always work along the `1` axis for frequency operations

            idxs_sorted = np.argsort(knots_full[:, 0])
            knots_full = knots_full[idxs_sorted]

            knots_positions = knots_full[:, 0]
            knots_weights = knots_full[:, 1:]

            knots = self.xp.hstack((self.logfmin, knots_positions, self.logfmax))
            weights = self.xp.concatenate((edges_weights[:,0::2], knots_weights, edges_weights[:,1::2]), axis=0).T

            return knots, weights


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

            if not hasattr(self, 'isotropicresponse'):
                self.set_isotropicresponse(freqs)
            
            sgwbs_all = self.xp.zeros_like(PSDS)

            for i in range(self.nbackgrounds):
                
                if len(backargs[i]) > 0:
                    back = self.backgrounds[i]
                    h2omega = self.backgrounds_fn[i](freqs, backargs[i], **kwargs[back])
                    Sh = [self.convert_to_psd(freqs, h2omega) for i in range(self.Ncov)]

                    Shs = self.xp.array(Sh).transpose(1, 2, 0)

                    response = self.isotropicresponse[self.xp.newaxis, :, :]            

                    if self.backgroundperturbation:

                        bknots, bweights = self.prepare_interp_input_numba(backargs[self.nbackgrounds+2*i:self.nbackgrounds+2*(i+1)], backgroups[self.nbackgrounds+2*i:self.nbackgrounds+2*(i+1)])

                        ftol_mask = self.xp.any(self.xp.abs(self.xp.diff(bknots)) < self.ftol, axis=-1)
                        ftol_mask = self.xp.broadcast_to(ftol_mask, (freqs.shape[0], self.Ncov, PSDS.shape[0])).transpose(2, 0, 1)
                        
                        logperturbation = self.logperturbation_numba(freqs=freqs, knots=bknots, weights=bweights)

                        Shs = Shs * 10**logperturbation
                
                sgwbs_all = sgwbs_all + Shs   

            PSDS = PSDS + sgwbs_all * response 
        
        if self.nforegrounds > 0:

            if not hasattr(self, 'GBresponse'):
                self.set_GBresponse(freqs)

            sgwfs_all = self.xp.zeros_like(PSDS)

            for i in range(self.nforegrounds):
                
                if len(foreargs[i]) > 0:
                    fore = self.foregrounds[i]
                    h2omega = self.foregrounds_fn[i](freqs, foreargs[i], **kwargs[fore])
                    Sh = [self.convert_to_psd(freqs, h2omega) for i in range(self.Ncov)]

                    # Shs = self.xp.array(Sh).transpose(1, 2, 0)
                    # response = self.GBresponse[self.xp.newaxis, :, :]
                    # sgwbs_all += Shs * response 


        return PSDS
