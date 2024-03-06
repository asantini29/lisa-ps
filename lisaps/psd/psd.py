from ..baseclasses import BaseNoise, TDIresponse
from ..stochasticbackgrounds import StochasticBackgrounds
from typing import Any, Callable

import numpy as np

class Psd(BaseNoise, StochasticBackgrounds):

    def __init__(self, 
                 asdTM=2.4e-15, 
                 asdOMS=7.9e-12, 
                 fmin=1e-4, 
                 fmax=2.5e-2, 
                 freqs=None,
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

        BaseNoise.__init__(self, asdTM=asdTM, asdOMS=asdOMS, Ncov=Ncov, channels=channels, use_gpu=use_gpu, units=units, interpkwargs=interpkwargs)

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

        #asdTM, asdOMS = self.xp.asarray(args[:, 0, np.newaxis]), self.xp.asarray(args[:, 1, np.newaxis])
        asdTM, asdOMS = self.xp.asarray(args[:, 0:1]), self.xp.asarray(args[:, 1:2])
        self.update_params(asdTM=asdTM, asdOMS=asdOMS)    

        for i, channel in enumerate(self.channels):

            if (args.shape[1] > 2) and (i > 0):
                # asdTM, asdOMS = self.xp.asarray(args[:, i, np.newaxis]), self.xp.asarray(args[:, i+1, np.newaxis])
                asdTM, asdOMS = self.xp.asarray(args[:, 2*i:2*i+1]), self.xp.asarray(args[:, 2*i+1:2*i+2])
                self.update_params(asdTM=asdTM, asdOMS=asdOMS)

            PSDS[:, :, i] = self.available_functions[channel](freqs)  

        return PSDS
    

    def splinemod(self, freqs, args, knots=None, **kwargs):
        '''
        args -> spline 
        ASDs -> TM and OMS ASDs, shape: (n_in, 2)
        '''
        #breakpoint()
        if self.fitASDs:
            self.PSDS_design = self.constmod(freqs, args[:1])    
            args = args[1:]  

        else:
            if self.PSDS_design is None:
                self.set_PSDS(freqs)   

        if not isinstance(args, list):
            args = [args]

        nin = min([arg.shape[0] for arg in args])
        PSDS = self.xp.empty((nin, len(freqs), self.Ncov))
        
        # inputs = max(1, int(len(args) / 2))
        
        # for i in range(inputs):
        #     knots, weights = self.prepare_interp_input(args=args[2*i : 2*i+2])

        #     if self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol):  
        #         print('Knots are too close')          
        #         PSDS[:, :, :] = self.xp.nan
        #         return PSDS

        #     logperturbation = self.logperturbation(freqs=freqs, knots=knots, weights=weights)
        
        #     for j in range(logperturbation.shape[-1]):
        #         PSDS[:, :, i+j] = self.PSDS_design[:, :, i] * 10**(logperturbation[:, :, j])

        knots, weights = self.prepare_interp_input(args=args)
        if self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol):  
            #print('Knots are too close')          
            PSDS[:, :, :] = self.xp.nan
            return PSDS
        
        logperturbation = self.logperturbation(freqs=freqs, knots=knots, weights=weights)
        for j in range(logperturbation.shape[-1]):
            PSDS[:, :, j] = self.PSDS_design[:, :, j] * 10**(logperturbation[:, :, j])

        return PSDS
    
    def logperturbation(self, freqs, knots, weights):
        '''
        Spline perturbation.
        '''
        interp = self.interp(knots, weights, **self.interpkwargs)
        logperturbation = (interp(self.xp.log10(freqs))).reshape(-1, len(freqs), weights.shape[0]) #put it in the same shape of PSDS
        
        return logperturbation
    

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


    def __call__(self, freqs, noiseargs=[], backargs=[], foreargs=[], **kwargs):
        '''
        compute the total PSD in each channel.
        
        Args:
            freqs (array): array of frequencies at which the PSD is computed.
            noiseargs (array): arguments to be passed to the noise function. The shape of the array must be ``(num positions, ndim)``.
            backargs (dict): arguments to be passed to the background functions. 
                The keys must be the names of the different backgrounds (see ``psd.backlist`` for a list of implemented backgrounds).
                Entries are arrays of shape  ``(num positions, ndim)``.
            kwargs (dict): eventual kwargs to be passed to the noise and/or background functions. It is a dictionary of dictionaries to 
                be passed to the individual functions.
                The keys must be:
                1) `noise`: for the noise function;
                2) the name of the background for the relative function
        '''
        PSDS = self.noisefn(freqs=freqs, args=noiseargs, **kwargs['noise'])
        # TODO: make sure the dimensions are fine
        # PSDS = self.get_PSDS(freqs) * self.xp.ones(backargs[self.back[0]].shape[0])[:, self.xp.newaxis, self.xp.newaxis]
        if self.xp.any(self.xp.isnan(PSDS)):
            return PSDS
        
        else:
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
            
                        sgwbs_all += Shs * response         

                PSDS = PSDS + sgwbs_all
            
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
