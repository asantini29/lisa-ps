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

        if len(backgrounds) > 0:
            StochasticBackgrounds.__init__(self, backgrounds=backgrounds, background_kwargs=background_kwargs, isotropicresponse=isotropicresponse, GBresponse=GBresponse, channels=self.channels, units=units, use_gpu=use_gpu)

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

        key = [*args][0]
        PSDS = self.xp.empty((args[key].shape[0], len(freqs), self.Ncov))
        ASDargs = np.atleast_2d(args[key])
        asdTM, asdOMS = self.xp.asarray(ASDargs[:, 0, np.newaxis]), self.xp.asarray(ASDargs[:, 1, np.newaxis])

        self.update_params(asdTM=asdTM, asdOMS=asdOMS)    

        for i, channel in enumerate(self.channels):
            PSDS[:, :, i] = self.available_functions[channel](freqs)  

        return PSDS
    

    def splinemod(self, freqs, args, knots=None, **kwargs):
        '''
        args -> spline 
        ASDs -> TM and OMS ASDs, shape: (n_in, 2)
        '''

        if self.fitASDs:
            self.PSDS_design = self.constmod(freqs, args)      

        else:
            if self.PSDS_design is None:
                self.set_PSDS(freqs)   

        nin = min([args[key].shape[0] for key in [*args]])
        PSDS = self.xp.empty((nin, len(freqs), self.Ncov))
        
        if knots is not None:
            weights = self.xp.asarray(args['splines'].reshape(-1, len(knots)))

        else:      
            '''
            RJ, provide fmin and fmax when constructing the class
            '''

            if args['knots'].shape[0] > 0:
                idxs = np.argsort(args['knots'][:, 0])
                weights = self.xp.concatenate((self.xp.asarray(args['edges'][:,0::2]), self.xp.asarray(args['knots'][:,1:][idxs]), self.xp.asarray(args['edges'][:,1::2])), axis=0).T#weights of the knots
                knots = self.xp.hstack((self.logfmin, self.xp.array(args['knots'][:, 0][idxs]), self.logfmax))
            else:

                weights = self.xp.concatenate((self.xp.asarray(args['edges'][:,0::2]), self.xp.asarray(args['edges'][:,1::2])), axis=0).T#weights of the knots
                knots = self.xp.hstack((self.logfmin, self.logfmax))

            if self.xp.any(self.xp.abs(self.xp.diff(knots)) < self.ftol):            
                PSDS[:, :, :] = self.xp.nan
                return PSDS

        logperturbation = self.logperturbation(freqs=freqs, knots=knots, weights=weights)
        
        for i in range(self.Ncov):
            PSDS[:, :, i] = self.PSDS_design[:, :, i] * 10**(logperturbation[:, :, i])

        return PSDS
    
    def logperturbation(self, freqs, knots, weights):
        '''
        Spline perturbation.
        '''
        interp = self.interp(knots, weights, **self.interpkwargs)
        logperturbation = (interp(self.xp.log10(freqs))).reshape(-1, len(freqs), self.Ncov) #put it in the same shape of PSDS
        
        return logperturbation

    

    def __call__(self, freqs, noiseargs=None, backargs={}, **kwargs):
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
        if noiseargs is not None:
            PSDS = self.noisefn(freqs=freqs, args=noiseargs, **kwargs['noise'])

        else:
            PSDS = self.get_PSDS(freqs) * self.xp.ones(backargs[self.back[0]].shape[0])[:, self.xp.newaxis, self.xp.newaxis]

        if hasattr(self, 'backgrounds'):

            if not hasattr(self, 'isotropicresponse'):
                self.set_isotropicresponse(freqs)
            
            sgwbs_all = self.xp.zeros_like(PSDS)

            for back in self.backgrounds:
                
                if len(backargs[back]) > 0:

                    h2omega = self.backgrounds_fn[back](freqs, backargs[back], **kwargs[back])
                    Sh = [self.convert_to_psd(freqs, h2omega) for i in range(self.Ncov)]

                    Shs = self.xp.array(Sh).transpose(1, 2, 0)

                    response = self.isotropicresponse[self.xp.newaxis, :, :] if back != 'gb' else self.GBresponse[self.xp.newaxis, :, :]
        
                    sgwbs_all += Shs * response        

            PSDS = PSDS + sgwbs_all

        return PSDS
