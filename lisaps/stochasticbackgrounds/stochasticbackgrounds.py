# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable
from ..baseclasses import GPUobject, TDIresponse
from ..constants import *


class StochasticBackgrounds(GPUobject):

    def __init__(self, backgrounds=[], background_kwargs={}, use_gpu=False, isotropicresponse=None, GBresponse=None, channels=None, units='hertz', **kwargs):

        GPUobject.__init__(self, use_gpu=use_gpu)

        self.channels = channels
        self.units = units

        self._implemented_backgrounds = self.implemented_backgrounds
        self._implemented_functions = self.implented_classes

        if not isinstance(backgrounds, list):
                backgrounds = [backgrounds]

        for back in backgrounds:
                assert back in self._implemented_backgrounds
        
        self.backgrounds = backgrounds

        self.isotropicresponse_interp = self.set_responseinterp(isotropicresponse)
        #ƒself.isotropicresponse = self.set_isotropicresponse(freqs=freqs)

        if GBresponse is not None:
            self.GBresponse_interp = self.set_responseinterp(GBresponse)

        self.set_backgrounds_fn(background_kwargs)

        if units == 'hertz':
            self.conversion = CENTRAL_FREQ**2
        elif units == 'meters':
            self.conversion = 1 / ( (2 * self.xp.pi * self.freqs) / C)**2 
        elif units == 'strain':
            self.conversion = 1 

    @property
    def conversion(self):
        return self._conversion

    @conversion.setter
    def conversion(self, freqs):
        if self.units == 'hertz':
            self._conversion = CENTRAL_FREQ**2
        elif self.units == 'meters':
            self._conversion = 1 / ( (2 * self.xp.pi * freqs) / C)**2 
        elif self.units == 'strain':
            self._conversion = 1 

    def set_backgrounds_fn(self, background_kwargs):
        self.backgrounds_fn = {}

        for back in self.backgrounds:
            if back in background_kwargs.keys():
                bkwargs = background_kwargs[back]
            else:
                bkwargs = {}
            self.backgrounds_fn[back] = self.implented_classes[back](use_gpu=self.use_gpu, **bkwargs)

    def convert_to_psd(self, freqs, h2omega):
        
        Sh = h2omega * (3 * H0h**2 / (4 * self.xp.pi**2 * freqs**3)) * (2 * self.xp.pi) #strain units
        if not hasattr(self, '_conversion'):
            self.conversion(freqs)
        return Sh * self.conversion


    @property
    def implemented_backgrounds(self):
        return [
            'sobhs',
            'cs',
            'fopt'
        ]

    @property
    def implented_classes(self):
        return {
             'sobhs': PowerLaw,
             'cs': PowerLaw,
             'fopt': PhaseTransitions
        }
    
    def set_responseinterp(self, response):

        if isinstance(response, str):
            responseinterp = TDIresponse(filename=response, use_gpu=self.use_gpu)
            
        elif isinstance(response, Callable):
            responseinterp = response
        
        else: 
            raise ValueError('if fitting for a background provide the TDI response as well. Provide either the file for the interpolation \
                                or a callable to be evaluated on a custom range of frequencies'
                )

        return responseinterp


    def set_isotropicresponse(self, freqs):
        '''
        Set the isotropic response for the TDI channels selected (only works with A, E, and T).
        '''
        idxs = [self.available_channels.index(channel) for channel in self.channels]
        self.isotropicresponse = self.isotropicresponse_interp(freqs)[:, idxs]
    
    def set_GBresponse(self, freqs):
        '''
        Set the GB response for the TDI channels selected (only works with A, E, and T).
        '''
        idxs = [self.available_channels.index(channel) for channel in self.channels]
        self.GBresponse = self.GBresponse_interp(freqs)[:, idxs]




class EnergyDensity(ABC, GPUobject):
    
    def __init__(self, use_gpu=False, interpkwargs=None):
        GPUobject.__init__(self, use_gpu, interpkwargs)

    @abstractproperty
    def ndim(self):
        pass

    @abstractmethod
    def check_ndim(self, args):
         pass
    

class PowerLaw(EnergyDensity):

    def __init__(self, use_gpu):

        EnergyDensity.__init__(self, use_gpu=use_gpu)

        self._ndim = self.ndim
        self._fknee = self.fknee

    def ndim(self):
        return 2   

    @property
    def fknee(self):
        return 3e-3
    
    def check_ndim(self, args):
        assert args.shape[-1] == self._ndim()

    def __call__(self, freqs, args):
        self.check_ndim(args)

        A = self.xp.array(args[:, 0])[:, self.xp.newaxis]
        n = self.xp.array(args[:, 1])[:, self.xp.newaxis]

        h2omega = A * (self.xp.atleast_2d(freqs) / self.fknee)**n

        return h2omega



class PhaseTransitions(EnergyDensity):
     
    def __init__(self, use_gpu, turb=False):

        EnergyDensity.__init__(self, use_gpu=use_gpu)

        self.turb = turb
        self._ndim = self.ndim
        self._n = self.n
        self._norm = self.norm
        self._zp = self.zp
        self._gstar = self.gstar

    @property
    def ndim(self):
        return 4 if self.turb else 2
    
    @property
    def n(self):
        return 7/2
    
    @property
    def norm(self):
        return 1.0
    
    @property
    def zp(self):
        return 10
    
    @property
    def gstar(self):
        return 100
    
    def check_ndim(self, args):
        assert args.shape[-1] == self._ndim()

    def h2omega_sw(self, freqs, Asw, fsw):
        fp = freqs / fsw
        h2omega = Asw * self.Csw(fp)
        return h2omega
    
    def Csw(self, fp):
        return self.norm * fp**3. * (7. / (4. + 3.*fp**2.))**self._n

    def fturb_from_sw(self, fsw):
        fturb = 27 / 26 * (8*self.xp.pi)**(1/3) * (10 / self._zp) * fsw
        return fturb
    
    def hstar(self, Tstar):
        return 165e-7 * (Tstar / 1e2) * (self.gstar / 1e2)**(1./6.)
    
    def Sturb_norm(self, freqs, fsw, Tstar):
        '''
        From ArXiv:1512.06239, I remove a term hstar from here to include it in the powerlaw amplitude
        '''

        fturb = self.fturb_from_sw(fsw, self.zp)

        fp = freqs / fturb
        hstar = self.hstar(Tstar, self.gstar)

        return fp**3 / ( (1 + fp)**(11/3) * (hstar + 8 * self.xp.pi * freqs) )
    
    def h2omega_turb(self, freqs, Aturb, fsw, Tstar):
        '''
        From ArXiv:1512.06239
        '''
        h2omega = Aturb * self.Sturb_norm(freqs=freqs, fsw=fsw, Tstar=Tstar)

        return h2omega

    def __call__(self, freqs, args):
        self.check_ndim(args)

        Asw = self.xp.array(args[:, 0])[:, self.xp.newaxis]
        fsw = self.xp.array(args[:, 1])[:, self.xp.newaxis]

        h2omega_sw = self.h2omega_sw(self, freqs, Asw, fsw)

        if self.turb:
            Aturb = self.xp.array(args[:, 2])[:, self.xp.newaxis]
            Tstar = self.xp.array(args[:, 3])[:, self.xp.newaxis]

            h2omega_turb = self.h2omega_turb(freqs=freqs, Aturb=Aturb, fsw=fsw, Tstar=Tstar)

            return h2omega_sw + h2omega_turb

        return h2omega_sw


class GBForeground(EnergyDensity):
    pass