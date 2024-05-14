# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable
from .baseclasses import GPUobject, TDIresponse
from .constants import *
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial


class StochasticBackgrounds(GPUobject):

    def __init__(self, 
                 backgrounds=[], 
                 background_kwargs={}, 
                 foregrounds=[], 
                 foreground_kwargs={}, 
                 use_gpu=False, 
                 TDIsetup='AET', 
                 isotropicresponse=None, 
                 GBresponse=None, 
                 channels=None, 
                 units='strain', 
                 correct_sagnac=True, 
                 **kwargs):
        """
        Initialize the StochasticBackgrounds object.

        Args:
            backgrounds (list): List of background types to include.
            background_kwargs (dict): Keyword arguments for background functions.
            foregrounds (list): List of foreground types to include.
            foreground_kwargs (dict): Keyword arguments for foreground functions.
            use_gpu (bool): Flag indicating whether to use GPU acceleration.
            TDIsetup (str): TDI setup to use.
            isotropicresponse (None or callable): Isotropic response function.
            GBresponse (None or callable): GB response function.
            channels (None or list): List of channels to include.
            units (str): Units of the output.
            correct_sagnac (bool): Flag indicating whether to correct for Sagnac effect.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If a background type is not supported.

        """
        GPUobject.__init__(self, use_gpu=use_gpu, **kwargs)

        self.units = units

        self._implemented_backgrounds = self.implemented_backgrounds
        self._implemented_functions = self.implented_classes

        self.available_channels = ['AA', 'EE', 'TT', 'XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']

        if not isinstance(backgrounds, list):
            backgrounds = [backgrounds]

        for back in backgrounds:
            if back not in self._implemented_backgrounds:
                raise ValueError(str(back) + ' is not a supported background')
        
        self.backgrounds = backgrounds
        self.nbackgrounds = len(backgrounds)

        self.TDIsetup = TDIsetup

        if channels is not None:
            self.channels = channels
        else:
            if TDIsetup == 'AET':
                self.channels = ['AA', 'EE', 'TT']
            elif TDIsetup == 'XYZ':
                self.channels = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
            else:
                raise ValueError('TDIsetup not recognized. Choose between AET and XYZ')
            
        self.correct_sagnac = correct_sagnac
        self.isotropicresponse_interp = self.set_responseinterp(isotropicresponse)

        if not isinstance(foregrounds, list):
            foregrounds = [foregrounds]

        for fore in foregrounds:
            assert fore in self._implemented_foregrounds
        
        self.foregrounds = foregrounds
        self.nforegrounds = len(foregrounds)

        if GBresponse is not None:
            self.GBresponse_interp = self.set_responseinterp(GBresponse)

        self.set_backgrounds_fn(background_kwargs)
        self.set_foregrounds_fn(foreground_kwargs)

    @property
    def conversion(self):
        return self._conversion

    @conversion.setter
    def conversion(self, freqs):
        if self.units == 'hertz':
            self._conversion = CENTRAL_FREQ**2
        elif self.units == 'meters':
            self._conversion = 1 / ( (2 * jnp.pi * freqs) / C)**2 
        elif self.units == 'strain':
            self._conversion = 1 

    @partial(jax.jit, static_argnums=(0,))
    def convert_to_psd(self, freqs, h2omega):
        
        Sh = h2omega * (3 * H0h**2 / (4 * jnp.pi**2 * freqs[None, :, None]**3)) * (2 * jnp.pi) #strain units
        if not hasattr(self, '_conversion'):
            self.conversion = freqs
        return Sh * self.conversion
    
    def set_backgrounds_fn(self, background_kwargs):
        self.backgrounds_fn = []
        for back in self.backgrounds:
            if back in background_kwargs.keys():
                bkwargs = background_kwargs[back]
            else:
                bkwargs = {}
            self.backgrounds_fn += [self.implented_classes[back](use_gpu=self.use_gpu, **bkwargs)]

    def set_foregrounds_fn(self, foreground_kwargs):
        self.foregrounds_fn = []
        for fore in self.foregrounds:
            if fore in foreground_kwargs.keys():
                fkwargs = foreground_kwargs[fore]
            else:
                bkwargs = {}
            self.foregrounds_fn += [self.implented_classes[fore](use_gpu=self.use_gpu, **fkwargs)]

    @property
    def implemented_backgrounds(self):
        return [
            'sobhs',
            'cs',
            'fopt'
        ]
    
    @property
    def implemented_foregrounds(self):
        return [

        ]

    @property
    def implented_classes(self):
        return {
             'sobhs': PowerLaw,
             'cs': PowerLaw,
             'fopt': PhaseTransitions
        }
    
    @property
    def injection(self):
        return {
            'sobhs': jnp.array([3.4e-13, 2/3]),
            'cs': jnp.array([5.5e-12, 0]),
            'fopt': jnp.array([4.22e-12, 9.86e-4, 2.88e-14, 200])
        }
    
    def set_responseinterp(self, response):

        if response is None: #use default files
            TFdir = '/data/asantini/packages/lisa-ps/utils/'

            if self.TDIsetup == 'AET':
                response = TFdir + 'TDItransferfunction_AET.csv' if self.correct_sagnac else TFdir + 'TDItransferfunction_AET_nosagnac.csv'
            elif self.TDIsetup == 'XYZ':
                response = [TFdir + 'TDItransferfunction_XYZreal.csv', TFdir + 'TDItransferfunction_XYZimag.csv']

            else:
                raise ValueError('TDIsetup not recognized. Choose between AET and XYZ')

        if isinstance(response, (str, list)):
            responseinterp = TDIresponse(filename=response, use_gpu=self.use_gpu)
            
        elif isinstance(response, Callable):
            responseinterp = response
        
        else: 
            raise ValueError('if fitting for a background provide the TDI response as well. Provide either the file for the interpolation \
                                or a callable to be evaluated on a custom range of frequencies')

        return responseinterp


    def set_isotropicresponse(self, freqs):
        '''
        Set the isotropic response for the TDI channels selected (only works with A, E, and T).
        '''
        if self.TDIsetup == 'AET':
            idxs = [self.available_channels.index(channel) for channel in self.channels]
            self.isotropicresponse = self.isotropicresponse_interp(freqs)[:, idxs]
        else:
            self.isotropicresponse = self.isotropicresponse_interp(freqs)
    
    def set_GBresponse(self, freqs):
        '''
        Set the GB response for the TDI channels selected (only works with A, E, and T).
        '''
        if self.TDIsetup == 'AET':
            idxs = [self.available_channels.index(channel) for channel in self.channels]
            self.GBresponse = self.GBresponse_interp(freqs)[:, idxs]
        else:
            self.GBresponse = self.GBresponse_interp(freqs)



class EnergyDensity(ABC, GPUobject):
    
    def __init__(self, use_gpu=False, interpkwargs=None):
        GPUobject.__init__(self, use_gpu, interpkwargs)

    @property 
    @abstractmethod
    def ndim(self):
        pass

    @abstractmethod
    def check_ndim(self, args):
         pass
    
    @abstractmethod
    def __call__(self, freqs, args):
        pass

    

class PowerLaw(EnergyDensity):

    def __init__(self, use_gpu):

        EnergyDensity.__init__(self, use_gpu=use_gpu)

        self._ndim = self.ndim()
        self._fknee = self.fknee

    def ndim(self):
        return 2   

    @property
    def fknee(self):
        return 3e-3
    
    def check_ndim(self, args):
        assert args.shape[-1] == self._ndim

    def __call__(self, freqs, args):
        #self.check_ndim(args)

        A = jnp.array(args[:, 0])[:, jnp.newaxis]
        n = jnp.array(args[:, 1])[:, jnp.newaxis]

        freqs = jnp.atleast_2d(freqs)

        h2omega = self.h2omega(freqs, A, n)

        return h2omega
    
    @partial(jax.jit, static_argnums=(0,))
    def h2omega(self, freqs, A, n):
        return A * (freqs / self.fknee)**n
        
    
    @property
    def true_params(self):
        return self._true_params
    
    @true_params.setter
    def true_params(self, true_params):
        self._true_params = true_params



class PhaseTransitions(EnergyDensity):
     
    def __init__(self, use_gpu, turb=False):

        EnergyDensity.__init__(self, use_gpu=use_gpu)

        self.turb = turb
        self._ndim = self.ndim()
        self._n = self.n
        self._norm = self.norm
        self._zp = self.zp
        self._gstar = self.gstar

    def ndim(self):
        return  4 if self.turb else 2
        
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
        assert args.shape[-1] == self._ndim, args.shape

    @partial(jax.jit, static_argnums=(0,))
    def h2omega_sw(self, freqs, Asw, fsw):
        fp = freqs / fsw
        h2omega = Asw * self.Csw(fp)
        return h2omega
    
    @partial(jax.jit, static_argnums=(0,))
    def Csw(self, fp):
        return self.norm * fp**3. * (7. / (4. + 3.*fp**2.))**self._n

    def fturb_from_sw(self, fsw):
        fturb = 27 / 26 * (8*jnp.pi)**(1/3) * (10 / self._zp) * fsw
        return fturb
    
    def hstar(self, Tstar):
        return 165e-7 * (Tstar / 1e2) * (self.gstar / 1e2)**(1./6.)
    
    @partial(jax.jit, static_argnums=(0,))
    def Sturb_norm(self, freqs, fsw, Tstar):
        '''
        From ArXiv:1512.06239, I remove a term hstar from here to include it in the powerlaw amplitude
        '''

        fturb = self.fturb_from_sw(fsw)

        fp = freqs / fturb
        hstar = self.hstar(Tstar)

        return fp**3 / ( (1 + fp)**(11/3) * (hstar + 8 * jnp.pi * freqs) )
    
    @partial(jax.jit, static_argnums=(0,))
    def h2omega_turb(self, freqs, Aturb, fsw, Tstar):
        '''
        From ArXiv:1512.06239
        '''
        h2omega = Aturb * self.Sturb_norm(freqs=freqs, fsw=fsw, Tstar=Tstar)

        return h2omega

    def __call__(self, freqs, args):
        #self.check_ndim(args)

        Asw = jnp.asarray(args[:, 0])[:, jnp.newaxis]
        fsw = jnp.asarray(args[:, 1])[:, jnp.newaxis]

        h2omega_sw = self.h2omega_sw(freqs, Asw, fsw)

        if self.turb:
            Aturb = jnp.asarray(args[:, 2])[:, jnp.newaxis]
            Tstar = jnp.asarray(args[:, 3])[:, jnp.newaxis]

            h2omega_turb = self.h2omega_turb(freqs=freqs, Aturb=Aturb, fsw=fsw, Tstar=Tstar)

            return h2omega_sw + h2omega_turb

        return h2omega_sw
    
    @property
    def true_params(self):
        return self._true_params
    
    @true_params.setter
    def true_params(self, true_params):
        self._true_params = true_params


class GBForeground(EnergyDensity):
    pass