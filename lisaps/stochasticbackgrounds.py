# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod
from typing import Any, Callable
from .baseclasses import GPUobject, TDIresponse
from .constants import *
import numpy as np

import warnings

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial


class StochasticContribution(GPUobject):
    """
    StochasticContribution class for handling stochastic background and foreground contributions in TDI channels.
    Attributes:
        available_channels (list): List of available TDI channels.
        custom_arms (bool): Flag indicating whether to use custom armlength.
        armlength (float): Length of the arms.
        channels (list): List of channels to include.
        nbackgrounds (int): Number of background types.
        isotropicresponse_interp (callable): Interpolated isotropic response function.
        nforegrounds (int): Number of foreground types.
        GBresponse_interp (callable): Interpolated GB response function.
        convert_to_psd (callable): Function to convert to PSD.
        backgrounds_fn (list): List of background functions.
        foregrounds_fn (list): List of foreground functions.
        implemented_backgrounds (list): List of implemented background types.
        implemented_foregrounds (list): List of implemented foreground types.
        implemented_classes (dict): Dictionary of implemented classes.
        injection (dict): Dictionary of injection values.
        TDIsetup (str): TDI setup to use.
        correct_sagnac (bool): Flag indicating whether to correct for Sagnac effect.
        units (str): Units of the output.
    """

    def __init__(self, 
                 backgrounds=[], 
                 background_kwargs={}, 
                 foregrounds=[], 
                 foreground_kwargs={}, 
                 injection=None,
                 use_gpu=False, 
                 interpkwargs=None,
                 custom_armlength=None,
                 TDIsetup='AET',
                 isotropicresponse=None, 
                 GBresponse=None, 
                 channels=None, 
                 units='strain', 
                 correct_sagnac=True, 
                 **kwargs):
        """
        Initialize the StochasticContribution object.

        Args:
            backgrounds (list): List of background types to include.
            background_kwargs (dict): Keyword arguments for background functions.
            foregrounds (list): List of foreground types to include.
            foreground_kwargs (dict): Keyword arguments for foreground functions.
            use_gpu (bool): Flag indicating whether to use GPU acceleration.
            custom_armlength (None or float): Custom armlength to use.
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
        GPUobject.__init__(self, use_gpu=use_gpu, interpkwargs=interpkwargs, **kwargs)

        self.units = units

        self._implemented_backgrounds = self.implemented_backgrounds
        self._implemented_foregrounds = self.implemented_foregrounds
        self._implemented_functions = self.implemented_classes

        self.available_channels = ['AA', 'EE', 'TT', 'XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']

        if injection is None:
            injection = {
                'sobhs': jnp.array([3.4e-13, 2/3]),
                'cs': jnp.array([5.5e-12, 0]),
                'fopt': jnp.array([4.22e-12, 9.86e-4, 2.88e-14, 200]),
                'galactic': jnp.array([1.5e-15, 1e-3, 2.5, 1e-3, 1e-3])
            }
        
        self.injection = injection

        self.TDIsetup = TDIsetup

        if custom_armlength is not None:
            self.armlength = custom_armlength
            self.custom_armlength = True
        else:
            self.armlength = ARMLENGTH_AVERAGE
            self.custom_armlength = False
        
        #breakpoint()

        if channels is not None:
            if not isinstance(channels, list):
                channels = [channels]
            self.channels = channels
        else:
            if TDIsetup.split(' ')[0] == 'AET':
                self.channels = ['AA', 'EE', 'TT']
            elif TDIsetup.split(' ')[0] == 'XYZ':
                self.channels = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
            else:
                raise ValueError('TDIsetup not recognized. Choose between `AET 1.5`, `AET 2.0`, `XYZ 1.5`, `XYZ 2.0`')

        # backgrounds setup
        if not isinstance(backgrounds, list):
            backgrounds = [backgrounds]

        for back in backgrounds:
            if back not in self._implemented_backgrounds:
                raise ValueError(str(back) + ' is not a supported background')
        
        self.backgrounds = backgrounds
        self.nbackgrounds = len(backgrounds)
            
        self.correct_sagnac = correct_sagnac

        self.isotropicresponse_interp = self.set_responseinterp(isotropicresponse)

        # foregrounds setup
        if not isinstance(foregrounds, list):
            foregrounds = [foregrounds]

        for fore in foregrounds:
            if fore not in self._implemented_foregrounds:
                raise ValueError(str(fore) + ' is not a supported foreground')
        
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
        """
        Compute the psd given the background energy density functional form. 
        It returns the psd per polarization integrated over the sky.

        Args:
            freqs (array): Array of frequencies.    
            h2omega (array): Array of h2omega values.

        Returns:
            array: Array of Sh values. 
        """
        Sh = h2omega * (3 * H0h**2 / (4 * jnp.pi**2 * freqs[None, :, None]**3)) #strain units
        return Sh / 2 # for the two polarizations
    
    @partial(jax.jit, static_argnums=(0,))
    def convert_units(self, Sh):
        """
        Convert the power spectral density (PSD) to the desired units.

        Args:
            Sh (array): Array of PSD values.
        
        Returns:
            array: Array of PSD values in the desired units.
        """
        return Sh * self.conversion

    
    def set_backgrounds_fn(self, background_kwargs):
        """
        Set the background functions based on the provided keyword arguments.
        This method initializes the `backgrounds_fn` attribute with instances of the
        implemented background classes. Each background class is instantiated with
        the corresponding keyword arguments from `background_kwargs`.
        Args:
            background_kwargs (dict): A dictionary where keys are background names
                                      and values are dictionaries of keyword arguments
                                      to be passed to the background class constructors.
        """

        self.backgrounds_fn = []
        for back in self.backgrounds:
            if back in background_kwargs.keys():
                bkwargs = background_kwargs[back]
            else:
                bkwargs = {}
            cls = self.implemented_classes[back](use_gpu=self.use_gpu, **bkwargs)
            if back in self.injection.keys():
                cls.true_params = self.injection[back]
            self.backgrounds_fn += [cls]

    def set_foregrounds_fn(self, foreground_kwargs):
        """
        Set the foreground functions based on the provided keyword arguments.
        This method initializes the `foregrounds_fn` attribute with instances of the
        implemented foreground classes. Each foreground class is instantiated with
        the corresponding keyword arguments from `foreground_kwargs`.
        Args:
            foreground_kwargs (dict): A dictionary where keys are foreground names
                                      and values are dictionaries of keyword arguments
                                      to be passed to the foreground class constructors.
        """

        self.foregrounds_fn = []
        for fore in self.foregrounds:
            if fore in foreground_kwargs.keys():
                fkwargs = foreground_kwargs[fore]
            else:
                fkwargs = {}
            cls = self.implemented_classes[fore](use_gpu=self.use_gpu, **fkwargs)
            if fore in self.injection.keys():
                cls.true_params = self.injection[fore]
            self.foregrounds_fn += [cls]

    @property
    def implemented_backgrounds(self):
        return [
            'powerlaw',
            'sobhs',
            'cs',
            'fopt'
        ]
    
    @property
    def implemented_foregrounds(self):
        return [
            'galactic',
            'galactic_bump'
        ]

    @property
    def implemented_classes(self):
        return {
             'powerlaw': PowerLaw,
             'sobhs': PowerLaw,
             'cs': PowerLaw,
             'fopt': PhaseTransitions,
             'galactic': HyperbolicTangent,
             'galactic_bump': GaussianBumpHyperbolicTangent
        }
    
    @property
    def injection(self):
        return self._injection
    
    @injection.setter
    def injection(self, injection):
        if not isinstance(injection, dict):
            raise ValueError("Injection must be a dictionary.")
        self._injection = injection
    
    def update_injection(self, source, values):
        """
        Update the injection values for a given source.
        Parameters:
        source (str): The source to update.
        values (array): The new values for the source.
        """
        self._injection[source] = values

    
    def set_responseinterp(self, response):
        """
        Set the response interpolant for the TDI (Time Delay Interferometry) response.
        Parameters:
        response (str, list, or Callable): The response can be provided in three forms:
            - None: Use default files based on the TDI setup and arm length configuration.
            - str or list: Filename(s) of the TDI response files to be used for interpolation.
            - Callable: A custom function to be evaluated on a range of frequencies.
        Returns:
        responseinterp: An interpolant for the TDI response, either created from the provided files or the custom function.
        Raises:
        ValueError: If the TDI setup is not recognized or if the response is not provided in an acceptable form.
        Notes:
        - If the response is None, default TDI response files will be used, which are valid only in the interval [3e-5, 5.9e-2] Hz.
        - The default files are located in the directory 'lisa-ps/utils/'.
        """

        if response is None: #use default files
            # get the package path
            TFdir = os.path.join(os.path.dirname(__file__), '..', 'utils/')

            if self.custom_armlength: #assume constant equal armlengths
                files = ['TDItransferfunction_AET_equal.csv', 'TDItransferfunction_AET_equal_nosagnac.csv', 'TDItransferfunction_XYZreal_equal.csv', 'TDItransferfunction_XYZimag_equal.csv']
            
            else: #assume average armlengths
                files = ['TDItransferfunction_AET.csv', 'TDItransferfunction_AET_nosagnac.csv', 'TDItransferfunction_XYZreal.csv', 'TDItransferfunction_XYZimag.csv']

            if self.TDIsetup.split(' ')[0] == 'AET':
                response = TFdir + files[0] if self.correct_sagnac else TFdir + files[1]
            elif self.TDIsetup.split(' ')[0] == 'XYZ':
                response = [TFdir + files[2], TFdir + files[3]]

            else:
                raise ValueError('TDIsetup not recognized. Choose between AET and XYZ')

            warnings.warn('Using default TDI response files to build up an interpolant. They hold only in the interval [3e-5. 5.9e-2] Hz.')

        if isinstance(response, (str, list)):
            responseinterp = TDIresponse(filename=response, use_gpu=self.use_gpu)
            
        elif isinstance(response, Callable):
            responseinterp = response
        
        else: 
            raise ValueError('if fitting for a background provide the TDI response as well. Provide either the file for the interpolation \
                                or a callable to be evaluated on a custom range of frequencies')

        return responseinterp

    def get_isotropicresponse(self, freqs):
        """
        Get the isotropic response for the TDI channels selected (this only works with A, E, and T).

        Parameters:
        freqs (array): Array of frequencies.

        Returns:
        isotropicresponse (array): Array of isotropic response values.
        """
        if self.TDIsetup == 'AET':
            idxs = [self.available_channels.index(channel) for channel in self.channels]
            isotropicresponse = self.isotropicresponse_interp(freqs)[:, idxs]
        else:
            isotropicresponse = self.isotropicresponse_interp(freqs)
        
        return isotropicresponse
    
    def get_GBresponse(self, freqs):
        """
        Get the GB response for the TDI channels selected (this only works with A, E, and T).

        Parameters:
        freqs (array): Array of frequencies.

        Returns:
        GBresponse (array): Array of GB response values.
        """
        
        GBresponse = self.GBresponse_interp(freqs)

        return GBresponse

    def set_isotropicresponse(self, freqs):
        """
        Set the isotropic response for the TDI channels selected (only works with A, E, and T).
    
        Parameters:
        freqs (array): Array of frequencies.
        """
        
        self.isotropicresponse = self.get_isotropicresponse(freqs)
    
    def set_GBresponse(self, freqs, analytical=False):
        """
        Set the GB response for the TDI channels selected (only works with A, E, and T).
        
        Parameters:
        freqs (array): Array of frequencies.
        analytical (bool): Flag indicating whether to use the analytical expression for the GB response.
        """

        if self.TDIsetup.split(' ')[0] == 'AET':
            idxs = [self.available_channels.index(channel) for channel in self.channels]
        if analytical:
            self.GBresponse = self.analytical_GBresponse(freqs)[:, idxs]
        else:
            self.GBresponse = self.get_GBresponse(freqs)[:, idxs]

    
    @partial(jax.jit, static_argnums=(0,))
    def analytical_GBresponse(self, freqs):
        """
        Calculate the analytical gravitational background response for given frequencies.
        Parameters
        ----------
        freqs : array-like
            Array of frequency values at which to calculate the response.
        Returns
        -------
        jnp.ndarray
            A 2D array where each row corresponds to the response [respA, respE, respT] 
            for a given frequency in `freqs`. The responses are calculated as follows:
            - respA: Response A, proportional to 6 * x^2 * sin(x)^2
            - respE: Response E, proportional to 6 * x^2 * sin(x)^2
            - respT: Response T, which is always 0.0 * x^2
        Notes
        -----
        - `x` is defined as 2 * pi * freqs * self.armlength.
        - This function uses JAX's numpy (jnp) for array operations.
        """
        
        x = 2 * jnp.pi * freqs * self.armlength

        tdi2_factor = 4 * jnp.sin(2 * x)**4 if self.TDIsetup.split(' ')[1] == '2.0' else 1.0

        respA =  6 * x**2 * jnp.sin(x)**2 * tdi2_factor
        respE =  6 * x**2 * jnp.sin(x)**2 * tdi2_factor
        respT =  0.0 * x**2
        
        return jnp.array([respA, respE, respT]).T
    
    def setup_frequency_dependences(self, freqs):
        """
        Set up all the frequency-dependent quantities for the TDI channels selected.
        
        Parameters:
        freqs (array): Array of frequencies.
        """
        self.isotropicresponse = self.get_isotropicresponse(freqs)[jnp.newaxis, :, :]
        self.set_GBresponse(freqs, analytical=True)
        self.GBresponse = self.GBresponse[jnp.newaxis, :, :]

        self.conversion = freqs

    def TDI_background(self, freqs, args):
        """
        Compute the Time-Delay Interferometry (TDI) background power spectral density (PSD).
        Parameters:
        -----------
        freqs : array-like
            Array of frequency values at which to compute the TDI background.
        args : list
            List of arguments for each background function in `self.backgrounds_fn`.
        Returns:
        --------
        Sh : array-like
            The computed power spectral density (PSD) of the TDI background, adjusted by the isotropic response.
        Notes:
        ------
        - `self.backgrounds_fn` is expected to be a list of functions that take `freqs` and an argument from `args` and return an array.
        - `self.convert_to_psd` is a method that converts the computed `h2omega` to a power spectral density.
        - `self.set_isotropicresponse` is a method that sets the isotropic response for the given frequencies.
        - The final PSD is scaled by the isotropic response before being returned.
        """
        
        h2omega = jnp.zeros(shape = (1, freqs.shape[0], 1))

        for i, back in enumerate(self.backgrounds_fn):
            h2omega += back(freqs, args[i])#[:, :, None]

        Sh = self.convert_to_psd(freqs, h2omega)

        self.set_isotropicresponse(freqs)

        return Sh * self.isotropicresponse



class EnergyDensity(ABC, GPUobject):
    """
    EnergyDensity is an parent abstract base class that represents the energy density of a stochastic background.
    It inherits from ABC and GPUobject.
    Attributes:
        use_gpu (bool): Indicates whether to use GPU for computations.
        interpkwargs (dict): Keyword arguments for interpolation.
    Methods:
        ndim: Abstract property that should return the number of dimensions.
        check_ndim(args): Abstract method to check the dimensions of the input arguments.
        __call__(freqs, args): Abstract method to compute the energy density given frequencies and other arguments.

    All the methods in this class are abstract and should be implemented in the derived classes.
    """
    
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

    @property
    def true_params(self):
        return self._true_params
    
    @true_params.setter
    def true_params(self, true_params):
        self._true_params = jnp.atleast_2d(true_params)

    def injected_signal(self, freqs):
        """
        Compute the injected signal for the energy density.
        Parameters:
        freqs (array): Array of frequencies.
        Returns:
        array: Array of injected signal values.
        """
        return self(freqs, self.true_params)

    

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

        # A = jnp.array(args[:, 0:1])#[:, jnp.newaxis]
        # n = jnp.array(args[:, 1:2])#[:, jnp.newaxis]

        A = args[:, 0:1]#[:, jnp.newaxis]
        n = args[:, 1:2]#[:, jnp.newaxis]

        freqs = jnp.atleast_2d(freqs)

        h2omega = self.h2omega(freqs, A, n)

        return h2omega
    
    @partial(jax.jit, static_argnums=(0,))
    def h2omega(self, freqs, A, n):
        return (A * (freqs / self.fknee)**n)[:, :, jnp.newaxis]
        



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
        return h2omega[:, :, jnp.newaxis]
    
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

        return h2omega[:, :, jnp.newaxis]

    def __call__(self, freqs, args):
        #self.check_ndim(args)

        Asw = args[:, 0:1]
        fsw = args[:, 1:2]

        h2omega_sw = self.h2omega_sw(freqs, Asw, fsw)

        if self.turb:
            Aturb = args[:, 2:3]
            Tstar = args[:, 3:4]

            h2omega_turb = self.h2omega_turb(freqs=freqs, Aturb=Aturb, fsw=fsw, Tstar=Tstar)

            return h2omega_sw + h2omega_turb

        return h2omega_sw


class HyperbolicTangent(EnergyDensity):
    '''
    Model from https://arxiv.org/pdf/2405.04690
    '''
    def __init__(self, use_gpu=False, fit_exp=False):

        EnergyDensity.__init__(self, use_gpu=use_gpu)

        self.fit_exp = fit_exp
        self._ndim = self.ndim()

    def ndim(self):
        return 6 if self.fit_exp else 5
    
    def check_ndim(self, args):
        assert args.shape[-1] == self._ndim, args.shape
    
    def __call__(self, freqs, args):

        A = args[:, 0:1]
        s1 = args[:, 1:2]
        alpha = args[:, 2:3]
        fknee = args[:, 3:4]
        s2 = args[:, 4:5]

        exp = args[:, 5:6] if self.fit_exp else (-7.0 / 3.0)

        freqs = jnp.atleast_2d(freqs)

        Sgal = self.Sgal(freqs, A, s1, alpha, fknee, s2, exp)

        return Sgal
    
    @partial(jax.jit, static_argnums=(0,))
    def Sgal(self, freqs, A, s1, alpha, fknee, s2, exp):
        #breakpoint()
        Sgal = (
            0.5
            * A
            * jnp.exp(-(freqs*s1)**alpha)
            * (freqs ** exp)
            * (1.0 + jnp.tanh(-(freqs - fknee) * s2))
        )[:, :, jnp.newaxis]

       
        #same units of the cosmological backgrounds
        #h2omega = Sgal / (3 * H0h**2 / (2 * jnp.pi * freqs**3)) 

        return Sgal
    

class GaussianBumpHyperbolicTangent(HyperbolicTangent):
    '''
    Model from https://arxiv.org/pdf/2405.04690, with the addition of one (atm) Gaussian bump.
    '''
    def __init__(self, use_gpu=False, fit_exp=False):

        HyperbolicTangent.__init__(self, use_gpu=use_gpu, fit_exp=fit_exp)

        self._ndim = self.ndim()

    def ndim(self):
        return super(self).ndim() + 3
    
    def check_ndim(self, args):
        assert args.shape[-1] == self._ndim, args.shape
    
    def __call__(self, freqs, args):

        A = args[:, 0:1]
        s1 = args[:, 1:2]
        alpha = args[:, 2:3]
        fknee = args[:, 3:4]
        s2 = args[:, 4:5]
    
        A_bump = args[:, 5:6]
        f_bump = args[:, 6:7]
        sigma_bump = args[:, 7:8]

        exp = args[:, 8:9] if self.fit_exp else (-7.0 / 3.0)

        freqs = jnp.atleast_2d(freqs)

        Sgal = self.Sgal(freqs, A, s1, alpha, fknee, s2, exp)
        bump = self.gaussian_bump(freqs, A_bump, f_bump, sigma_bump)

        return Sgal + bump
    
       
    @partial(jax.jit, static_argnums=(0,))
    def gaussian_bump(self, freq, A, f_center, width):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        return (A * jnp.exp(-((freq - f_center)**2) / (2 * width**2)))[:, :, jnp.newaxis]
    
    @partial(jax.jit, static_argnums=(0,))
    def gaussian_bump_sum(self, freq, A, f_center, width):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        return jnp.sum(self.gaussian_bump(freq, A, f_center, width), axis=1)
    
    @partial(jax.jit, static_argnums=(0,))
    def gaussian_bump_rj(self, freq, args, groups):
        '''
        TODO change the function structure if we want to use RJ here
        '''
        A = jnp.array(args[:, 0])[:, jnp.newaxis]
        f_center = jnp.array(args[:, 1])[:, jnp.newaxis]
        width = jnp.array(args[:, 2])[:, jnp.newaxis]

        group_unique, group_index, group_inverse, group_count = jnp.unique(groups, return_index=True, return_counts=True, return_inverse=True)

        A_full = jnp.zeros((len(group_unique), max(group_count), freq.shape[0]))
        f_center_full = jnp.zeros((len(group_unique), max(group_count), freq.shape[0]))
        width_full = jnp.ones((len(group_unique), max(group_count), freq.shape[0]))

        for i, group in enumerate(group_unique):
            idxs = jnp.where(group_inverse == i)[0]
            A_full[i, :len(idxs)] = A[idxs]
            f_center_full[i, :len(idxs)] = f_center[idxs]
            width_full[i, :len(idxs)] = width[idxs]

        bump = self.gaussian_bump_sum(freq, A_full, f_center_full, width_full)
        
        return bump
    
