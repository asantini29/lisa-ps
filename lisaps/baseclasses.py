from abc import ABC
from typing import Callable
import numpy as np
import pickle


try:
    import cupy as xp
    from cupyx.scipy.interpolate import make_interp_spline as cupy_make_interp_spline
    from cupyx.scipy.interpolate import Akima1DInterpolator as cupy_Akima1DInterpolator

except (ModuleNotFoundError, ImportError):
    import numpy as xp

from scipy.interpolate import make_interp_spline as scipy_make_interp_spline
from scipy.interpolate import Akima1DInterpolator as scipy_Akima1DInterpolator
from scipy import signal

from .constants import *
from cudakima import AkimaInterpolant1D


import jax
import jax.numpy as jnp
from functools import partial
import warnings
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class GPUobject:
    '''
    Represents an object that can utilize a GPU for computations.

    Attributes:
        use_gpu (bool): Flag indicating whether to use GPU for computations.
        xp: The library to use for computations (numpy or cupy).
        interpkwargs (dict): Dictionary containing the arguments for the spine interpolant.

    Methods:
        __init__(self, use_gpu=False, interpkwargs=None): Initializes a GPUobject instance.
        gpu_capable(self): Returns True if the object is capable of using a GPU.
        adjust_interpolant(self, interpkwargs=None): Adjusts the interpolant to use for computations.
        __getstate__(self): Controls what gets pickled. Excludes GPU-specific interpolator functions or other unpicklable objects.
        __setstate__(self, state): Controls how the object is restored. Reinitializes interp based on interpkwargs.
        save(self, filename): Saves the object to a file.
    '''

    def __init__(self, use_gpu=False, interpkwargs=None):
        self.use_gpu = use_gpu
        self.xp = xp if use_gpu else np
        self.interpkwargs = None
        
        if interpkwargs is not None:
            self.adjust_interpolant(interpkwargs)

    @property
    def gpu_capable(self):
        return True
    
    def adjust_interpolant(self, interpkwargs=None):
        '''
        Adjusts the interpolant to use for computations.

        Args:
            interpkwargs (dict): Dictionary containing the arguments for the spine interpolant.

        Raises:
            AssertionError: If interpkwargs is not a dictionary.

        '''
        if interpkwargs is None:
            self.interpkwargs = dict(
                kind='akima',
                axis=1
            )
        else:
            assert isinstance(interpkwargs, dict), 'interpkwargs must be a dictionary containing the argument of the spine interpolant'

            self.interpkwargs = interpkwargs

        try:
            kind = self.interpkwargs.pop('kind')
        except KeyError:
            kind = 'akima'


        if kind == 'bsplines':
            if self.use_gpu:    
                self.interp = cupy_make_interp_spline
            else:
                self.interp = scipy_make_interp_spline

            if 'bc_type' not in self.interpkwargs.keys():
                self.interpkwargs['bc_type'] = 'natural'
                
            if 'k' not in self.interpkwargs.keys():
                self.interpkwargs['k'] = 3

        elif kind == 'akima':

            if ('use_numba' not in self.interpkwargs.keys()) or (self.interpkwargs['use_numba'] is False):
                if self.use_gpu:    
                    self.interp = cupy_Akima1DInterpolator
                else:
                    self.interp = scipy_Akima1DInterpolator
            
            else:
                threadsperblock = self.interpkwargs['threadsperblock'] if 'threadsperblock' in self.interpkwargs.keys() else 64
                
                self.interp = AkimaInterpolant1D(use_gpu=self.use_gpu, threadsperblock=threadsperblock)

        else:
            raise NotImplementedError

    def __getstate__(self):
        """
        Controls what gets pickled. Excludes GPU-specific interpolator functions or
        other unpicklable objects.
        """
        state = self.__dict__.copy()
        # Remove the interp function from the state, as it may not be picklable
        state['interp'] = None
        del state['xp']
        return state

    def __setstate__(self, state):
        """
        Controls how the object is restored. Reinitializes interp based on interpkwargs.
        """
        self.__dict__.update(state)
        # Reinitialize the interpolator if needed
        if self.interpkwargs is not None:
            self.adjust_interpolant(self.interpkwargs) 
        self.xp = xp if self.use_gpu else np

    def save(self, filename):
        """
        Saves the object to a file.

        Args:
            filename (str): The name of the file to save the object to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

class BaseNoise(GPUobject):
    """
    BaseNoise class for modeling noise in LISA.
    Attributes:
        asdTM (float): Amplitude spectral density for test mass noise.
        asdOMS (float): Amplitude spectral density for OMS noise.
        fkneeTM (float): Knee frequency for test mass noise.
        fkneeOMS (float): Knee frequency for OMS noise.
        armlength (float): Length of the detector arms.
        fs (float): Sampling frequency.
        FMIN (float): Minimum frequency.
        scirdv1 (bool): Flag indicating whether to use the scirdv1 configuration.
        available_channels (list): List of available channels.
        available_functions (dict): Dictionary mapping channels to their corresponding functions.
        channels (list): List of selected channels.
        Ncov (int): Number of channels to consider.
        TDIsetup (str): TDI setup configuration.
        units (str): Units for the noise ('hertz', 'meters', or 'strain').
    """

    def __init__(self, asdTM=2.4e-15, asdOMS=7.9e-12, fkneeTM=0.4e-3, fkneeOMS=2e-3, equal_arms=False, custom_armlength=None, T=1.0, fs=None, Ncov=None, channels='AET', use_gpu=False, units='strain', scirdv1=False, interpkwargs=dict(kind='akima', axis=1)):
        """
        Initialize the base class for LISA simulation.
        Parameters:
        -----------
        asdTM : float, optional
            Amplitude spectral density for test mass noise (default is 2.4e-15).
        asdOMS : float, optional
            Amplitude spectral density for optical metrology system noise (default is 7.9e-12).
        fkneeTM : float, optional
            Knee frequency for test mass noise (default is 0.4e-3).
        fkneeOMS : float, optional
            Knee frequency for optical metrology system noise (default is 2e-3).
        equal_arms : bool, optional
            If True, use equal arm lengths (default is False).
        custom_armlength : float, optional
            Custom arm length to use (default is None).
        T : float, optional
            Observation time in years (default is 1.0).
        fs : float, optional
            Sampling frequency (default is None).
        Ncov : int, optional
            Number of channels to consider (default is None).
        channels : str or list, optional
            Channels to use, either 'AET', 'XYZ', or a list of specific channels (default is 'AET').
        use_gpu : bool, optional
            If True, use GPU acceleration (default is False).
        units : str, optional
            Units for the noise, either 'strain' or other (default is 'strain').
        scirdv1 : bool, optional
            If True, use SCIR DV1 noise parameters (default is False).
        interpkwargs : dict, optional
            Keyword arguments for interpolation (default is dict(kind='akima', axis=1)).
        Raises:
        -------
        ValueError
            If neither the number nor the name of channels to consider is provided.
        AssertionError
            If the provided channels or Ncov values are invalid.
        """
        
        
        GPUobject.__init__(self, use_gpu=use_gpu, interpkwargs=interpkwargs)
        if custom_armlength is not None:
            self.armlength = custom_armlength
            print('Using custom armlength: {}'.format(custom_armlength))
        else:
            self.armlength = ARMLENGTH_EQUAL if equal_arms else ARMLENGTH_AVERAGE
            print('Using default armlength: {}'.format(self.armlength))
        self.fs = fs if fs is not None else FS
        self.FMIN = 1 / (T * YRSID_SI)
        self.scirdv1 = scirdv1

        self.setup_noise()


        if self.scirdv1:
            asdTM = 3.0e-15
            asdOMS = 15.0e-12
            fkneeTM = 4e-4
            fkneeOMS = 2e-3

        self.asdTM = asdTM
        self.asdOMS = asdOMS
        self.fkneeTM = fkneeTM
        self.fkneeOMS = fkneeOMS
        
        #channel selection
        self.available_channels = ['AA', 'EE', 'TT', 'XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
        available_functions = [self.get_SA, self.get_SA, self.get_ST, self.get_SXX, self.get_SXX, self.get_SXX, self.get_SXY, self.get_SXY, self.get_SXY]

        self.available_functions = dict(zip(self.available_channels, available_functions))

        if channels is None:
            if Ncov is None:
                raise ValueError('Provide either the number or the name of channels to consider')
        
            else:
                print('Defaulting to AET configuration')
                self.Ncov = Ncov
                self.channels = self.available_channels[:self.Ncov]
                self.TDIsetup = 'AET'

        else:
            if isinstance(channels, str):
                assert channels in ['AET', 'XYZ'] + self.available_channels, 'Provide either the TDI setup (AET / XYZ) or a list of channels'
                if channels == 'AET':
                    assert Ncov in [None, 3], 'Ncov is either not provided or set equal to 3'
                    self.Ncov = 3
                    self.channels = self.available_channels[:self.Ncov]

                    self.TDIsetup = 'AET'

                elif channels == 'XYZ':
                    assert Ncov in [None, 6], 'Ncov is either not provided or set equal to 6'
                    self.Ncov = 6
                    self.channels = self.available_channels[-self.Ncov:]

                    self.TDIsetup = 'XYZ'
                
                #* single channels can be selected only with AET configuration
                else: 
                    channels = [channels]
                    self.TDIsetup = 'AET'
            
            if isinstance(channels, list):
                self.channels = channels
                if Ncov is not None:
                    assert len(channels) == Ncov
                self.Ncov = len(channels)

                self.TDIsetup = 'AET'

        self.units = units

    @property
    def asdTM(self):
        return self._asdTM
    
    @asdTM.setter
    def asdTM(self, asdTM=2.4e-15):
        self._asdTM = asdTM

    @property
    def asdOMS(self):
        return self._asdOMS
    
    @asdOMS.setter
    def asdOMS(self, asdOMS=7.9e-12):
        self._asdOMS = asdOMS

    def oms_in_isi_carrier(self, asdOMS, freqs):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.
        
        Args:
            asdOMS (float, ndarray): The ASD (Amplitude Spectral Density) for the OMS (Optical Metrology System) noise.
            freqs (float): frequencies [Hz]
        """
        asd = jnp.atleast_1d(asdOMS)
        psd_meters = asd**2 * jnp.atleast_1d(1 + (self.fkneeOMS / freqs)**4)
        psd_hertz = jnp.atleast_1d(2 * jnp.pi * freqs * CENTRAL_FREQ / C)**2 * psd_meters

        if self.units == 'hertz':
            return jnp.sqrt(psd_hertz)
        
        elif self.units == 'meters':
            return jnp.atleast_1d(jnp.sqrt(psd_meters))
        
        elif self.units == 'strain':
            return jnp.sqrt(psd_hertz / CENTRAL_FREQ**2)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')

    def filtered_oms_in_isi_carrier(self, asdOMS, freqs):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.

        Include transfer function of derivative filter instead of perfect 2 pi f
        
        Args:
            asdOMS (float, ndarray): The ASD (Amplitude Spectral Density) for the OMS (Optical Metrology System) noise.
            freqs (float): frequencies [Hz]
        """
        asd = asdOMS 
        psd_highfreq = (asd * self.fs * CENTRAL_FREQ / C) ** 2 * jnp.sin(
            2 * jnp.pi * freqs / self.fs
        ) ** 2
        psd_lowfreq = (#jnp.atleast_2d(
            (2 * jnp.pi * asd * CENTRAL_FREQ * self.fkneeOMS**2 / C) ** 2
            * jnp.abs(
                (2 * jnp.pi * self.FMIN)
                / (
                    1
                    - jnp.exp(-2 * jnp.pi * self.FMIN / self.fs)
                    * jnp.exp(-2j * jnp.pi * freqs / self.fs)
                )
            ) ** 2
            * 1 / (self.fs * self.FMIN) ** 2
        )
        psd_hertz = psd_highfreq + psd_lowfreq

        if self.units == 'hertz':
            return jnp.sqrt(psd_hertz) 
        
        elif self.units == 'meters':
            psd_meters = psd_hertz / jnp.atleast_2d(2 * jnp.pi * freqs * CENTRAL_FREQ / C)**2
            return jnp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = psd_hertz / CENTRAL_FREQ**2
            return jnp.sqrt(psd_strain)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')


    def testmass_single(self, asdTM, freqs):
        """
        Model for single TM noise PSD including filters used in the data generation
        
        Args:
            asdTM (float, ndarray): The ASD (Amplitude Spectral Density) for the TM (Test Mass) noise.
            freqs (float): frequencies [Hz]
        """
        asd = jnp.atleast_2d(asdTM) # m / s^2 / sqrt(Hz)
        psd_acc = asd**2 * jnp.atleast_2d(1 + (self.fkneeTM / freqs)**2)  # m^2 / s^4 / Hz

        if self.units == 'hertz':
            psd_hertz = jnp.atleast_2d(CENTRAL_FREQ / (2 * jnp.pi * C * freqs))**2 * psd_acc
            return jnp.sqrt(psd_hertz)

        elif self.units == 'meters':
            psd_meters = jnp.atleast_2d(2 * jnp.pi * freqs)**(-4) * psd_acc
            return jnp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = jnp.atleast_2d(1 / (2 * jnp.pi * C * freqs))**2 * psd_acc
            return jnp.sqrt(psd_strain)


        else:
            raise ValueError('units must be `hertz`, `meters` or `strain`')
        
    def testmass_scirdv1(self, asdTM, freqs):
        """
        Model for single TM noise PSD in the scirdv1 configuration.
        
        Args:
            asdTM (float, ndarray): The ASD (Amplitude Spectral Density) for the TM (Test Mass) noise.
            freqs (float): frequencies [Hz]
        """
        asd = jnp.atleast_1d(asdTM) # m / s^2 / sqrt(Hz)
        psd_acc = asd**2 * jnp.atleast_1d( (1 + (self.fkneeTM / freqs)**2) * (1.0 + (freqs / 8e-3) ** 4)) # m^2 / s^4 / Hz

        if self.units == 'hertz':
            psd_hertz = jnp.atleast_1d(CENTRAL_FREQ / (2 * jnp.pi * C * freqs))**2 * psd_acc
            return jnp.sqrt(psd_hertz)

        elif self.units == 'meters':
            psd_meters = jnp.atleast_1d(2 * jnp.pi * freqs)**(-4) * psd_acc
            return jnp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = jnp.atleast_1d(1 / (2 * jnp.pi * C * freqs))**2 * psd_acc
            return jnp.sqrt(psd_strain)

        else:
            raise ValueError('units must be `hertz`, `meters` or `strain`')
        
    def filtered_testmass_single(self, asdTM, freqs):
        """
        Model for single TM noise PSD
        
        Args:
            asdTM (float, ndarray): The ASD (Amplitude Spectral Density) for the TM (Test Mass) noise.
            freqs (float): frequencies [Hz]
        """
        #asd = jnp.atleast_2d(asdTM)
        asd = asdTM
        psd_highfreq = (#jnp.atleast_2d(
            (asd * CENTRAL_FREQ / (2 * jnp.pi * C)) ** 2
            * jnp.abs(
                (2 * jnp.pi * FMIN)
                / (
                    1
                    - jnp.exp(-2 * jnp.pi * FMIN / self.fs)
                    * jnp.exp(-2j * jnp.pi * freqs / self.fs)
                )
            )
            ** 2
            * 1
            / (self.fs * FMIN) ** 2
        )
        psd_lowfreq = ((asd * CENTRAL_FREQ * self.fkneeTM / (2 * jnp.pi * C)) ** 2
            * jnp.abs(
                (2 * jnp.pi * FMIN)
                / (
                    1
                    - jnp.exp(-2 * jnp.pi * FMIN / self.fs)
                    * jnp.exp(-2j * jnp.pi * freqs / self.fs)
                )
            )
            ** 2
            * 1
            / (self.fs * FMIN) ** 2
            * jnp.abs(1 / (1 - jnp.exp(-2j * jnp.pi * freqs / self.fs))) ** 2
            * (2 * jnp.pi / self.fs) ** 2
        )
        psd_hertz = psd_lowfreq + psd_highfreq

        if self.units == 'hertz':
            return jnp.sqrt(psd_hertz) 
        
        elif self.units == 'meters':
            psd_meters = psd_hertz / jnp.atleast_2d(2 * jnp.pi * freqs * CENTRAL_FREQ / C)**2
            return jnp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = psd_hertz / CENTRAL_FREQ**2
            return jnp.sqrt(psd_strain)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')
        
    def tdi_common(self, freqs):
        '''
        TDI common factor for both XYZ and AET

        Args:
            freqs (float): frequencies [Hz]
        '''
        return 16 * jnp.sin(2 * jnp.pi * freqs * self.armlength) * jnp.sin(4 * jnp.pi * freqs * self.armlength)**2
    
    def tdi_common_AET(self, freqs):
        return 2 * self.tdi_common(freqs) * jnp.sin(2 * jnp.pi * freqs * self.armlength)
    
    def tdi_tf_oms_A(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI A,E.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = self.tdi_common_AET(freqs) * (2 + jnp.cos(2 * xp.pi * freqs * self.armlength))
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
        
    def tdi_tf_oms_T(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI T.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = 2 * self.tdi_common_AET(freqs) * (1 - jnp.cos(2 * jnp.pi * freqs * self.armlength))
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_testmass_A(self, freqs):
        """
        TDI transfer function for testmass noise in TDI A,E.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = 4 * self.tdi_common_AET(freqs) * (1 + jnp.cos(2 * xp.pi * freqs * self.armlength) + jnp.cos(2 * xp.pi * freqs * self.armlength)**2 )
                            
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
        
    def tdi_tf_testmass_T(self, freqs):
        """
        TDI transfer function for testmass noise in TDI T.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = 4 * self.tdi_common_AET(freqs) * (1 - jnp.cos(2 * jnp.pi * freqs * self.armlength))**2
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_oms_XX(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI XX, YY, ZZ.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = 4 * self.tdi_common(freqs) * jnp.sin(2 * jnp.pi * freqs * self.armlength)
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_oms_XY2(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI XY, XZ, YZ.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = - self.tdi_common(freqs) * jnp.sin(4 * jnp.pi * freqs * self.armlength)
        #return jnp.atleast_2d(psd)
        return psd
    
    def tdi_tf_testmass_XX(self, freqs):
        """
        TDI transfer function for testmass noise in TDI XX, YY, ZZ.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = 4 * self.tdi_common(freqs) * jnp.sin(2 * jnp.pi * freqs * self.armlength) * (3 + jnp.cos(4 * jnp.pi * freqs * self.armlength))
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_testmass_XY2(self, freqs):
        """
        TDI transfer function for testmass noise in TDI XY, XZ, YZ.
        
        Args:
            freqs (float): frequencies [Hz]
        """
        psd = - 4 * self.tdi_common(freqs) * jnp.sin(4 * jnp.pi * freqs * self.armlength)
        #return jnp.atleast_2d(psd)
        return psd

    #! AET
    def testmass_A(self, asdTM, freqs):
        """
        Calculate the test mass response for channel A.
        This method computes the test mass response for channel A by multiplying 
        the transfer function of the test mass for channel A with the filtered 
        test mass single response.
        Parameters:
        asdTM (array-like): Amplitude spectral density of the test mass.
        freqs (array-like): Frequencies at which the response is calculated.
        Returns:
        array-like: The test mass response for channel A.
        """

        return self.tdi_tf_testmass_A(freqs) * self.filtered_testmass_single(asdTM, freqs)

    def testmass_T(self, asdTM, freqs):
        """
        Calculate the test mass response for channel T.
        This method computes the test mass response for channel T by multiplying 
        the transfer function of the test mass for channel T with the filtered 
        test mass single response.
        Parameters:
        asdTM (array-like): Amplitude spectral density of the test mass.
        freqs (array-like): Frequencies at which the response is calculated.
        Returns:
        array-like: The test mass response for channel T.
        """
        
        return self.tdi_tf_testmass_T(freqs) * self.filtered_testmass_single(asdTM, freqs)

    def oms_A(self, asdOMS, freqs):
        """
        Calculate the optical metrology system response for channel A.
        This method computes the optical metrology system response for channel A by multiplying
        the transfer function of the optical metrology system for channel A with the filtered
        optical metrology system single response.
        Parameters:
        asdOMS (array-like): Amplitude spectral density of the optical metrology system.
        freqs (array-like): Frequencies at which the response is calculated.
        Returns:
        array-like: The optical metrology system response for channel A.
        """
        return self.tdi_tf_oms_A(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)

    def oms_T(self, asdOMS, freqs):
        """
        Calculate the optical metrology system response for channel T.
        This method computes the optical metrology system response for channel T by multiplying
        the transfer function of the optical metrology system for channel T with the filtered
        optical metrology system single response.
        Parameters:
        asdOMS (array-like): Amplitude spectral density of the optical metrology system.
        freqs (array-like): Frequencies at which the response is calculated.
        Returns:
        array-like: The optical metrology system response for channel T.
        """
        return self.tdi_tf_oms_T(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)

    @partial(jax.jit, static_argnums=(0,))
    def get_SA_base(self, asdTM, asdOMS, freqs):
        """
        Calculate the sum of the squares of the test mass and optical metrology system acceleration noise.
        Parameters:
        asdTM (array-like): Amplitude spectral density of the test mass acceleration noise.
        asdOMS (array-like): Amplitude spectral density of the optical metrology system acceleration noise.
        freqs (array-like): Frequencies at which the noise is evaluated.
        Returns:
        array-like: Sum of the squares of the test mass and optical metrology system acceleration noise.
        """
        
        return self.testmass_A(asdTM, freqs)**2 + self.oms_A(asdOMS, freqs)**2 
    
    @partial(jax.jit, static_argnums=(0,))
    def get_ST_base(self, asdTM, asdOMS, freqs):
        """
        Calculate the sum of the squares of the test mass and optical metrology system acceleration noise.
        Parameters:
        asdTM (array-like): Amplitude spectral density of the test mass acceleration noise.    
        asdOMS (array-like): Amplitude spectral density of the optical metrology system acceleration noise.
        freqs (array-like): Frequencies at which the noise is evaluated.
        Returns:
        array-like: Sum of the squares of the test mass and optical metrology system acceleration noise.
        """

        return self.testmass_T(asdTM, freqs)**2 + self.oms_T(asdOMS, freqs)**2 
    
    @partial(jax.jit, static_argnums=(0,))
    def get_SA_scirdv1(self, asdTM, asdOMS, freqs):
        '''
        Uncorrelated noise in the A, E tdi channels. Code from: https://mikekatz04.github.io/LISAanalysistools/build/html/index.html#
        '''

        x = 2.0 * np.pi * self.armlength * freqs

        Spm = self.testmass_scirdv1(asdTM, freqs) ** 2
        Sop = self.oms_in_isi_carrier(asdOMS, freqs) ** 2

        Sa = (
            8.0
            * jnp.sin(x) ** 2
            * (
                2.0 * Spm * (3.0 + 2.0 * jnp.cos(x) + jnp.cos(2 * x))
                + Sop * (2.0 + jnp.cos(x))
            )
        )

        return Sa
        
    
    @partial(jax.jit, static_argnums=(0,))
    def get_ST_scirdv1(self, asdTM, asdOMS, freqs):
        '''
        Uncorrelated noise in the T tdi channel. Code from: https://mikekatz04.github.io/LISAanalysistools/build/html/index.html#
        '''
        x = 2.0 * np.pi * self.armlength * freqs

        Spm = self.testmass_scirdv1(asdTM, freqs) ** 2
        Sop = self.oms_in_isi_carrier(asdOMS, freqs) ** 2

        return (
            16.0 * Sop * (1.0 - jnp.cos(x)) * jnp.sin(x) ** 2
            + 128.0 * Spm * jnp.sin(x) ** 2 * jnp.sin(0.5 * x) ** 4
        )
    
    #! XYZ
    def testmass_XX(self, asdTM, freqs):
        """
        Calculate the test mass XX response.
        This method computes the test mass XX response by multiplying the 
        transfer function of the test mass XX with the filtered test mass 
        single response.
        Parameters:
        asdTM : array-like
            The amplitude spectral density of the test mass.
        freqs : array-like
            The frequency values at which the response is calculated.
        Returns:
        array-like
            The calculated test mass XX response.
        """
        
        return self.tdi_tf_testmass_XX(freqs) * self.filtered_testmass_single(asdTM, freqs)

    def testmass_XY(self,asdTM, freqs):
        """
        Calculate the test mass XY response.
        This function computes the test mass XY response by multiplying the 
        transfer function of the test mass XY with the square of the filtered 
        test mass single.
        Parameters:
        asdTM (array-like): Amplitude spectral density of the test mass.
        freqs (array-like): Frequencies at which the response is calculated.
        Returns:
        array-like: The test mass XY response.
        """

        return self.tdi_tf_testmass_XY2(freqs) * self.filtered_testmass_single(asdTM, freqs)**2

    def oms_XX(self, asdOMS, freqs):
        """
        Calculate the output of the oms_XX function.
        This function computes the product of the transfer function of the 
        time delay interferometry (TDI) for the oms_XX channel and the 
        filtered optical metrology system (OMS) input in the inter-satellite 
        interferometer (ISI) carrier.
        Parameters:
        asdOMS (array-like): The amplitude spectral density (ASD) of the OMS.
        freqs (array-like): The frequency array.
        Returns:
        array-like: The result of the oms_XX calculation.
        """

        return self.tdi_tf_oms_XX(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)

    def oms_XY(self, asdOMS, freqs):
        """
        Calculate the OMS XY component.
        This method computes the OMS XY component by multiplying the result of 
        `tdi_tf_oms_XY2` with the square of the result from `filtered_oms_in_isi_carrier`.
        Args:
            asdOMS (array-like): The amplitude spectral density of the OMS.
            freqs (array-like): The frequencies at which to evaluate the OMS XY component.
        Returns:
            array-like: The computed OMS XY component.
        """

        return self.tdi_tf_oms_XY2(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)**2

    @partial(jax.jit, static_argnums=(0,))
    def get_SXX(self, asdTM, asdOMS, freqs):
        """
        Calculate the SXX power spectral density.
        This function computes the SXX power spectral density by summing the 
        squared values of the test mass and optical metrology system (OMS) 
        contributions.
        Parameters:
        asdTM (array-like): Amplitude spectral density of the test mass.
        asdOMS (array-like): Amplitude spectral density of the optical metrology system.
        freqs (array-like): Frequencies at which the spectral densities are evaluated.
        Returns:
        array-like: The SXX power spectral density.
        """
        
        return self.testmass_XX(asdTM, freqs)**2 + self.oms_XX(asdOMS, freqs)**2 

    @partial(jax.jit, static_argnums=(0,))
    def get_SXY(self, asdTM, asdOMS, freqs):
        """
        Calculate the combined SXY value from test mass and optical metrology system (OMS) data.
        Parameters:
        asdTM (array-like): Amplitude spectral density data for the test mass.
        asdOMS (array-like): Amplitude spectral density data for the optical metrology system.
        freqs (array-like): Frequency values corresponding to the ASD data.
        Returns:
        array-like: Combined SXY value calculated from the test mass and OMS data.
        """
        
        return (self.testmass_XY(asdTM, freqs) + self.oms_XY(asdOMS, freqs))
    
    def setup_noise(self):
        '''
        Sets up the noise model for the given configuration.
        '''
        if self.scirdv1:
            self.get_SA = self.get_SA_scirdv1
            self.get_ST = self.get_ST_scirdv1
        else:
            self.get_SA = self.get_SA_base
            self.get_ST = self.get_ST_base

    
    @partial(jax.vmap, in_axes=(None, 0, 0, None))
    def compute_PSDS(self, asdTM, asdOMS, freqs):
        """
        Compute the Power Spectral Density (PSD) for each channel.

        Args:
            asdTM (ndarray): The ASD (Amplitude Spectral Density) for the TM (Test Mass) channel.
            asdOMS (ndarray): The ASD for the OMS (Optical Metrology System) channel.
            freqs (ndarray): The frequency values.

        Returns:
            ndarray: The computed PSDs for each channel, transposed to have channels as columns.

        """
        return (jnp.asarray([self.available_functions[channel](asdTM, asdOMS, freqs) for channel in self.channels])).transpose(1,0)
    
    
    def set_PSDS(self, freqs, squeeze=False, out=False, **kwargs):
        """
        Sets the PSDS (Power Spectral Density Sensitivity) for the given frequencies using the stored amplitudes.

        Parameters:
        - freqs (array-like): The frequencies at which to calculate the PSDS.
        - squeeze (bool, optional): Whether to squeeze the output arrays. Defaults to False.
        - out (bool, optional): Whether to return the calculated PSDS. Defaults to False.

        Returns:
        - None: If `out` is False.
        - array-like: The calculated PSDS if `out` is True.
        """
        asdTM, asdOMS = jnp.atleast_1d(self.asdTM), jnp.atleast_1d(self.asdOMS)
        self.PSDS_design = self.get_PSDS(asdTM, asdOMS, freqs, squeeze)
        if out:
            return self.PSDS_design
    
    def get_PSDS(self, asdTM, asdOMS, freqs=None, squeeze=False, **kwargs):
        """
        Compute the Power Spectral Density (PSD) using the given ASDs (Amplitude Spectral Densities).

        Parameters:
        - asdTM: array-like
            The ASD (Amplitude Spectral Density) for the TM (Test Mass) channel.
        - asdOMS: array-like
            The ASD (Amplitude Spectral Density) for the OMS (Optical Metrology System) channel.
        - freqs: array-like, optional
            The frequencies at which to compute the PSD. If not provided, the PSD will be computed at all frequencies.
        - squeeze: bool, default False
            Whether to squeeze the output array if it has a single dimension.

        Returns:
        - PSDS: array-like
            The computed Power Spectral Density (PSD) values.

        """
        asdTM, asdOMS = jnp.atleast_1d(asdTM), jnp.atleast_1d(asdOMS)
        if squeeze:
            return jnp.squeeze(self.compute_PSDS(asdTM, asdOMS, freqs))
        else:
            return self.compute_PSDS(asdTM, asdOMS, freqs)

class TDIresponse(GPUobject):
    """
    Represents the response of a TDI (Time Delay Interferometry) system.
    
    Args:
        filename (str): The path to the file containing TDI response data.
        use_gpu (bool, optional): Flag indicating whether to use GPU acceleration. Default: `False`.
        interpkwargs (dict, optional): Additional keyword arguments for the interpolation function. Defaults to dict(kind='akima', axis=0).
    """

    def __init__(self, filename, use_gpu=False, interpkwargs=dict(kind='akima', axis=0)):
        """
        Initializes a TDIresponse object.
        
        Args:
            filename (st or list): The path to the file containing TDI response data. 
            use_gpu (bool, optional): Flag indicating whether to use GPU acceleration. Default: `False`.
            interpkwargs (dict, optional): Additional keyword arguments for the interpolation function. Default: dict(kind='akima', axis=0).
        """
        GPUobject.__init__(self, use_gpu=use_gpu, interpkwargs=interpkwargs)

        if isinstance(filename, str):
            TDIcsd_real = self.xp.transpose(self.xp.genfromtxt(filename, delimiter=','))
            TDIcsd_imag = self.xp.zeros_like(TDIcsd_real)

        elif isinstance(filename, list):
            TDIcsd_real = self.xp.transpose(self.xp.genfromtxt(filename[0], delimiter=','))
            TDIcsd_imag = self.xp.transpose(self.xp.genfromtxt(filename[1], delimiter=','))
        else:
            raise ValueError('filename must be a string or a list of strings')
        
        self.freqs = self.xp.real(TDIcsd_real[0])

        self.tfs_real = self.xp.stack([self.xp.real(TDIcsd_real[i]) for i in range(1, len(TDIcsd_real))]).T
        self.tfs_imag = self.xp.stack([self.xp.real(TDIcsd_imag[i]) for i in range(1, len(TDIcsd_imag))]).T

        self.TDIinterpolant_real = self.interp(self.freqs, self.tfs_real)
        self.TDIinterpolant_imag = self.interp(self.freqs, self.tfs_imag)

    def __call__(self, freqs, return_jax=True, return_gpu=True):
        """
        Calculates the TDI response at the given frequencies.
        
        Args:
            freqs (array-like): The frequencies at which to calculate the TDI response.
            return_jax (bool, optional): Flag indicating whether to return the response as a JAX array. Default: `True`.
            return_gpu (bool, optional): Flag indicating whether to return the response as a GPU array. Default: `True`.
        
        Returns:
            array-like: The TDI response at the given frequencies.
        """
        response_real = self.TDIinterpolant_real(freqs)
        response_imag = self.TDIinterpolant_imag(freqs)

        response = response_real + 1j * response_imag

        if return_jax:
            return jnp.asarray(response)
        else:
            if return_gpu:
                return response  # return a self.xp array
            else:
                try:
                    return response.get()  # return a np array
                except AttributeError:
                    return response

class DataContainer(GPUobject):
    """
    Store the time or frequency domain data and perform the necessary operations.
    This class is heavily based on the classes contained in the `datacontainer.py` file from the `lisatools` package (https://mikekatz04.github.io/LISAanalysistools/build/html/user/datacontainer.html).
    """
    def __init__(self, 
                t=None,
                d=None,
                freqs=None,
                dtilde=None,
                weights=None,
                nchannels=3,
                fmin=1e-4,
                fmax=2.9e-2,
                average=False,
                f_segments=1e-5,
                fullmatrix=False,
                window=('kaiser', 30),
                use_gpu=False,
                ):
        """
        Initialize the object with provided parameters.
        Parameters:
        t (array-like, optional): Time domain data.
        d (array-like, optional): Data in the time domain.
        freqs (array-like, optional): Frequency domain data.
        dtilde (array-like, optional): Data in the frequency domain.
        weights (array-like, optional): Weights for the data.
        nchannels (int, optional): Number of channels. Default is 3.
        fmin (float, optional): Minimum frequency. Default is 1e-4.
        fmax (float, optional): Maximum frequency. Default is 2.9e-2.
        average (bool, optional): Whether to average the periodogram. Default is False.
        f_segments (float, optional): Frequency segments for averaging. Default is 1e-5.
        fullmatrix (bool, optional): Whether to use the full matrix. Default is False.
        window (tuple, optional): Window function and its parameter. Default is ('kaiser', 30).
        use_gpu (bool, optional): Whether to use GPU for computations. Default is False.
        Raises:
        ValueError: If neither time nor frequency data is provided.
        ValueError: If neither time domain nor frequency domain data is provided.
        AssertionError: If dimensionality of data does not match the expected shape.
        """

        GPUobject.__init__(self, use_gpu=use_gpu)

        self.nchannels = nchannels
        self.fullmatrix = fullmatrix
        self.average = average

        if (t is None) and (freqs is None):
            raise ValueError('Provide either the times or frequencies')

        if (d is None) and (dtilde is None):
            raise ValueError('Provide the data in either the time or frequency domain')
        
        if (t is None) and (freqs is not None) and (d is None) and (dtilde is not None):
            self.domain = 'frequency'
            assert (dtilde.shape[0] == freqs.shape[0]) and (dtilde.shape[1] == self.nchannels), 'Dimensionality mismatch'
            self.df = np.concatenate(([freqs[1] - freqs[0]], np.diff(freqs)))[:, None]

            self.window, self.Nbw = self.get_window(None, 2 * dtilde.shape[0])

            freqs = jnp.real(freqs)
            d_tmp = dtilde
            self.d = None
        
        if (t is not None) and (freqs is None) and (d is not None) and (dtilde is None):
            self.domain = 'time'
            assert (d.shape[0] == t.shape[0]) and (d.shape[1] == self.nchannels), 'Dimensionality mismatch'
            self.d = d
            self.dt = t[1] - t[0]

            self.window, self.Nbw = self.get_window(window, d.shape[0])
            
            freqs = np.fft.rfftfreq(d.shape[0], self.dt)

            d_tmp = d

        if fmin is not None and fmax is not None:
            self.fmin = fmin
            self.fmax = fmax

        else:
            self.fmin = freqs.min()
            self.fmax = freqs.max()


        self.dtilde = self.get_Xtilde(d_tmp)   
        P = self.periodogram_matrix(self.dtilde)

        if average: # without signal we can use an averaged likelihood

            freqs, P, sizes = self.average_periodogram(P, freqs, f_segments)

            self.frequencymask = (freqs > self.fmin) & (freqs < self.fmax) # here to remove the zeros of the transfer functions. #todo: work on a way not to waste all these data
            self.freqs = jnp.real(jnp.array(freqs[self.frequencymask]))

            if hasattr(self, 'df'):
                self.df = np.concatenate(([freqs[1] - freqs[0]], np.diff(freqs)))[self.frequencymask, None]

            self.periodgram = P[self.frequencymask]
            self.sizes = sizes[self.frequencymask]

            self.nu = self.sizes / self.Nbw

            p = min(self.nchannels, 3)

            if self.fullmatrix:
                self.Y = self.nu[None, :, None, None] * self.periodgram[None, :, :, :]
            else:
                self.Y = self.nu[None, :, None] * self.periodgram[None, :, :]

            self.dtildedtilde = None
            
            self.averaged = True

        else:
            self.frequencymask = (freqs > self.fmin) & (freqs < self.fmax)
            self.freqs = freqs[self.frequencymask]
            if hasattr(self, 'df'): 
                self.df = self.df[self.frequencymask]
            self.dtildedtilde =self.get_XtildeXtilde()[:, self.frequencymask, :]
            self.dtilde = self.dtilde[self.frequencymask, :]
            self.periodgram = P[self.frequencymask]
            self.nu = 1.0

            self.averaged = False

        if weights is not None:
            self.weights = jnp.asarray(weights[self.frequencymask])[None, :, :]
        else:
            self.weights = jnp.ones_like(self.dtilde)[None, :, :]

    def get_Xtilde(self, d=None):
        """
        Compute the normalized Xtilde based on the domain (time or frequency).
        Parameters:
        d (array-like, optional): Input data array. If None, defaults to self.d for 'time' domain
                                  or self.dtilde for 'frequency' domain.
        Returns:
        jnp.ndarray: The normalized Xtilde array.
        Notes:
        - For 'time' domain, the normalization factor is computed using the time step (self.dt) and the window function.
        - For 'frequency' domain, the normalization factor is computed using the frequency step (self.df).
        - The normalization follows the method described in arXiv:2302.12573.
        """
        
        if self.domain == 'time':
            if d is None:
                d = self.d

            norm = 2.0 * self.dt / jnp.sum(self.window**2)
            Xtilde = jnp.asarray([np.fft.rfft(d[:, i] * self.window) for i in range(self.nchannels)]).T * jnp.sqrt(norm) #ALREADY NORMALIZED, refer to arXiv:2302.12573

        elif self.domain == 'frequency':
            if d is None:
                d = self.dtilde

            norm = 2.0 * self.df

            Xtilde = jnp.asarray(d * np.sqrt(norm)) #ALREADY NORMALIZED, refer to arXiv:2302.12573
        return Xtilde

    
    def get_XtildeXtilde(self, dtilde=None):
        """
        Compute the XtildeXtilde matrix.
        Parameters:
        -----------
        dtilde : array-like, optional
            The input array for which the XtildeXtilde matrix is computed. If not provided, 
            the instance's `dtilde` attribute is used.
        Returns:
        --------
        jnp.ndarray
            The computed XtildeXtilde matrix. If `fullmatrix` is True, the result is a 
            4-dimensional array vectorized over axis 0. If `fullmatrix` is False, the 
            result is a 3-dimensional array vectorized over axis 0.
        """

        if dtilde is None:
            dtilde = self.dtilde
        if self.fullmatrix:
            return jnp.einsum('...i,...j->...ij', jnp.conj(dtilde), dtilde)[jnp.newaxis, :, :, :] #vectorized over axis 0
        else:
            return  jnp.abs(jnp.conj(dtilde) * dtilde)[jnp.newaxis, :, :] #vectorized over axis 0

    def periodogram_matrix(self, data_fd):
        """
        Compute the periodogram matrix of the data.
        
        Parameters
        ----------
        data_fd : array
            The data in frequency domain to compute the periodogram matrix of.
    
        Returns
        -------
        P : array
            The periodogram matrix of the data.
        """
        
        dtilde = jnp.atleast_3d(data_fd)
        dtilde_conj = jnp.conj(dtilde)

        P = (dtilde @ dtilde_conj.transpose(0, 2, 1))

        if not self.fullmatrix: # only the diagonal
            P = jnp.einsum('...ii->...i', P)
        
        return P
    
    def average_periodogram(self, P, freqs, f_seg):
        """
        Average the periodogram matrix over segments. Snippet credits: Nikolaos Karnesis.
        
        Parameters
        ----------
        P : array
            The periodogram matrix to average.
        freqs : array
            The frequencies of the periodogram matrix.
        f_seg : float
            The segment frequency.
            
        Returns
        -------
        freqs_h : array
            The frequencies where the averaged periodogram is computed.
        P_avg : array
            The averaged periodogram matrix.
        segment_sizes : array
            The sizes of the frequency segments.
        """

        df = (freqs[1] - freqs[0])
        if isinstance(f_seg, float):
            # Smoothing bandwidth
            bandwidth = int(f_seg / df)
            # Segment frequencies
            f_seg_arr = freqs[0::bandwidth]
            f_seg_arr = jnp.concatenate((f_seg_arr, jnp.atleast_1d(freqs[-1])))
        elif hasattr(f_seg, '__array__') or isinstance(f_seg, list):
            f_seg_arr = jnp.asarray(f_seg)
        else:
            raise TypeError("f0 should be a float or array_like")
        
        # Number of segments
        n_seg = len(f_seg_arr)
        # Indices of the segment bounds
        i_seg = np.round(f_seg_arr / df).astype(int)
        # Sizes of all intervals
        segment_sizes = i_seg[1:] - i_seg[:-1]
        # Middle frequencies
        freqs_h = (f_seg_arr[:-1] + f_seg_arr[1:]) / 2.0

        # Compute the averages over each segment
        P_avg = jnp.array(
            [jnp.sum(P[i_seg[j]:i_seg[j+1]], axis=0) / segment_sizes[j]
            for j in range(n_seg-1)], dtype=P.dtype)
        
        return freqs_h, P_avg, segment_sizes
       

    def get_window(self, window_func, n):
        """
        Compute the window function and the normalized equivalent noise bandwidth.

        Parameters
        ----------
        window_func : str, tuple, or callable, optional
            The window function to use. If None, a rectangular window is used.
        n : int
            The length of the window.
        
        Returns
        -------
        window : array
            The window function.
        nenbw : float
            The normalized equivalent noise bandwidth.
        
        Notes
        -----
        The normalized equivalent noise bandwidth (nenbw) is defined as:
        .. math::
            \\text{nenbw} = \\frac{N \\sum w^2}{(\\sum w)^2}
        where :math:`N` is the length of the window and :math:`w` is the window function.
        """
        if window_func is None:
            window = np.ones(n)

        elif isinstance(window_func, (str, tuple)):
            window = signal.get_window(window_func, n)

        elif isinstance(window_func, Callable):
            window = window_func(n)

        nenbw = n * np.sum(window**2) / np.sum(window)**2

        return window, nenbw
    
    def loglog(self, fig=None, axs=None, channels=None, **kwargs):
        """
        Plot the log-log periodogram of the data.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot the log-log periodogram on. If not provided, a new figure will be created.
        axs : matplotlib.axes.Axes, optional
            The axes to plot the log-log periodogram on. If not provided, a new axes will be created.
        channels : list, optional
            The channels to plot the log-log periodogram of. If not provided, all channels will be plotted.
        **kwargs : dict
            Additional keyword arguments to pass to the plot
        """
    
        if channels is None:
            channels = range(self.nchannels)
            nchannels = self.nchannels
        elif isinstance(channels, int):
            channels = [channels]
            nchannels = 1
        else:
            nchannels = len(channels)

        if fig is None:
            fig = plt.figure(figsize=(6 * nchannels, 6))
        if axs is None:
            axs = fig.subplots(1, nchannels)
            axs = axs if nchannels > 1 else [axs]
        
        for i, ax in enumerate(axs):
            ax.loglog(self.freqs, self.periodgram.real[:, channels[i]], **kwargs)
            ax.set_xlabel('Frequency [Hz]')
            #ax.set_title(f'Channel {self.channels[i]}')

        axs[0].set_ylabel('PSD [Hz$^{-1}$]')

        plt.tight_layout()
        return fig, axs

