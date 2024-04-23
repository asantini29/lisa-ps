from abc import ABC
import numpy as np

try:
    import cupy as xp
    from cupyx.scipy.interpolate import make_interp_spline as cupy_make_interp_spline
    from cupyx.scipy.interpolate import Akima1DInterpolator as cupy_Akima1DInterpolator

except (ModuleNotFoundError, ImportError):
    import numpy as xp

from few.summation.interpolatedmodesum import CubicSplineInterpolant
from scipy.interpolate import make_interp_spline as scipy_make_interp_spline
from scipy.interpolate import Akima1DInterpolator as scipy_Akima1DInterpolator

from .constants import *
from .akima import AkimaInterpolant


import jax
import jax.numpy as jnp
from functools import partial

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
    '''

    def __init__(self, use_gpu=False, interpkwargs=None):
        self.use_gpu = use_gpu
        self.xp = xp if use_gpu else np
        
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

        if kind == 'cubic':
            self.interp =  CubicSplineInterpolant
            self.interpkwargs = dict(use_gpu=self.use_gpu)

        elif kind == 'bsplines':
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
                ndim_out = self.interpkwargs['ndim_out'] if 'ndim_out' in self.interpkwargs.keys() else 2
                threadsperblock = self.interpkwargs['threadsperblock'] if 'threadsperblock' in self.interpkwargs.keys() else 32
                
                AkimaInterpolantNumba = AkimaInterpolant(ndim_out=ndim_out, threadsperblock=threadsperblock)
                self.interp = AkimaInterpolantNumba

        else:
            raise NotImplementedError

    

class BaseNoise(GPUobject):
    """
    Base class for modeling noise in a LISA.

    Args:
        asdTM (float): Amplitude spectral density of test mass noise [m/s^2/sqrt(Hz)].
        asdOMS (float): Amplitude spectral density of optical metrology system noise [m/sqrt(Hz)].
        fkneeTM (float): Test mass noise knee frequency [Hz].
        fkneeOMS (float): Optical metrology system noise knee frequency [Hz].
        equal_arms (bool): Flag indicating whether the detector has equal arm lengths.
        fs (float): Sampling frequency [Hz].
        Ncov (int): Number of channels to consider.
        channels (str or list): Channels to consider. Can be 'AET', 'XYZ', or a list of channel names.
        use_gpu (bool): Flag indicating whether to use GPU acceleration.
        units (str): Units of the noise PSD. Can be 'hertz', 'meters', or 'strain'.
        interpkwargs (dict): Keyword arguments for interpolation.

    Attributes:
        armlength (float): Length of the detector arms.
        fs (float): Sampling frequency.
        asdTM (float): Amplitude spectral density of test mass noise.
        asdOMS (float): Amplitude spectral density of optical metrology system noise.
        fkneeTM (float): Test mass noise knee frequency.
        fkneeOMS (float): Optical metrology system noise knee frequency.
        available_channels (list): List of available channel names.
        available_functions (dict): Dictionary mapping channel names to corresponding functions.
        Ncov (int): Number of channels to consider.
        channels (list): List of channel names to consider.
        TDIsetup (str): TDI setup configuration.
        units (str): Units of the noise PSD.

    Methods:
        oms_in_isi_carrier(freqs): Model for OMS noise PSD in ISI carrier beatnote fluctuations.
        filtered_oms_in_isi_carrier(freqs): Model for OMS noise PSD in ISI carrier beatnote fluctuations with filtered transfer function.
        testmass_single(freqs): Model for single test mass noise PSD including filters used in the data generation.
        filtered_testmass_single(freqs): Model for single test mass noise PSD with filtered transfer function.
        tdi_common(freqs): TDI common factor for both XYZ and AET configurations.
        tdi_common_AET(freqs): TDI common factor for AET configuration.
        tdi_tf_oms_A(freqs): TDI transfer function for ISI OMS noise in TDI A,E.
        tdi_tf_oms_T(freqs): TDI transfer function for ISI OMS noise in TDI T.
    """

    def __init__(self, asdTM=2.4e-15, asdOMS=7.9e-12, fkneeTM=0.4e-3, fkneeOMS=2e-3, equal_arms=False, fs=None, Ncov=None, channels='AET', use_gpu=False, units='hertz', interpkwargs=dict(kind='akima', axis=1)):
        GPUobject.__init__(self, use_gpu=use_gpu, interpkwargs=interpkwargs)

        self.armlength = ARMLENGTH_EQUAL if equal_arms else ARMLENGTH_AVERAGE
        self.fs = fs if fs is not None else FS

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
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = jnp.atleast_2d(asdOMS)
        psd_meters = asd**2 * jnp.atleast_2d(1 + (self.fkneeOMS / freqs)**4)
        psd_hertz = jnp.atleast_2d(2 * jnp.pi * freqs * CENTRAL_FREQ / C)**2 * psd_meters

        if self.units == 'hertz':
            return jnp.sqrt(psd_hertz)
        
        elif self.units == 'meters':
            return jnp.atleast_2d(jnp.sqrt(psd_meters))
        
        elif self.units == 'strain':
            return jnp.sqrt(psd_hertz / CENTRAL_FREQ**2)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')

    def filtered_oms_in_isi_carrier(self, asdOMS, freqs):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.

        Include transfer function of derivative filter instead of perfect 2 pi f
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        #asd = jnp.atleast_2d(asdOMS) # m / sqrt(Hz)
        asd = asdOMS # m / sqrt(Hz)
        psd_meters = asd**2 * (1 + (self.fkneeOMS / freqs)**4) #jnp.atleast_2d(1 + (self.fkneeOMS / freqs)**4)  # m^2 / Hz

        #psd_hertz = jnp.atleast_2d(jnp.abs((-0.5 * jnp.exp(-2j * jnp.pi * freqs * 1/FS) + 0.5 * jnp.exp(2j * jnp.pi * freqs * 1/FS)))**2 * FS**2 * (CENTRAL_FREQ / C)**2) * psd_meters
        
        #psd_highfreq = jnp.atleast_2d(asd * self.fs * CENTRAL_FREQ / C) ** 2 * jnp.sin(
        psd_highfreq = (asd * self.fs * CENTRAL_FREQ / C) ** 2 * jnp.sin(
            2 * jnp.pi * freqs / self.fs
        ) ** 2
        psd_lowfreq = (#jnp.atleast_2d(
            (2 * jnp.pi * asd * CENTRAL_FREQ * self.fkneeOMS**2 / C) ** 2
            * jnp.abs(
                (2 * jnp.pi * FMIN)
                / (
                    1
                    - jnp.exp(-2 * jnp.pi * FMIN / self.fs)
                    * jnp.exp(-2j * jnp.pi * freqs / self.fs)
                )
            ) ** 2
            * 1 / (self.fs * FMIN) ** 2
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
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
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
        
    def filtered_testmass_single(self, asdTM, freqs):
        """
        Model for single TM noise PSD
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
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
        #psd_lowfreq = jnp.atleast_2d(
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
        '''
        return 16 * jnp.sin(2 * jnp.pi * freqs * self.armlength) * jnp.sin(4 * jnp.pi * freqs * self.armlength)**2
    
    def tdi_common_AET(self, freqs):
        return 2 * self.tdi_common(freqs) * jnp.sin(2 * jnp.pi * freqs * self.armlength)
    
    def tdi_tf_oms_A(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI A,E.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = self.tdi_common_AET(freqs) * (2 + jnp.cos(2 * xp.pi * freqs * self.armlength))
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
        
    def tdi_tf_oms_T(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI T.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 2 * self.tdi_common_AET(freqs) * (1 - jnp.cos(2 * jnp.pi * freqs * self.armlength))
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_testmass_A(self, freqs):
        """
        TDI transfer function for testmass noise in TDI A,E.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common_AET(freqs) * (1 + jnp.cos(2 * xp.pi * freqs * self.armlength) + jnp.cos(2 * xp.pi * freqs * self.armlength)**2 )
                            
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
        
    def tdi_tf_testmass_T(self, freqs):
        """
        TDI transfer function for testmass noise in TDI T.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common_AET(freqs) * (1 - jnp.cos(2 * jnp.pi * freqs * self.armlength))**2
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_oms_XX(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI XX, YY, ZZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common(freqs) * jnp.sin(2 * jnp.pi * freqs * self.armlength)
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_oms_XY2(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI XY, XZ, YZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = - self.tdi_common(freqs) * jnp.sin(4 * jnp.pi * freqs * self.armlength)
        #return jnp.atleast_2d(psd)
        return psd
    
    def tdi_tf_testmass_XX(self, freqs):
        """
        TDI transfer function for testmass noise in TDI XX, YY, ZZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common(freqs) * jnp.sin(2 * jnp.pi * freqs * self.armlength) * (3 + jnp.cos(4 * jnp.pi * freqs * self.armlength))
        #return jnp.sqrt(jnp.atleast_2d(psd))
        return jnp.sqrt(psd)
    
    def tdi_tf_testmass_XY2(self, freqs):
        """
        TDI transfer function for testmass noise in TDI XY, XZ, YZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = - 4 * self.tdi_common(freqs) * jnp.sin(4 * jnp.pi * freqs * self.armlength)
        #return jnp.atleast_2d(psd)
        return psd

    #! AET
    def testmass_A(self, asdTM, freqs):
        return self.tdi_tf_testmass_A(freqs) * self.filtered_testmass_single(asdTM, freqs)

    def testmass_T(self, asdTM, freqs):
        return self.tdi_tf_testmass_T(freqs) * self.filtered_testmass_single(asdTM, freqs)

    def oms_A(self, asdOMS, freqs):
        return self.tdi_tf_oms_A(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)

    def oms_T(self, asdOMS, freqs):
        return self.tdi_tf_oms_T(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)

    @partial(jax.jit, static_argnums=(0,))
    def get_SA(self, asdTM, asdOMS, freqs):
        '''
        Uncorrelated noise in the A, E tdi channels
        '''
        #return xp.atleast_2d(testmass_A(asd=asdTM)**2) + xp.atleast_2d(oms_A(asd=asdOMS)**2)
        return self.testmass_A(asdTM, freqs)**2 + self.oms_A(asdOMS, freqs)**2 
    
    @partial(jax.jit, static_argnums=(0,))
    def get_ST(self, asdTM, asdOMS, freqs):
        '''
        Uncorrelated noise in the T tdi channel
        '''
        #return xp.atleast_2d(testmass_T(asd=asdTM)**2) + xp.atleast_2d(oms_T(asd=asdOMS)*2)
        return self.testmass_T(asdTM, freqs)**2 + self.oms_T(asdOMS, freqs)**2 
    
    #! XYZ
    def testmass_XX(self, asdTM, freqs):
        return self.tdi_tf_testmass_XX(freqs) * self.filtered_testmass_single(asdTM, freqs)

    def testmass_XY(self,asdTM, freqs):
        return self.tdi_tf_testmass_XY2(freqs) * self.filtered_testmass_single(asdTM, freqs)**2

    def oms_XX(self, asdOMS, freqs):
        return self.tdi_tf_oms_XX(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)

    def oms_XY(self, asdOMS, freqs):
        return self.tdi_tf_oms_XY2(freqs) * self.filtered_oms_in_isi_carrier(asdOMS, freqs)**2

    @partial(jax.jit, static_argnums=(0,))
    def get_SXX(self, asdTM, asdOMS, freqs):
        '''
        noise in the XX, YY, ZZ tdi channels
        '''
        #return xp.atleast_2d(testmass_A(asd=asdTM)**2) + xp.atleast_2d(oms_A(asd=asdOMS)**2)
        return self.testmass_XX(asdTM, freqs)**2 + self.oms_XX(asdOMS, freqs)**2 

    @partial(jax.jit, static_argnums=(0,))
    def get_SXY(self, asdTM, asdOMS, freqs):
        '''
        noise in the XY, XZ, YZ tdi channels
        '''
        #return xp.atleast_2d(testmass_T(asd=asdTM)**2) + xp.atleast_2d(oms_T(asd=asdOMS)*2)
        return (self.testmass_XY(asdTM, freqs) + self.oms_XY(asdOMS, freqs))
    
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
    
    def compute_batched_PSDS(self, asdTM, asdOMS, freqs):
        """
        Compute the batched Power Spectral Density (PSD) using the given ASDs and frequencies.

        Parameters:
        - asdTM (ndarray): The ASD (Amplitude Spectral Density) for the TM (Test Mass) channel.
        - asdOMS (ndarray): The ASD for the OMS (Optical Metrology System) channel.
        - freqs (ndarray): The frequencies at which to compute the PSD.

        Returns:
        - ndarray: The computed batched PSDs.

        """
        return jax.vmap(self.compute_PSDS, in_axes=(0, 0, None))(asdTM, asdOMS, freqs)

    def set_PSDS(self, freqs):
        asdTM, asdOMS = jnp.atleast_1d(self.asdTM), jnp.atleast_1d(self.asdOMS)
        self.PSDS_design = self.compute_batched_PSDS(asdTM, asdOMS, freqs)

    
    def get_PSDS(self, asdTM, asdOMS, freqs=None):
        asdTM, asdOMS = jnp.atleast_1d(asdTM), jnp.atleast_1d(asdOMS)
        return self.compute_batched_PSDS(asdTM, asdOMS, freqs)
    

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

    def __call__(self, freqs, return_gpu=True):
        """
        Calculates the TDI response at the given frequencies.
        
        Args:
            freqs (array-like): The frequencies at which to calculate the TDI response.
            return_gpu (bool, optional): Flag indicating whether to return the response as a GPU array. Default: `True`.
        
        Returns:
            array-like: The TDI response at the given frequencies.
        """
        response_real = self.TDIinterpolant_real(freqs)
        response_imag = self.TDIinterpolant_imag(freqs)

        response = response_real + 1j * response_imag

        if not return_gpu:
            return response.get()  # return a numpy array
        else:
            return response  # return a self.xp array
