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


class GPUobject:
    '''
    Object which can be run on GPUs
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
        Adjust here which interpolant to use
        '''
        if interpkwargs is None:
            self.interpkwargs = dict(
                kind='akima',
                axis=1
            )
        else:
            assert isinstance(interpkwargs, dict), 'interpkwargs must be a dictionary containing the argument of the spine interpolant'

            self.interpkwargs = interpkwargs

        kind = self.interpkwargs.pop('kind')

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

        else:
            if isinstance(channels, str):
                assert channels in ['AET', 'XYZ'] + self.available_channels, 'Provide either the TDI setup (AET / XYZ) or a list of channels'
                if channels == 'AET':
                    assert Ncov in [3, 6], 'If not providing a list of channels provide the number of independent entries of the covariance matrix (3 or 6 for AET)'
                    self.Ncov = Ncov
                    self.channels = self.available_channels[:self.Ncov]

                elif channels == 'XYZ':
                    assert Ncov in [None, 6], 'Ncov is either not provided or set equal to 6'
                    self.Ncov = 6
                    self.channels = self.available_channels[-self.Ncov:]
                
                else: 
                    channels = [channels]
            
            if isinstance(channels, list):
                self.channels = channels
                if Ncov is not None:
                    assert len(channels) == Ncov
                self.Ncov = len(channels)

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

    def oms_in_isi_carrier(self, freqs):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdOMS)
        psd_meters = asd**2 * self.xp.atleast_2d(1 + (self.fkneeOMS / freqs)**4)
        psd_hertz = self.xp.atleast_2d(2 * self.xp.pi * freqs * CENTRAL_FREQ / C)**2 * psd_meters

        if self.units == 'hertz':
            return self.xp.sqrt(psd_hertz)
        
        elif self.units == 'meters':
            return self.xp.atleast_2d(self.xp.sqrt(psd_meters))
        
        elif self.units == 'strain':
            return self.xp.sqrt(psd_hertz / CENTRAL_FREQ**2)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')

    def filtered_oms_in_isi_carrier(self, freqs):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.

        Include transfer function of derivative filter instead of perfect 2 pi f
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdOMS) # m / sqrt(Hz)
        psd_meters = asd**2 * self.xp.atleast_2d(1 + (self.fkneeOMS / freqs)**4)  # m^2 / Hz

        #psd_hertz = self.xp.atleast_2d(self.xp.abs((-0.5 * self.xp.exp(-2j * self.xp.pi * freqs * 1/FS) + 0.5 * self.xp.exp(2j * self.xp.pi * freqs * 1/FS)))**2 * FS**2 * (CENTRAL_FREQ / C)**2) * psd_meters
        
        psd_highfreq = self.xp.atleast_2d(asd * self.fs * CENTRAL_FREQ / C) ** 2 * self.xp.sin(
            2 * self.xp.pi * freqs / self.fs
        ) ** 2
        psd_lowfreq = self.xp.atleast_2d(
            (2 * self.xp.pi * asd * CENTRAL_FREQ * self.fkneeOMS**2 / C) ** 2
            * self.xp.abs(
                (2 * self.xp.pi * FMIN)
                / (
                    1
                    - self.xp.exp(-2 * self.xp.pi * FMIN / self.fs)
                    * self.xp.exp(-2j * self.xp.pi * freqs / self.fs)
                )
            ) ** 2
            * 1 / (self.fs * FMIN) ** 2
        )
        psd_hertz = psd_highfreq + psd_lowfreq

        if self.units == 'hertz':
            return self.xp.sqrt(psd_hertz) 
        
        elif self.units == 'meters':
            psd_meters = psd_hertz / self.xp.atleast_2d(2 * self.xp.pi * freqs * CENTRAL_FREQ / C)**2
            return self.xp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = psd_hertz / CENTRAL_FREQ**2
            return self.xp.sqrt(psd_strain)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')


    def testmass_single(self, freqs):
        """
        Model for single TM noise PSD including filters used in the data generation
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdTM) # m / s^2 / sqrt(Hz)
        psd_acc = asd**2 * self.xp.atleast_2d(1 + (self.fkneeTM / freqs)**2)  # m^2 / s^4 / Hz

        if self.units == 'hertz':
            psd_hertz = self.xp.atleast_2d(CENTRAL_FREQ / (2 * self.xp.pi * C * freqs))**2 * psd_acc
            return self.xp.sqrt(psd_hertz)

        elif self.units == 'meters':
            psd_meters = self.xp.atleast_2d(2 * self.xp.pi * freqs)**(-4) * psd_acc
            return self.xp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = self.xp.atleast_2d(1 / (2 * self.xp.pi * C * freqs))**2 * psd_acc
            return self.xp.sqrt(psd_strain)


        else:
            raise ValueError('units must be `hertz`, `meters` or `strain`')
        
    def filtered_testmass_single(self, freqs):
        """
        Model for single TM noise PSD
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdTM)
        psd_highfreq = self.xp.atleast_2d(
            (asd * CENTRAL_FREQ / (2 * self.xp.pi * C)) ** 2
            * self.xp.abs(
                (2 * self.xp.pi * FMIN)
                / (
                    1
                    - self.xp.exp(-2 * self.xp.pi * FMIN / self.fs)
                    * self.xp.exp(-2j * self.xp.pi * freqs / self.fs)
                )
            )
            ** 2
            * 1
            / (self.fs * FMIN) ** 2
        )
        psd_lowfreq = self.xp.atleast_2d(
            (asd * CENTRAL_FREQ * self.fkneeTM / (2 * self.xp.pi * C)) ** 2
            * self.xp.abs(
                (2 * self.xp.pi * FMIN)
                / (
                    1
                    - self.xp.exp(-2 * self.xp.pi * FMIN / self.fs)
                    * self.xp.exp(-2j * self.xp.pi * freqs / self.fs)
                )
            )
            ** 2
            * 1
            / (self.fs * FMIN) ** 2
            * self.xp.abs(1 / (1 - self.xp.exp(-2j * self.xp.pi * freqs / self.fs))) ** 2
            * (2 * self.xp.pi / self.fs) ** 2
        )
        psd_hertz = psd_lowfreq + psd_highfreq

        if self.units == 'hertz':
            return self.xp.sqrt(psd_hertz) 
        
        elif self.units == 'meters':
            psd_meters = psd_hertz / self.xp.atleast_2d(2 * self.xp.pi * freqs * CENTRAL_FREQ / C)**2
            return self.xp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = psd_hertz / CENTRAL_FREQ**2
            return self.xp.sqrt(psd_strain)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')
        
    def tdi_common(self, freqs):
        '''
        TDI common factor for both XYZ and AET
        '''
        return 16 * self.xp.sin(2 * self.xp.pi * freqs * self.armlength) * self.xp.sin(4 * self.xp.pi * freqs * self.armlength)**2
    
    def tdi_common_AET(self, freqs):
        return 2 * self.tdi_common(freqs) * self.xp.sin(2 * self.xp.pi * freqs * self.armlength)
    
    def tdi_tf_oms_A(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI A,E.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = self.tdi_common_AET(freqs) * (2 + self.xp.cos(2 * xp.pi * freqs * self.armlength))
        return self.xp.sqrt(self.xp.atleast_2d(psd))
        
    def tdi_tf_oms_T(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI T.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 2 * self.tdi_common_AET(freqs) * (1 - self.xp.cos(2 * self.xp.pi * freqs * self.armlength))
        return self.xp.sqrt(self.xp.atleast_2d(psd))
    
    def tdi_tf_testmass_A(self, freqs):
        """
        TDI transfer function for testmass noise in TDI A,E.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common_AET(freqs) * (1 + self.xp.cos(2 * xp.pi * freqs * self.armlength) + self.xp.cos(2 * xp.pi * freqs * self.armlength)**2 )
                            
        return self.xp.sqrt(self.xp.atleast_2d(psd))
        
    def tdi_tf_testmass_T(self, freqs):
        """
        TDI transfer function for testmass noise in TDI T.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common_AET(freqs) * (1 - self.xp.cos(2 * self.xp.pi * freqs * self.armlength))**2
        return self.xp.sqrt(self.xp.atleast_2d(psd))
    
    def tdi_tf_oms_XX(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI XX, YY, ZZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common(freqs) * self.xp.sin(2 * self.xp.pi * freqs * self.armlength)
        return self.xp.sqrt(self.xp.atleast_2d(psd))
    
    def tdi_tf_oms_XY2(self, freqs):
        """
        TDI transfer function for ISI OMS noise in TDI XY, XZ, YZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = - self.tdi_common(freqs) * self.xp.sin(4 * self.xp.pi * freqs * self.armlength)
        return self.xp.atleast_2d(psd)
    
    def tdi_tf_testmass_XX(self, freqs):
        """
        TDI transfer function for testmass noise in TDI XX, YY, ZZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common(freqs) * self.xp.sin(2 * self.xp.pi * freqs * self.armlength) * (3 + self.xp.cos(4 * self.xp.pi * freqs * self.armlength))
        return self.xp.sqrt(self.xp.atleast_2d(psd))
    
    def tdi_tf_testmass_XY2(self, freqs):
        """
        TDI transfer function for testmass noise in TDI XY, XZ, YZ.
        
        Args:
            freqs (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = - 4 * self.tdi_common(freqs) * self.xp.sin(4 * self.xp.pi * freqs * self.armlength)
        return self.xp.atleast_2d(psd)

    #! AET
    def testmass_A(self, freqs):
        return self.tdi_tf_testmass_A(freqs) * self.filtered_testmass_single(freqs)

    def testmass_T(self, freqs):
        return self.tdi_tf_testmass_T(freqs) * self.filtered_testmass_single(freqs)

    def oms_A(self, freqs):
        return self.tdi_tf_oms_A(freqs) * self.filtered_oms_in_isi_carrier(freqs)

    def oms_T(self, freqs):
        return self.tdi_tf_oms_T(freqs) * self.filtered_oms_in_isi_carrier(freqs)

    def get_SA(self, freqs):
        '''
        Uncorrelated noise in the A, E tdi channels
        '''
        #return xp.atleast_2d(testmass_A(asd=asdTM)**2) + xp.atleast_2d(oms_A(asd=asdOMS)**2)
        return self.testmass_A(freqs)**2 + self.oms_A(freqs)**2 

    def get_ST(self, freqs):
        '''
        Uncorrelated noise in the T tdi channel
        '''
        #return xp.atleast_2d(testmass_T(asd=asdTM)**2) + xp.atleast_2d(oms_T(asd=asdOMS)*2)
        return self.testmass_T(freqs)**2 + self.oms_T(freqs)**2 
    
    #! XYZ
    def testmass_XX(self, freqs):
        return self.tdi_tf_testmass_XX(freqs) * self.filtered_testmass_single(freqs)

    def testmass_XY(self, freqs):
        return self.tdi_tf_testmass_XY2(freqs) * self.filtered_testmass_single(freqs)**2

    def oms_XX(self, freqs):
        return self.tdi_tf_oms_XX(freqs) * self.filtered_oms_in_isi_carrier(freqs)

    def oms_XY(self, freqs):
        return self.tdi_tf_oms_XY2(freqs) * self.filtered_oms_in_isi_carrier(freqs)**2

    def get_SXX(self, freqs):
        '''
        noise in the XX, YY, ZZ tdi channels
        '''
        #return xp.atleast_2d(testmass_A(asd=asdTM)**2) + xp.atleast_2d(oms_A(asd=asdOMS)**2)
        return self.testmass_XX(freqs)**2 + self.oms_XX(freqs)**2 

    def get_SXY(self, freqs):
        '''
        noise in the XY, XZ, YZ tdi channels
        '''
        #return xp.atleast_2d(testmass_T(asd=asdTM)**2) + xp.atleast_2d(oms_T(asd=asdOMS)*2)
        return (self.testmass_XY(freqs) + self.oms_XY(freqs))

    def set_PSDS(self, freqs):
        '''TODO: allow specific channnels'''
        self.PSDS_design = (self.xp.asarray([self.available_functions[channel](freqs) for channel in self.channels])).transpose(1,2,0)

        #self.PSDS_design = (self.xp.asarray([self.get_SA(freqs)[0], self.get_SA(freqs)[0], self.get_ST(freqs)[0]]).T)[self.xp.newaxis, : , :]

    def get_PSDS(self, freqs=None, overwrite=False, **kwargs):
        if (self.PSDS_design is None):
            if freqs is None:
                raise ValueError('provide frequencies')
            else:
                self.set_PSDS(freqs)
        
        if overwrite:
            self.set_PSDS(freqs)
        
        return self.PSDS_design
    

class TDIresponse(GPUobject):
        '''
        Compute a spline interpolant for the TDI response in the A, E, T channels. Points used for the interpolation comes from the
        Mathematica notebook.
        The file must contain the frequencies used to compute the transfer functions and the latter for A, E, T in this exact order.
        The class supports both CPUs and GPUs.
        '''
        def __init__(self, filename, use_gpu=False, interpkwargs=dict(kind='akima', axis=0)):

            GPUobject.__init__(self, use_gpu=use_gpu, interpkwargs=interpkwargs)

            TDIcsd_re = self.xp.transpose( self.xp.genfromtxt(filename, delimiter=','))
            self.freqs = TDIcsd_re[0]
            # psd A,E and T
            self.tf_A = TDIcsd_re[1]
            self.tf_E = TDIcsd_re[2]
            self.tf_T = TDIcsd_re[3]
            
            self.tfs = self.xp.stack((self.tf_A, self.tf_E, self.tf_T)).T

            self.TDIinterpolant = self.interp(self.freqs, self.tfs)

        def __call__(self, freqs, return_gpu=True):
            
            response = self.TDIinterpolant(freqs)

            if not return_gpu:
                return response.get() #return a numpy array

            else:
                return response #return a self.xp array