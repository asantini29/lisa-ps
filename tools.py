#!/usr/bin/env python3

import sys, os
from typing import Any, Callable
import numpy as np
from inspect import signature
# try to import cupy
use_gpu = True

if use_gpu:
    try:
        import cupy as xp    
        from cupyx import scipy
        import cupyx
        from cupyx.scipy.interpolate import make_interp_spline as cupy_make_interp_spline
        from cupyx.scipy.interpolate import Akima1DInterpolator as cupy_Akima1DInterpolator
        from scipy.interpolate import CubicSpline
        gpu_available = True

    except:
        import numpy as xp
        gpu_available = False
        import scipy
        # from scipy.interpolate import make_interp_spline as scipy_make_interp_spline
        # from scipy.interpolate import Akima1DInterpolator as scipy_Akima1DInterpolator
else:
    xp = np

from few.summation.interpolatedmodesum import CubicSplineInterpolant
import scipy
from scipy.interpolate import make_interp_spline as scipy_make_interp_spline
from scipy.interpolate import Akima1DInterpolator as scipy_Akima1DInterpolator

from constants import *
import time

'''
----------------------------
    BASE NOISE FUNCTIONS    
----------------------------
'''

def oms_in_isi_carrier(freq, asd=7.9e-12, fknee=2e-3, units='hertz'):
    """
    Model for OMS noise PSD in ISI carrier beatnote fluctuations.
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    asd = xp.atleast_2d(asd)
    psd_meters = asd**2 * xp.atleast_2d(1 + (fknee / freq)**4)
    psd_hertz = xp.atleast_2d(2 * xp.pi * freq * CENTRAL_FREQ / C)**2 * psd_meters

    if units == 'hertz':
        return xp.sqrt(psd_hertz)
    
    elif units == 'meters':
        return xp.atleast_2d(xp.sqrt(psd_meters))

    else:
        raise ValueError('units must be `hertz` or `meters`')

def filtered_oms_in_isi_carrier(freq, asd=7.9e-12, fknee=2e-3, units='hertz'):
    """
    Model for OMS noise PSD in ISI carrier beatnote fluctuations.

    Include transfer function of derivative filter instead of perfect 2 pi f
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    asd = xp.atleast_2d(asd) # m / sqrt(Hz)
    psd_meters = asd**2 * xp.atleast_2d(1 + (fknee / freq)**4)  # m^2 / Hz

    psd_hertz = xp.atleast_2d(xp.abs((-0.5 * xp.exp(-2j * xp.pi * freq * 1/FS) + 0.5 * xp.exp(2j * xp.pi * freq * 1/FS)))**2 * FS**2 * (CENTRAL_FREQ / C)**2) * psd_meters
    
    if units == 'hertz':
        return xp.sqrt(psd_hertz) 
    
    elif units == 'meters':
        psd_meters = psd_hertz / xp.atleast_2d(2 * xp.pi * freq * CENTRAL_FREQ / C)**2
        return xp.sqrt(psd_meters)

    else:
        raise ValueError('units must be `hertz` or `meters`')


def testmass_single(freq, asd=2.4e-15, fknee=0.4E-3, units='hertz'):
    """
    Model for single TM noise PSD
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    asd = xp.atleast_2d(asd) # m / s^2 / sqrt(Hz)
    psd_acc = asd**2 * xp.atleast_2d(1 + (fknee / freq)**2)  # m^2 / s^4 / Hz

    if units == 'hertz':
        psd_hertz = xp.atleast_2d(CENTRAL_FREQ / (2 * xp.pi * C * freq))**2 * psd_acc
        return xp.sqrt(psd_hertz)

    elif units == 'meters':
        psd_meters = xp.atleast_2d(2 * xp.pi * freq)**(-4) * psd_acc
        return xp.sqrt(psd_meters)

    else:
        raise ValueError('units must be `hertz` or `meters`')
    

def tdi_common(freq):
    """
    TDI common factor.
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    return 16 * xp.sin(2 * xp.pi * freq * ARMLENGTH)**2 \
        * xp.sin(4 * xp.pi * freq * ARMLENGTH)**2

def tdi_tf_oms_A(freq):
    """
    TDI transfer function for ISI OMS noise in TDI A,E.
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    psd = 2 * tdi_common(freq) * (2 + xp.cos(2 * xp.pi * freq * ARMLENGTH))
    return xp.sqrt(xp.atleast_2d(psd))
    
def tdi_tf_oms_T(freq):
    """
    TDI transfer function for ISI OMS noise in TDI T.
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    psd = 4 * tdi_common(freq) * (1 - xp.cos(2 * xp.pi * freq * ARMLENGTH))
    return xp.sqrt(xp.atleast_2d(psd))
        
def tdi_tf_testmass_A(freq):
    """
    TDI transfer function for test mass noise in TDI A,E.

    Note that we remove a factor 4 wrt. the usual expression in the literature, since we included a factor 4 in the TMI expression
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    psd = 4 * tdi_common(freq) * (3 + 2 * xp.cos(2 * xp.pi * freq * ARMLENGTH) + xp.cos(4 * xp.pi * freq * ARMLENGTH))
    return xp.sqrt(xp.atleast_2d(psd))


def tdi_tf_testmass_T(freq):
    """
    TDI transfer function for test mass noise in TDI T.

    Note that we remove a factor 4 wrt. the usual expression in the literature, since we included a factor 4 in the TMI expression
    
    Args:
        freq (float): frequencies [Hz]
        instru (Instrument): LISA instrument object
    """
    psd = 32 * tdi_common(freq) * xp.sin(2 * xp.pi * freq * ARMLENGTH / 2)**4
    return xp.sqrt(xp.atleast_2d(psd))

def testmass_A(freq, asd=2.4e-15, units='hertz'):
    return tdi_tf_testmass_A(freq) * testmass_single(freq, asd=asd, units=units)

def testmass_T(freq, asd=2.4e-15, units='hertz'):
    return tdi_tf_testmass_T(freq) * testmass_single(freq, asd=asd, units=units)

def oms_A(freq, asd=7.9e-12, units='hertz'):
    return tdi_tf_oms_A(freq) * filtered_oms_in_isi_carrier(freq, asd=asd, units=units)

def oms_T(freq, asd=7.9e-12, units='hertz'):
    return tdi_tf_oms_T(freq) * filtered_oms_in_isi_carrier(freq, asd=asd, units=units)

def get_SA(freq, asdTM=2.4e-15, asdOMS=7.9e-12, units='hertz'):
    '''
    Uncorrelated noise in the A, E tdi channels
    '''
    #return xp.atleast_2d(testmass_A(asd=asdTM)**2) + xp.atleast_2d(oms_A(asd=asdOMS)**2)
    return testmass_A(freq, asd=asdTM, units=units)**2 + oms_A(freq, asd=asdOMS, units=units)**2

def get_ST(freq, asdTM=2.4e-15, asdOMS=7.9e-12, units='hertz'):
    '''
    Uncorrelated noise in the T tdi channel
    '''
    #return xp.atleast_2d(testmass_T(asd=asdTM)**2) + xp.atleast_2d(oms_T(asd=asdOMS)**2)
    return testmass_T(freq, asd=asdTM, units=units)**2 + oms_T(freq, asd=asdOMS, units=units)**2 

'''
----------------------------
    LIKELIHOOD FUNCTIONS    
----------------------------
'''

def log_like_fn_cpu_uncorrelated(args, get_PSDS, freq, XtildeXtilde, T, **kwargs):
        '''
        This likelihood works only for uncorrelated TDI channels but it's fast, as it does not occupy memory for matrices full of zeros
        '''
        args = np.atleast_2d(args)
        PSDS = get_PSDS(freq, args, **kwargs)

        return - xp.sum( xp.sum( 2 / T * XtildeXtilde / PSDS + xp.log(PSDS), axis = -1) , axis = -1)
    

def log_like_fn_gpu_uncorrelated(args, get_PSDS, freq, XtildeXtilde, T, backgroundkeys = [], includenoise = True, **kwargs):
        '''
        This likelihood works only for uncorrelated TDI channels but it's fast, as it does not occupy memory for matrices full of zeros
        Args:
            args: Eryn input
            get_PSDS (callable): a callable object that returns the psds corresponding to args input
            freq (array): array of frequencies at which the PSD is computed.
            XtildeXtilde
            T
            backgroundkeys (list): list of backgrounds the code has to look for (default: [])
            includenoise (bool): whether the noise has to be included in the analysis (default: True)
            kwargs (dict): additional arguments to be passed to `get_PSDS`
        '''
        #breakpoint()
        if includenoise:
            if isinstance(args, list):
                noiseargs_all, backargs_all = args[0], args[1:]
                backargs_all_dict = dict(zip(backgroundkeys, backargs_all))
            else:
                noiseargs_all = args
                backargs = None

            ninputs = noiseargs_all.shape[0]

            
            noiseargs_all = np.atleast_2d(noiseargs_all)

        else:
            noiseargs = None
            if not isinstance(args, list):
                backargs_all = [args]
            backargs_all_dict = dict(zip(backgroundkeys, backargs_all))
            
            ninputs = backargs_all[0].shape[0]
            
        if 'subset' in kwargs.keys():
            subset = kwargs.pop('subset', '')
        else:
            subset = ninputs

        inds = np.arange(0, ninputs + 1, subset)

        if inds[-1] < ninputs:
            inds = np.concatenate([inds, np.array([ninputs])])
        logl_all = []
        
        for i in range(len(inds) - 1):
            start = inds[i]
            end = inds[i + 1]
            if includenoise:
                noiseargs = noiseargs_all[start:end]
            backargs = {}
            for key in backgroundkeys:
                backargs[key] = backargs_all_dict[key][start:end]

            PSDS = get_PSDS(freq, noiseargs, backargs, **kwargs)

            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()

            logl = - xp.sum( xp.sum( 2 / T * XtildeXtilde / PSDS + xp.log(PSDS), axis = -1) , axis = -1)

            logl_all.append(logl)
        
        logl_out = np.concatenate(logl_all)

        return logl_out.get()

def log_like_fn_gpu_averaged_uncorrelated(args, get_PSDS, freq, XtildeXtilde, nu = 1, includenoise = True, noisekeys = [], backgroundkeys = [], **kwargs):
        '''
        This likelihood works only for uncorrelated TDI channels but it's fast, as it does not occupy memory for matrices full of zeros
        Args:
            args: Eryn input
            get_PSDS (callable): a callable object that returns the psds corresponding to args input
            freq (array): array of frequencies at which the PSD is computed.
            XtildeXtilde
            T
            backgroundkeys (list): list of backgrounds the code has to look for (default: [])
            includenoise (bool): whether the noise has to be included in the analysis (default: True)
            kwargs (dict): additional arguments to be passed to `get_PSDS`
        '''
        
        if includenoise:
            if isinstance(args, list):
                noiseargs_all, backargs_all = args[:len(noisekeys)], args[len(noisekeys):]
                backargs_all_dict = dict(zip(backgroundkeys, backargs_all))
                ninputs = noiseargs_all[0].shape[0]
            else:
                noiseargs_all = args
                backargs = None
                ninputs = noiseargs_all.shape[0]
            
            noiseargs_all_dict = dict(zip(noisekeys, noiseargs_all))

        else:
            noiseargs = None
            if not isinstance(args, list):
                backargs_all = [args]
            backargs_all_dict = dict(zip(backgroundkeys, backargs_all))
            
            ninputs = backargs_all[0].shape[0]
            
        if 'subset' in kwargs.keys():
            subset = kwargs.pop('subset', '')
        else:
            subset = ninputs

        inds = np.arange(0, ninputs + 1, subset)

        if inds[-1] < ninputs:
            inds = np.concatenate([inds, np.array([ninputs])])
        logl_all = []
        
        #breakpoint()
        for i in range(len(inds) - 1):
            start = inds[i]
            end = inds[i + 1]
        
            if includenoise:
                noiseargs = {}
                for key in noisekeys:
                    noiseargs[key] = noiseargs_all_dict[key][start:end]
                
            backargs = {}
            for key in backgroundkeys:
                backargs[key] = backargs_all_dict[key][start:end]

            PSDS = get_PSDS(freq, noiseargs, backargs, **kwargs)

            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()

            logl = - xp.sum( xp.sum(XtildeXtilde / PSDS, axis = -1) + nu * xp.sum(xp.log(PSDS), axis = -1) , axis = -1)

            logl_all.append(logl)
        
        logl_out = np.concatenate(logl_all)

        return logl_out.get()

def log_like_fn_rj_gpu_averaged_uncorrelated(args, groups, get_PSDS, freq, XtildeXtilde, nu = 1, includenoise = True, noisekeys = [], backgroundkeys = [], inf=1e14, **kwargs):
        '''
        Args:
            args: Eryn input
            get_PSDS (callable): a callable object that returns the psds corresponding to args input
            freq (array): array of frequencies at which the PSD is computed.
            XtildeXtilde
            T
            backgroundkeys (list): list of backgrounds the code has to look for (default: [])
            includenoise (bool): whether the noise has to be included in the analysis (default: True)
            kwargs (dict): additional arguments to be passed to `get_PSDS`
        '''
        
        unique_groups = np.unique(np.concatenate([groups_i for groups_i in groups]))
        ngroups = unique_groups.max() + 1
        # for i, group in enumerate(groups):
        #     if len(group) > ngroups:
        #         rj_idx = i
        # counts, edges = np.histogram(groups[rj_idx], bins=ngroups)
        # counts_unique  = np.unique(counts)

        #tic = time.perf_counter()
        idx_sgwb = 0
        if includenoise:
            if isinstance(args, list):
                idx_sgwb = len(noisekeys)
                noiseargs_all, backargs_all = args[:idx_sgwb], args[idx_sgwb:]
                backargs_all_dict = dict(zip(backgroundkeys, backargs_all))
            else:
                noiseargs_all = args
                backargs = None
            
            noiseargs_all_dict = dict(zip(noisekeys, noiseargs_all))
    
        else:
            noiseargs = None
            if not isinstance(args, list):
                backargs_all = [args]
            backargs_all_dict = dict(zip(backgroundkeys, backargs_all))
        
        #toc = time.perf_counter()
        #print("sec 1: setup  ",toc-tic)

        #tic = time.perf_counter()
        logl_all = []
        
        for i in range(ngroups):
            #breakpoint()
            if includenoise:
                noiseargs = {}
                for j, key in enumerate(noisekeys):
                    inds = np.where(groups[j] == i)
                    noiseargs[key] = noiseargs_all_dict[key][inds]
                
            backargs = {}
            for j, key in enumerate(backgroundkeys):
                inds = np.where(groups[idx_sgwb + j] == i)
                backargs[key] = backargs_all_dict[key][inds]
        
        # for count in counts_unique:
        #     idxs_all = np.unique(groups[rj_idx])[counts == count]
        #     if includenoise:
        #         noiseargs = {}
        #         for j, key in enumerate(noisekeys):
        #             noiseargs[key] = np.array([noiseargs_all_dict[key][groups[j] == idx] for idx in idxs_all])
            
        #     backargs = {}
        #     for j, key in enumerate(backgroundkeys):
        #         backargs[key] = np.array([backargs_all_dict[key][groups[j] == idx] for idx in idxs_all])
        
            
            PSDS = get_PSDS(freq, noiseargs, backargs, **kwargs)
            #breakpoint()

            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()

            logl = - xp.sum( xp.sum(XtildeXtilde / PSDS, axis = -1) + nu * xp.sum(xp.log(PSDS), axis = -1) , axis = -1)
            #logl = - xp.sum((XtildeXtilde / PSDS) + nu * (xp.log(PSDS)), axis = (1, 2))

            logl_all.append(logl)

        #toc = time.perf_counter()
        #print("sec 2: psds and append ",toc-tic)

        logl_out = np.concatenate(logl_all)

        logl_out[~np.isfinite(logl_out)] = -inf

        return logl_out.get()

def get_XtildeXtilde(Xtilde):
     return xp.real(xp.conj(Xtilde) * Xtilde)[xp.newaxis, :, :]

'''
----------------------------
    PSDS FUNCTIONS, fit the ASDs    
----------------------------
'''

def compute_PSD_uncorrelated_2factors(freq, args, Ncov=3, **kwargs):
    '''
    The multiplicative factors are the same for all the channels
    '''

    PSDS = xp.empty((args.shape[0], len(freq), Ncov))

    A, B = xp.asarray(args[:, 0, np.newaxis]), xp.asarray(args[:, 1, np.newaxis])
    C, D = A, B
    E, F = A, B
    
    PSDS[:, :, 0] = get_SA(freq, asdTM=A, asdOMS=B, **kwargs) # A channel
    PSDS[:, :, 1] = get_SA(freq, asdTM=C, asdOMS=D, **kwargs) # E channel
    PSDS[:, :, 2] = get_ST(freq, asdTM=E, asdOMS=F, **kwargs) # T channel

    return PSDS
    
def compute_PSD_uncorrelated_4factors(freq, args, Ncov=3, **kwargs):
    '''
    The multiplicative factors are the same for the A, E channels but different for T
    '''

    PSDS = xp.empty((args.shape[0], len(freq), Ncov))

    A, B, E, F = xp.asarray(args[:, 0, np.newaxis]), xp.asarray(args[:, 1, np.newaxis]), xp.asarray(args[:, 2, np.newaxis]), xp.asarray(args[:, 3, np.newaxis])
    C, D = A, B
    
    PSDS[:, :, 0] = get_SA(freq, asdTM=A, asdOMS=B, **kwargs) # A channel
    PSDS[:, :, 1] = get_SA(freq, asdTM=C, asdOMS=D, **kwargs) # E channel
    PSDS[:, :, 2] = get_ST(freq, asdTM=E, asdOMS=F, **kwargs) # T channel

    return PSDS

def compute_PSD_uncorrelated_6factors(freq, args, Ncov=3, **kwargs):
    '''
    The multiplicative factors are all independent
    '''

    PSDS = xp.empty((args.shape[0], len(freq), Ncov))

    A, B, C, D, E, F = xp.asarray(args[:, 0, np.newaxis]), xp.asarray(args[:, 1, np.newaxis]), xp.asarray(args[:, 2, np.newaxis]), xp.asarray(args[:, 3, np.newaxis]), xp.asarray(args[:, 4, np.newaxis]), xp.asarray(args[:, 5, np.newaxis])

    PSDS[:, :, 0] = get_SA(freq, asdTM=A, asdOMS=B, **kwargs) # A channel
    PSDS[:, :, 1] = get_SA(freq, asdTM=C, asdOMS=D, **kwargs) # E channel
    PSDS[:, :, 2] = get_ST(freq, asdTM=E, asdOMS=F, **kwargs) # T channel

    return PSDS

'''
----------------------------
    PSDS FUNCTIONS, splines
----------------------------
'''

# USING FEW SPLINES
# def compute_PSD_uncorrelated_splines(freq, args, PSDS_unperturbed, knots, Ncov=3):
#     '''
#     compute the spline modification to the nominal noise level
#     '''
#     assert len(PSDS_unperturbed.shape) == 3, 'the shape of the unperturbed PSD array has to be (1, Nfreq, Ncov)'    

#     PSDS = xp.empty((args.shape[0], len(freq), Ncov))

#     #args = np.atleast_2d(args)
#     args = args.reshape(-1, len(knots))
#     cs = CubicSplineInterpolant(knots, args, use_gpu=False)
#     perturbation = xp.asarray( (cs(np.log10(freq))).reshape(-1, Ncov, len(freq)) )
#     print(type(perturbation))
    
#     for i in range(Ncov):
#         PSDS[:, :, i] = xp.asarray(PSDS_unperturbed[:, :, i] * 10**(perturbation[:, i, :]))

#     return PSDS


# USING SCIPY SPLINES
# def compute_PSD_uncorrelated_splines(freq, args, PSDS_unperturbed, knots, Ncov=3):
#     '''
#     compute the spline modification to the nominal noise level
#     '''
#     assert len(PSDS_unperturbed.shape) == 3, 'the shape of the unperturbed PSD array has to be (1, Nfreq, Ncov)'    

#     PSDS = xp.empty((args.shape[0], len(freq), Ncov))

#     args = np.atleast_2d(args)
#     args = args.reshape(-1, Ncov, len(knots))

#     cs = CubicSpline(knots, args, axis=2)
#     perturbation = np.asarray((cs(np.log10(freq))))#.reshape(-1, len(freq), Ncov) )

#     # args = xp.asarray(args.reshape(-1, Ncov, len(knots)))
#     # cs = BSpline(knots, args, k=3, axis=2)
#     # #cs = CubicSpline(knots, args, axis=2)
#     # perturbation = xp.asarray( (cs(xp.log10(freq))))#.reshape(-1, len(freq), Ncov) )
    
#     for i in range(Ncov):
#         PSDS[:, :, i] = xp.asarray(PSDS_unperturbed[:, :, i] * 10**(perturbation[:, i, :]))
#         #PSDS[:, :, i] = PSDS_unperturbed[:, :, i] * 10**(perturbation[:, i, :])

#     return PSDS

# USING CUPY MAKE_INTERP_SPLINE
def compute_PSD_uncorrelated_splines(freq, args, PSDS_unperturbed, knots, Ncov=3):
    '''
    compute the spline modification to the nominal noise level
    '''
    assert len(PSDS_unperturbed.shape) == 3, 'the shape of the unperturbed PSD array has to be (1, Nfreq, Ncov)'    

    PSDS = xp.empty((args.shape[0], len(freq), Ncov))

    args = np.atleast_2d(args)
    args = args.reshape(-1, Ncov, len(knots))

    cs = make_interp_spline(knots, args, k=3, axis=2, bc_type='natural')

    perturbation = xp.asarray((cs(xp.log10(freq)))).transpose(1,0,2) # put it in the same shape of PSDS
    
    for i in range(Ncov):
        PSDS[:, :, i] = xp.asarray(PSDS_unperturbed[:, :, i] * 10**(perturbation[:, :, i]))

    return PSDS



'''
----------------------------
    PSDS FUNCTIONS, backgrounds
----------------------------
'''

def compute_PSD_signal(freq, args, **kwargs):

    noisefunc = kwargs.pop('noisefunc')
    epsfunc = kwargs.pop('epsfunc')
    nsgwbparams = kwargs.pop('nsgwbparams')
    #breakpoint()
    argsSGWB = args[:, :nsgwbparams]
    argsPSDS = args[:, nsgwbparams:]

    PSDS = noisefunc(freq, argsPSDS, **kwargs)

    h2omega = epsfunc(freq, argsSGWB)
    Sh = stochbackground(freq, h2omega)

    Sh = xp.stack((Sh, Sh, Sh)).transpose(1, 2, 0)
    #breakpoint()
    PSDS = PSDS + Sh

    return PSDS

def stochbackground(freq, h2omega):

    Sh = h2omega * 3 * H0h**2 / (4 * xp.pi**2 * freq**3)
    
    return Sh

def epssobh(freq, args):
    args = xp.atleast_2d(args)
    R = args[:, 0][:, xp.newaxis]
    n = args[:, 1][:, xp.newaxis]
    h2omega = 3.4e-13 * (R / 24) * (xp.atleast_2d(freq) / 3e-3)**n

    return h2omega


class BaseNoise:

    def __init__(self, asdTM=2.4e-15, asdOMS=7.9e-12, fkneeTM=0.4e-3, fkneeOMS=2e-3, Ncov=None, channels=None, use_gpu=False, units='hertz', splinekwargs=None):
        self.asdTM = asdTM
        self.asdOMS = asdOMS
        self.fkneeTM = fkneeTM
        self.fkneeOMS = fkneeOMS
        
        #channel selection
        self.available_channels = ['AA', 'EE', 'TT']
        available_functions = [self.get_SA, self.get_SA, self.get_ST]

        self.available_functions = dict(zip(self.available_channels, available_functions))

        if (Ncov is None) and (channels is None):
            raise ValueError('Provide either the number or the name of channels to consider')
        
        elif (Ncov is not None) and (channels is None):
            self.Ncov = Ncov
            self.channels = self.available_channels[:self.Ncov]

        elif (Ncov is None) and (channels is not None):
            self.channels = channels
            self.Ncov = len(channels)
        
        elif (Ncov is not None) and (channels is not None):
            assert Ncov == len(channels)
            self.Ncov = Ncov 
            self.channels = channels

        self.units = units

        self.use_gpu = use_gpu
        self.adjust_gpu()

        if splinekwargs is None:
            self.splinekwargs = dict(
                kind='akima',
                axis=1
            )
        else:
            assert isinstance(splinekwargs, dict), 'splinekwargs must be a dictionary containing the argument of the spine interpolant'

            self.splinekwargs = splinekwargs
        
        self.adjust_interp()

        self.PSDS_design = None

    def adjust_gpu(self):
            if self.use_gpu:    
                self.xp = xp
                self.interp = cupy_make_interp_spline
            else:
                self.xp = np
                self.interp = scipy_make_interp_spline

    def adjust_interp(self):
        
        kind = self.splinekwargs.pop('kind')

        if kind == 'cubic':
            self.interp =  CubicSplineInterpolant
            self.splinekwargs = dict(use_gpu=self.use_gpu)

        elif kind == 'bsplines':
            if self.use_gpu:    
                self.interp = cupy_make_interp_spline
            else:
                self.interp = scipy_make_interp_spline

            if 'bc_type' not in self.splinekwargs.keys():
                self.splinekwargs['bc_type'] = 'natural'
                
            if 'k' not in self.splinekwargs.keys():
                self.splinekwargs['k'] = 3

        elif kind == 'akima':
            if self.use_gpu:    
                self.interp = cupy_Akima1DInterpolator
            else:
                self.interp = scipy_Akima1DInterpolator


        else:
            raise NotImplementedError
        

    def update_params(self, asdTM=2.4e-15, asdOMS=7.9e-12):
        self.asdTM = asdTM
        self.asdOMS = asdOMS

    def oms_in_isi_carrier(self, freq):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdOMS)
        psd_meters = asd**2 * self.xp.atleast_2d(1 + (self.fkneeOMS / freq)**4)
        psd_hertz = self.xp.atleast_2d(2 * self.xp.pi * freq * CENTRAL_FREQ / C)**2 * psd_meters

        if self.units == 'hertz':
            return self.xp.sqrt(psd_hertz)
        
        elif self.units == 'meters':
            return self.xp.atleast_2d(self.xp.sqrt(psd_meters))
        
        elif self.units == 'strain':
            return self.xp.sqrt(psd_hertz / CENTRAL_FREQ**2)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')

    def filtered_oms_in_isi_carrier(self, freq):
        """
        Model for OMS noise PSD in ISI carrier beatnote fluctuations.

        Include transfer function of derivative filter instead of perfect 2 pi f
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdOMS) # m / sqrt(Hz)
        psd_meters = asd**2 * self.xp.atleast_2d(1 + (self.fkneeOMS / freq)**4)  # m^2 / Hz

        psd_hertz = self.xp.atleast_2d(self.xp.abs((-0.5 * self.xp.exp(-2j * self.xp.pi * freq * 1/FS) + 0.5 * self.xp.exp(2j * self.xp.pi * freq * 1/FS)))**2 * FS**2 * (CENTRAL_FREQ / C)**2) * psd_meters
        
        if self.units == 'hertz':
            return self.xp.sqrt(psd_hertz) 
        
        elif self.units == 'meters':
            psd_meters = psd_hertz / self.xp.atleast_2d(2 * self.xp.pi * freq * CENTRAL_FREQ / C)**2
            return self.xp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = psd_hertz / CENTRAL_FREQ**2
            return self.xp.sqrt(psd_strain)

        else:
            raise ValueError('units must be `hertz`, `meters`, or `strain`')


    def testmass_single(self, freq):
        """
        Model for single TM noise PSD including filters used in the data generation
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        asd = self.xp.atleast_2d(self.asdTM) # m / s^2 / sqrt(Hz)
        psd_acc = asd**2 * self.xp.atleast_2d(1 + (self.fkneeTM / freq)**2)  # m^2 / s^4 / Hz

        if self.units == 'hertz':
            psd_hertz = self.xp.atleast_2d(CENTRAL_FREQ / (2 * self.xp.pi * C * freq))**2 * psd_acc
            return self.xp.sqrt(psd_hertz)

        elif self.units == 'meters':
            psd_meters = self.xp.atleast_2d(2 * self.xp.pi * freq)**(-4) * psd_acc
            return self.xp.sqrt(psd_meters)
        
        elif self.units == 'strain':
            psd_strain = self.xp.atleast_2d(1 / (2 * self.xp.pi * C * freq))**2 * psd_acc
            return self.xp.sqrt(psd_strain)


        else:
            raise ValueError('units must be `hertz`, `meters` or `strain`')
        
    def filtered_testmass_single(self, freq):
        """
        Model for single TM noise PSD
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        raise NotImplementedError
        

    def tdi_common(self, freq):
        """
        TDI common factor.
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        return 16 * self.xp.sin(2 * self.xp.pi * freq * ARMLENGTH)**2 \
            * self.xp.sin(4 * self.xp.pi * freq * ARMLENGTH)**2

    def tdi_tf_oms_A(self, freq):
        """
        TDI transfer function for ISI OMS noise in TDI A,E.
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 2 * self.tdi_common(freq) * (2 + self.xp.cos(2 * xp.pi * freq * ARMLENGTH))
        return self.xp.sqrt(self.xp.atleast_2d(psd))
        
    def tdi_tf_oms_T(self, freq):
        """
        TDI transfer function for ISI OMS noise in TDI T.
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common(freq) * (1 - self.xp.cos(2 * self.xp.pi * freq * ARMLENGTH))
        return self.xp.sqrt(self.xp.atleast_2d(psd))
            
    def tdi_tf_testmass_A(self, freq):
        """
        TDI transfer function for test mass noise in TDI A,E.

        Note that we remove a factor 4 wrt. the usual expression in the literature, since we included a factor 4 in the TMI expression
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 4 * self.tdi_common(freq) * (3 + 2 * self.xp.cos(2 * self.xp.pi * freq * ARMLENGTH) + self.xp.cos(4 * self.xp.pi * freq * ARMLENGTH))
        return self.xp.sqrt(self.xp.atleast_2d(psd))


    def tdi_tf_testmass_T(self, freq):
        """
        TDI transfer function for test mass noise in TDI T.

        Note that we remove a factor 4 wrt. the usual expression in the literature, since we included a factor 4 in the TMI expression
        
        Args:
            freq (float): frequencies [Hz]
            instru (Instrument): LISA instrument object
        """
        psd = 32 * self.tdi_common(freq) * self.xp.sin(2 * self.xp.pi * freq * ARMLENGTH / 2)**4
        return self.xp.sqrt(self.xp.atleast_2d(psd))

    def testmass_A(self, freq):
        return self.tdi_tf_testmass_A(freq) * self.testmass_single(freq)

    def testmass_T(self, freq):
        return self.tdi_tf_testmass_T(freq) * self.testmass_single(freq)

    def oms_A(self, freq):
        return self.tdi_tf_oms_A(freq) * self.filtered_oms_in_isi_carrier(freq)

    def oms_T(self, freq):
        return self.tdi_tf_oms_T(freq) * self.filtered_oms_in_isi_carrier(freq)

    def get_SA(self, freq):
        '''
        Uncorrelated noise in the A, E tdi channels
        '''
        #return xp.atleast_2d(testmass_A(asd=asdTM)**2) + xp.atleast_2d(oms_A(asd=asdOMS)**2)
        return self.testmass_A(freq)**2 + self.oms_A(freq)**2

    def get_ST(self, freq):
        '''
        Uncorrelated noise in the T tdi channel
        '''
        #return xp.atleast_2d(testmass_T(asd=asdTM)**2) + xp.atleast_2d(oms_T(asd=asdOMS)**2)
        return self.testmass_T(freq)**2 + self.oms_T(freq)**2 

    def set_PSDS(self, freq):
        '''TODO: allow specific channnels'''
        self.PSDS_design = (self.xp.asarray([self.available_functions[channel](freq) for channel in self.channels])).transpose(1,2,0)

        #self.PSDS_design = (self.xp.asarray([self.get_SA(freq)[0], self.get_SA(freq)[0], self.get_ST(freq)[0]]).T)[self.xp.newaxis, : , :]

    def get_PSDS(self, freq=None, overwrite=False, **kwargs):
        if (self.PSDS_design is None):
            if freq is None:
                raise ValueError('provide frequencies')
            else:
                self.set_PSDS(freq)
        
        if overwrite:
            self.set_PSDS(freq)
        
        return self.PSDS_design
    
class TDIresponse:
        '''
        Compute a cubic spline interpolant for the TDI response in the A, E, T channels. Points used for the interpolation comes from the
        Mathematica notebook.
        The file must contain the frequencies used to compute the transfer functions and the latter for A, E, T in this exact order.
        The class supports both CPUs and GPUs.
        '''
        def __init__(self, filename, bc_type='not-a-knot', use_gpu=False):

            self.use_gpu = use_gpu
            self.adjust_gpu()

            TDIcsd_re = self.xp.transpose( self.xp.genfromtxt(filename, delimiter=','))
            self.freq = TDIcsd_re[0]
            # psd A,E and T
            self.tf_A = TDIcsd_re[1]
            self.tf_E = TDIcsd_re[2]
            self.tf_T = TDIcsd_re[3]
            
            self.tfs = self.xp.stack((self.tf_A, self.tf_E, self.tf_T)).T

            self.TDIinterpolant = self.interp(self.freq, self.tfs, axis=0, bc_type=bc_type)

        def __call__(self, freqs, return_gpu=True):
            
            response = self.TDIinterpolant(freqs)

            if (self.use_gpu) and (not return_gpu):
                return response.get() #return a numpy array

            else:
                return response #return a self.xp array


        def adjust_gpu(self):
            if self.use_gpu:    
                self.xp = xp
                self.interp = cupy_make_interp_spline
            else:
                self.xp = np
                self.interp = scipy_make_interp_spline


class psd(BaseNoise, TDIresponse):

    def __init__(self, asdTM=2.4e-15, asdOMS=7.9e-12, fmin=1e-4, fmax=2.5e-2, Ncov=None, channels=None, use_gpu=False, units='hertz', noiseless=False, splineperturbation={}, splinekwargs=None, fitASDs=False, backgrounds=None, response=None, **kwargs):
        
        if noiseless:
            asdTM = 0.
            asdOMS = 0.

        BaseNoise.__init__(self, asdTM=asdTM, asdOMS=asdOMS, Ncov=Ncov, channels=channels, use_gpu=use_gpu, units=units, splinekwargs=splinekwargs)
        self.PSDS_design = None

        self.fmin = fmin
        self.fmax = fmax

        self.logfmin = self.xp.log10(self.fmin)
        self.logfmax = self.xp.log10(self.fmax)


        self.modlist = [None, 'constants', 'splines']
        self.backlist = [None, 'sobhs', 'cs', 'fopt']
        self.available_backfuncs = {
                                        'sobhs': self.epssobh,
                                        'cs': self.epscs,
                                        'fopt': self.epsfopt
        }

        #assert splineperturbation in self.modlist
        assert isinstance(splineperturbation, dict)
        self.noiseperturbation = splineperturbation['noise']
        self.backgroundperturbation = splineperturbation['background']

        self.fitASDs = fitASDs
        self.set_noisefunc()

        if backgrounds is not None:
            if not isinstance(backgrounds, list):
                backgrounds = [backgrounds]
            for back in backgrounds:
                assert back in self.backlist

            if isinstance(response, str):
                self.responseinterp = TDIresponse(filename=response, use_gpu=use_gpu, **kwargs)
            
            elif isinstance(response, Callable):
                self.responseinterp = response
            
            else: 
                raise ValueError('if fitting for a background provide the TDI response as well. Provide either the file for the interpolation \
                                    or a callable to be evaluated on a custom range of frequencies')
    
            self.response = None #placeholder, compute it for the required frequencies
        
        self.back = backgrounds

        self.set_backfunc()

        #self.branches = [self.mod] + self.back
    
    def __call__(self, freq, noiseargs=None, backargs={}, **kwargs):
        '''
        compute the total PSD in each channel.
        
        Args:
            freq (array): array of frequencies at which the PSD is computed.
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
        #breakpoint()
        if noiseargs is not None:
            PSDS = self.noisefunc(freq, args=noiseargs, **kwargs['noise'])
        else:
            #PSDS = self.xp.zeros(shape=(backargs[self.back[0]].shape[0], len(freq), self.Ncov))
            PSDS = self.get_PSDS(freq) * self.xp.ones(backargs[self.back[0]].shape[0])[:, self.xp.newaxis, self.xp.newaxis]
            #breakpoint()
  
        if self.back is not None:

            if self.response is None:
                self.set_response(freq)
            
            sgwbs_all = self.xp.zeros_like(PSDS)

            for back in self.back:
                    
                if len(backargs[back]) > 0:
                    h2omega = self.backfunc[back](freq, backargs[back], **kwargs[back])
                    Sh = [self.stochasticbackground(freq, h2omega) for i in range(self.Ncov)]

                    Shs = self.xp.array(Sh).transpose(1, 2, 0)
                    #breakpoint()
           
                    sgwbs_all += Shs                    
            
            sgwbs_all = sgwbs_all * self.response[self.xp.newaxis, :, :]
            PSDS = PSDS + sgwbs_all
        return PSDS

    def set_noisefunc(self):

        if self.noiseperturbation:
            self.noisefunc = self.splinemod

        else:
            if self.fitASDs:
                self.noisefunc = self.constmod
            else:
                self.noisefunc = self.get_PSDS                 
               
    def set_backfunc(self):
        self.backfunc = {}
        if self.back is not None:
            for back in self.back:
                self.backfunc[back] = self.available_backfuncs[back]

    def constmod(self, freq, args, **kwargs):
        key = [*args][0]
        PSDS = self.xp.empty((args[key].shape[0], len(freq), self.Ncov))
        ASDargs = np.atleast_2d(args[key])
        asdTM, asdOMS = self.xp.asarray(ASDargs[:, 0, np.newaxis]), self.xp.asarray(ASDargs[:, 1, np.newaxis])

        self.update_params(asdTM=asdTM, asdOMS=asdOMS)    

        for i, channel in enumerate(self.channels):
            PSDS[:, :, i] = self.available_functions[channel](freq)  

        return PSDS
    

    def splinemod(self, freq, args, knots=None, ftol=0.1, **kwargs):
        '''
        args -> spline 
        ASDs -> TM and OMS ASDs, shape: (n_in, 2)
        '''
        #breakpoint()
        if self.fitASDs:
            self.PSDS_design = self.constmod(freq, args)      

        else:
            if self.PSDS_design is None:
                self.set_PSDS(freq)   

        # key = [*args][0]
        nin = min([args[key].shape[0] for key in [*args]])
        PSDS = self.xp.empty((nin, len(freq), self.Ncov))
        
        if knots is not None:
            weights = self.xp.asarray(args['splines'].reshape(-1, len(knots)))

        else:      
            '''
            RJ, provide fmin and fmax when constructing the class
            '''
            #breakpoint()
            if args['knots'].shape[0] > 0:
                idxs = np.argsort(args['knots'][:, 0])
                weights = self.xp.concatenate((self.xp.asarray(args['edges'][:,0::2]), self.xp.asarray(args['knots'][:,1:][idxs]), self.xp.asarray(args['edges'][:,1::2])), axis=0).T#weights of the knots
                knots = self.xp.hstack((self.logfmin, self.xp.array(args['knots'][:, 0][idxs]), self.logfmax))
            else:
                #breakpoint()
                weights = self.xp.concatenate((self.xp.asarray(args['edges'][:,0::2]), self.xp.asarray(args['edges'][:,1::2])), axis=0).T#weights of the knots
                knots = self.xp.hstack((self.logfmin, self.logfmax))
            #print(self.xp.diff(knots))
            #breakpoint()
            if self.xp.any(self.xp.abs(self.xp.diff(knots)) < ftol):            
                PSDS[:, :, :] = self.xp.nan
                return PSDS

        perturbation = self.perturbation(freq=freq, knots=knots, weights=weights)
        
        for i in range(self.Ncov):
            PSDS[:, :, i] = self.PSDS_design[:, :, i] * 10**(perturbation[:, :, i])

        return PSDS
    
    def perturbation(self, freq, knots, weights):
        '''
        Spline perturbation.
        '''
        interp = self.interp(knots, weights, **self.splinekwargs)
        perturbation = (interp(self.xp.log10(freq))).reshape(-1, len(freq), self.Ncov) #put it in the same shape of PSDS
        
        return perturbation
    
#-------------------------

    def tmp_splinemod(self, freq, args, knots=None, bc_type='natural', ftol=0.1, **kwargs):
        '''
        args -> spline 
        ASDs -> TM and OMS ASDs, shape: (n_in, 2)
        '''
        #breakpoint()
        if self.fitASDs:
            self.PSDS_design = self.constmod(freq, args)      

        else:
            if self.PSDS_design is None:
                self.set_PSDS(freq)   

        key = [*args][0]
        PSDS = self.xp.empty((args[key].shape[0], len(freq), self.Ncov))
        
        if knots is not None:
            weights = self.xp.asarray(args['splines'].reshape(-1, self.Ncov, len(knots)))

        else:      
            '''
            RJ, provide fmin and fmax when constructing the class
            '''
            #breakpoint()
            idxs = np.argsort(args['knots'][:, 0])
            weights = self.xp.concatenate((self.xp.asarray(args['edges'][:,0::2]), self.xp.asarray(args['knots'][:,1:][idxs]), self.xp.asarray(args['edges'][:,1::2])), axis=0).T[self.xp.newaxis, :, :] #weights of the knots
            knots = self.xp.hstack((self.logfmin, self.xp.array(args['knots'][:, 0][idxs]), self.logfmax))
            #print(self.xp.diff(knots))
            #breakpoint()
            if self.xp.any(self.xp.abs(self.xp.diff(knots)) < ftol):            
                PSDS[:, :, :] = self.xp.nan
                return PSDS

        perturbation = self.perturbation(freq=freq, knots=knots, weights=weights, bc_type=bc_type)
        
        for i in range(self.Ncov):
            PSDS[:, :, i] = self.PSDS_design[:, :, i] * 10**(perturbation[:, :, i])

        return PSDS
    
    def tmp_perturbation(self, freq, knots, weights, bc_type='natural'):
        '''
        Cubic spline perturbation.
        '''
        cs = self.interp(knots, weights, k=3, axis=2, bc_type=bc_type)
        perturbation = (cs(self.xp.log10(freq))).transpose(1,0,2) #put it in the same shape of PSDS
        
        return perturbation
    
    def set_response(self, freq):
        '''
        Set the response for the TDI channels selected (only works with A, E, and T).
        '''
        idxs = [self.available_channels.index(channel) for channel in self.channels]
        self.response = self.responseinterp(freq)[:, idxs]

    def get_response(self, freq):
        '''
        Get the response for the TDI channels selected (only works with A, E, and T).
        '''
        if self.response is None:
            self.set_response(freq)
        
        return self.response
    
    def stochasticbackground(self, freq, h2omega):
        '''
        NOTE: including here a factor 2 \pi 
        '''
        Sh = h2omega * (3 * H0h**2 / (4 * self.xp.pi**2 * freq**3)) * (2 * self.xp.pi) #strain units

        # unit conversions
        if self.units == 'hertz':
            return Sh * CENTRAL_FREQ**2
        
        elif self.units == 'meters':
            return Sh / ( (2 * self.xp.pi * freq) / C)**2 
        
        elif self.units == 'strain':
            return Sh
    

    def epssobh(self, freq, args, **kwargs):

        R = self.xp.array(args[:, 0])[:, self.xp.newaxis]
        n = self.xp.array(args[:, 1])[:, self.xp.newaxis]
        fp = 3e-3
        h2omega = 3.4e-13 * (R / 24.0) * (freq[self.xp.newaxis, :] / fp)**n

        return h2omega
    
    def epscs(self, freq, args, **kwargs):

        Gmu = self.xp.array(args[:, 0])[:, self.xp.newaxis]
        n = self.xp.array(args[:, 1])[:, self.xp.newaxis]
        fp = 3e-3
        h2omega = 0.55e-11 * (Gmu / 1e-13)**0.5 * (freq[self.xp.newaxis, :] / fp)**n

        return h2omega
    
    def epsfopt(self, freq, args, **kwargs):
        '''
        Device-agnostic implementation of the equations contained in https://bitbucket.org/dweir/ptplot/src/main/ptplot/science/calculate_powerspectrum.py
        '''

        Asw = self.xp.array(args[:, 0])[:, self.xp.newaxis]
        fsw = self.xp.array(args[:, 1])[:, self.xp.newaxis]
        Aturb = self.xp.array(args[:, 2])[:, self.xp.newaxis]
        Tstar = self.xp.array(args[:, 3])[:, self.xp.newaxis]
        
        #gstar = self.xp.array(args[:, 4])[:, self.xp.newaxis]
        gstar = 1.e2
        zp = 10.
        #adiabaticratio = 4.0 / 3.0
        #kturb = 1.97/65.0 

        fturb = self.fturb_from_sw(fsw, zp)

        h2omega_sw = self.epsfopt_sw(freq, Asw, fsw)
        h2omega_turb = self.epsfopt_turb(freq, Aturb, fturb, Tstar, gstar)

        h2omega = h2omega_sw + h2omega_turb

        return h2omega
    
    def fturb_from_sw(self, fsw, zp=10):
        fturb = 27 / 26 * (8*self.xp.pi)**(1/3) * (10 / zp) * fsw
        return fturb
    
    def hstar(self, Tstar, gstar):
        return 165e-7 * (Tstar / 1e2) * (gstar / 1e2)**(1./6.)

    def fsw(self, Hrstar, zp, Tstar, gstar):
        return 26.e-6 * (1. / Hrstar) * (zp / 10.) * (Tstar / 1e2) * (gstar / 1e2)**(1./6.)

    def Csw(self, fp, norm=1.):
        return norm * fp**3. * (7. / (4. + 3.*fp**2.))**(7./2.) 

    def epsfopt_sw(self, freq, Asw, fsw):
        '''
        From ArXiv:1704.05871 (and the paper erratum)
        '''
        #breakpoint()
        fp = freq / fsw
        h2omega = Asw * self.Csw(fp)
        return h2omega
        #return 3. * h_planck**2. * 0.687*3.57e-5*0.012 * (1.e2 / gstar)**(1./3.) * KK**2. * Hrstar * self.Csw(fp)
        #return 3. * h_planck**2. * 0.687*3.57e-5*0.012 * (1.e2 / gstar)**(1./3.) * adiabaticratio**2. * ubarf2**2. * Hrstar * self.Csw(fp)

    def fturb(self, Hrstar, Tstar, gstar):
        '''
        From ArXiv:1512.06239
        '''
        return 27e-6 * (8*self.xp.pi)**(1/3) / Hrstar * (Tstar / 1e2) * (gstar / 1e2)**(1./6.) #Hz
    
    def Sturb(self, freq, fturb=None, fsw=None, Hrstar=None, Tstar=None, gstar=None, zp=10):
        '''
        From ArXiv:1512.06239
        '''
        if (fturb is None) and (fsw is None):
            fturb = self.fturb(Hrstar, Tstar, gstar)

        elif (fturb is None) and (fsw is not None):
            fturb = self.fturb_from_sw(fsw, zp)

        fp = freq / fturb
        hstar = self.hstar(Tstar, gstar)

        return fp**3 / ( (1 + fp)**(11/3) * (1 + 8 * self.xp.pi * freq / hstar) )
    
    def Sturb_norm(self, freq, fturb=None, fsw=None, Hrstar=None, Tstar=None, gstar=None, zp=10):
        '''
        From ArXiv:1512.06239, I remove a term hstar from here to include it in the powerlaw amplitude
        '''
        if (fturb is None) and (fsw is None):
            fturb = self.fturb(Hrstar, Tstar, gstar)

        elif (fturb is None) and (fsw is not None):
            fturb = self.fturb_from_sw(fsw, zp)

        fp = freq / fturb
        hstar = self.hstar(Tstar, gstar)

        return fp**3 / ( (1 + fp)**(11/3) * (hstar + 8 * self.xp.pi * freq) )
    
    def epsfopt_turb(self, freq, Aturb, fturb, Tstar, gstar):
        '''
        From ArXiv:1512.06239
        '''
        h2omega = Aturb * self.Sturb_norm(freq=freq, fturb=fturb, Tstar=Tstar, gstar=gstar)
        #h2omega = 335e-6 / (8*self.xp.pi)**(1/3) * Hrstar * ( kturb*alpha / (1+alpha) )**(3/2) * (1e2 / gstar)**(1/3) * self.Sturb(freq, Hrstar, Tstar, gstar)

        return h2omega
    

#----
    def ubarf2(self, vw, alpha, adiabaticratio):
        '''
        From J. R. Espinosa et al. JCAP06 (2010) 028 (arXiv:1004.4187)
        '''
        #breakpoint()
        return (1.0 / adiabaticratio) * self.kappav(vw, alpha) * alpha / (1.0 + alpha)
    
    def kappav(self, vw, alpha):
        '''
        From J. R. Espinosa et al. JCAP06 (2010) 028 (arXiv:1004.4187)
        '''
        
        kappaA = 6.9 * alpha * vw**(6./5.) / (1.36 - 0.037*self.xp.sqrt(alpha) + alpha)
        kappaB = alpha**(2./5.) / ( 0.017 + (0.997 + alpha)**(2./5.) )
        kappaC = self.xp.sqrt(alpha) / (0.135 + self.xp.sqrt(0.98 + alpha))
        kappaD = alpha / (0.73 + 0.083 * self.xp.sqrt(alpha) + alpha)

        cs = 1./3.**(0.5)
        xiJ = (self.xp.sqrt((2./3.)*alpha + alpha**2) + self.xp.sqrt(1./3.)) / (1 + alpha)
        deltaK = -0.9 * self.xp.log( (self.xp.sqrt(alpha) / (1 + self.xp.sqrt(alpha))) )

        output = self.xp.empty_like(vw)
        output[:] = self.xp.nan

        mask = (vw < cs)
        output[mask] = cs**(11./5.) * kappaA[mask] * kappaB[mask] / (( cs**(11./5.) - vw[mask]**(11./5.))*kappaB[mask] + kappaA[mask]*vw[mask]*cs**(6./5.) )

        mask = (vw > xiJ)
        output[mask] = (xiJ[mask] - 1.)**3. * xiJ[mask]**(5./2.) * vw[mask]**(-5./2.) * kappaC[mask] * kappaD[mask] / (((xiJ[mask] -1.)**3. - (vw[mask] - 1.)**3.) * xiJ[mask]**(5./2.) * kappaC[mask] + (vw[mask] - 1)**3. * kappaD[mask])
        
        mask = (vw > cs) & (vw < xiJ)
        output[mask] = kappaB[mask] + (vw[mask] - cs)*deltaK[mask] + ( ((vw[mask] - cs)**3.) / ((xiJ[mask] - cs)**3.) * (kappaC[mask] - kappaB[mask] - (xiJ[mask] - cs) * deltaK[mask]) )

        if self.xp.any(self.xp.isnan(output)):
            raise ValueError
        
        return output
        # if vw < cs:
        #     return cs**(11./5.) * kappaA * kappaB / ( cs**(11./5.) - kappaB*vw**(11./5.) + kappaA*vw*cs**(6./5.) )
        
        # elif vw > xiJ:
        #     return (xiJ - 1.)**3. * xiJ**(5./2.) * vw**(-5./2.) * kappaC * kappaD / (((xiJ -1.)**3. - (vw - 1.)**3.) * xiJ**(5./2.) * kappaC + (vw - 1)**3. * kappaD)
        
        # else:
        #     return kappaB + (vw - cs)*deltaK + ( ((vw - cs)**3.) / ((xiJ - cs)**3.) * (kappaC - kappaB - (xiJ - cs) * deltaK) )

    def KK(self, alpha):
        kappav1 = alpha / (0.73 + 0.083*self.xp.sqrt(alpha) + alpha)
        return kappav1 * alpha / (1 + alpha) 


    def Hrstar(self, beta_H, vw):
        return (8 * self.xp.pi)**(1/3) * vw / beta_H