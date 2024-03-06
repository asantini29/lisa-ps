# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
try:
    import cupy as xp
except:
    import numpy as xp

class Likelihood:

    def __init__(self, 
                    compute_psd, 
                    t=None,
                    d=None,
                    freqs=None,
                    dtilde=None,
                    fmin=1e-4,
                    fmax=2.5e-2,
                    source_wf_gen=None,
                    nchannels=3,
                    average=False,
                    Nbins=1000,
                    window=('kaiser', 30),
                    noisekeys=[],
                    backgroundkeys=[],
                    foregroundkeys=[],
                    inf=1e14,
                    use_gpu=True,
                    return_gpu=False,
                    nsubset=1,
                    rj=False,
                    **kwargs
                    ):
        '''
        TODO: write docstring
        '''

        self.use_gpu = use_gpu
        self.xp = xp if self.use_gpu else np
        self.return_gpu = return_gpu

        self.nchannels = nchannels
        self.fmin = fmin
        self.fmax = fmax

        if (t is None) and (freqs is None):
            raise ValueError('Provide either the times or frequencies')

        if (d is None) and (dtilde is None):
            raise ValueError('Provide the data in either the time or frequency domain')
        
        if (t is not None) and (freqs is None) and (d is not None) and (dtilde is None):

            assert (d.shape[0] == t.shape[0]) and (d.shape[1] == self.nchannels), 'Dimensionality mismatch'
            self.d = d
            self.window = signal.get_window(window, d.shape[0]) if window is not None else np.ones_like(t)
            self.Nbw = d.shape[0] * np.sum(self.window**2) / np.sum(self.window)**2
            
            self.dt = t[1] - t[0]
            freqs = np.fft.rfftfreq(d.shape[0], self.dt)
            self.frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL: we MUST find a way to include them

            freqs = self.xp.array(freqs[self.frequencymask])

            self.dtilde = self.get_Xtilde(d)

        if source_wf_gen is None: # fit only for the psd (noise, stochastic components)
            
            self.nsource_wf_gen = 0

            if average: # without signal we can use an averaged likelihood
                self.freqs, self.dtildedtilde, self.nu = self.average(freqs, Nbins)
            else:
                self.freqs = freqs
                self.dtildedtilde = self.get_XtildeXtilde(dtilde)
                self.nu = 1

        else:                
                
            if not isinstance(source_wf_gen, list):
                source_wf_gen = [source_wf_gen]

            self.nsource_wf_gen = len(source_wf_gen)

        self.compute_psd = compute_psd

        self.noisekeys = noisekeys
        self.backgroundkeys = backgroundkeys
        self.foregroundkeys = foregroundkeys

        self.setup_indeces()

        self.inf = inf

        self.rj = rj
        if self.rj:
            self.nsubset = 1
        else:     
            self.nsubset = nsubset

    
    def __call__(self, args, groups=None, tc_container=None,**kwargs):
        '''
        TODO: 
        2) what if groups are not provided?
        3) check vectorization
        4) check factors in front
        5) add response for individual sources

        The order that `args` has to follow is [(templates), (noise), (backgrounds), (foregrounds)]
        '''

        if not isinstance(args, list):
            args = [args]

        if groups is None:
            groups = [np.arange(np.atleast_2d(args[0]).shape[0])]

        if not isinstance(groups, list):
            groups = [groups]

        unique_groups = np.unique(np.concatenate([groups_i for groups_i in groups]))
        ngroups = unique_groups.max() + 1 if unique_groups.shape[0] > 0 else 0


        wf_args_all, noise_args_all, background_args_all, foreground_args_all = self.unpack_args(args)
        
        logl_all = []
        
        #breakpoint()

        subset = int(ngroups / self.nsubset)
        #  hardcoded for the moment
        if self.rj:
            subset = 1

        inds_all = np.arange(0, ngroups+1, subset)
        if inds_all[-1] < ngroups:
            inds_all = np.concatenate([inds_all, np.array([ngroups])])

        for i in range(len(inds_all) - 1):

            wf_args, noise_args, background_args, foreground_args = [], [], [], []

            for j in range(self.nsource_wf_gen):
                inds = np.where((groups[j] >= inds_all[i]) & (groups[j] < inds_all[i + 1]))
                wf_args += [wf_args_all[j][inds]]

            for j in range(len(self.noisekeys)):
                inds = np.where((groups[self.idx_noise + j] >= inds_all[i]) & (groups[self.idx_noise + j] < inds_all[i + 1]))
                noise_args += [noise_args_all[j][inds]]

            for j in range(len(self.backgroundkeys)):
                inds = np.where((groups[self.idx_background + j] >= inds_all[i]) & (groups[self.idx_background +j] < inds_all[i + 1]))
                background_args += [background_args_all[j][inds]]
            
            for j in range(len(self.foregroundkeys)):
                inds = np.where((groups[self.idx_foreground + j] >= inds_all[i]) & (groups[self.idx_foreground +j] < inds_all[i + 1]))
                foreground_args += [foreground_args_all[j][inds]]
        
            psd = self.compute_psd(self.freqs, noise_args, background_args, foreground_args, **kwargs)

            if self.nsource_wf_gen > 0:
                h = self.xp.zeros(shape=(self.freqs[0]))

                for source, wf_args_i in zip(self.source_wf_gen, wf_args):
                        
                        h += source(wf_args_i)

                n = self.d - h
                ntilde = self.get_Xtilde(n)
                ntildentilde = self.get_XtildeXtilde(ntilde)

            else:
                ntildentilde = self.dtildedtilde

            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()

            logl = - self.xp.sum( self.xp.sum(ntildentilde / psd, axis = -1) + self.nu * xp.sum(self.xp.log(psd), axis = -1) , axis = -1)
            logl_all.append(logl)

        logl_out = np.concatenate(logl_all)
        logl_out[~np.isfinite(logl_out)] = -self.inf

        if not self.return_gpu:
            return logl_out.get()

        else:
            return logl_out
        
    def setup_indeces(self):
        self.idx_wf = 0
        self.idx_noise = self.nsource_wf_gen
        self.idx_background = self.idx_noise + len(self.noisekeys)
        self.idx_foreground = self.idx_background + len(self.backgroundkeys)
        self.indeces = [self.idx_wf, self.idx_noise, self.idx_background, self.idx_foreground]
        
    def unpack_args(self, args):
        wf_args, noise_args, background_args, foreground_args = [], [], [], []
        components = [wf_args, noise_args, background_args, foreground_args]
        indeces = self.indeces + [len(args)]
        
        for i in range(len(components)):
            components[i] += args[indeces[i] : indeces[i+1]]

        return wf_args, noise_args, background_args, foreground_args
        

    def get_Xtilde(self, d=None):
        if d is None:
            d = self.d
        Xtilde = self.xp.asarray([np.fft.rfft(d[:, i] * self.window)[self.frequencymask] * np.sqrt(2 * self.dt / np.sum(self.window**2)) for i in range(self.nchannels)]).T #ALREADY NORMALIZED, refer to arXiv:2302.12573
        return Xtilde

    
    def get_XtildeXtilde(self, dtilde=None):
        if dtilde is None:
            dtilde = self.dtilde
        return xp.real(xp.conj(dtilde) * dtilde)[xp.newaxis, :, :] #vectorized over axis 0


    def average(self, freqs, Nbins):

        dtildedtilde = self.get_XtildeXtilde()
        if freqs.shape[0] // 2 != 0:
            freqs = freqs[1:]
            dtildedtilde = dtildedtilde[:, 1:, :]

        edges = self.xp.linspace(freqs.min(), freqs.max(), Nbins + 1, endpoint=True)
        centers = self.xp.zeros(Nbins)
        nu = self.xp.zeros(Nbins)       
        Y = self.xp.zeros(shape=(1, Nbins, self.nchannels))

        for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (freqs >= start) & (freqs < stop)
            centers[i] = self.xp.median(freqs[mask])
            nu[i] = len(freqs[mask]) / self.Nbw
 
            Y[:, i, :] = self.xp.mean(dtildedtilde[:, mask, :], axis = 1) * nu[i] # eq 29 in arXiv:2302.12573

        return centers, Y, nu
    

            


