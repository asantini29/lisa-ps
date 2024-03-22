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
                    correlated=False,
                    fullmatrix=False,
                    Nbins=1000,
                    window=('kaiser', 30),
                    Nbw=None,
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

            self.Nbw = Nbw if Nbw is not None else d.shape[0] * np.sum(self.window**2) / np.sum(self.window)**2
            
            self.dt = t[1] - t[0]
            freqs = np.fft.rfftfreq(d.shape[0], self.dt)
            #freqs = np.fft.fftfreq(d.shape[0], self.dt)
            self.frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL: we MUST find a way to include them

            freqs = self.xp.array(freqs[self.frequencymask])

            self.dtilde = self.get_Xtilde(d)

        if source_wf_gen is None: # fit only for the psd (noise, stochastic components)
            
            self.nsource_wf_gen = 0

            self.correlated = correlated
            self.fullmatrix = fullmatrix

            if average: # without signal we can use an averaged likelihood
                self.freqs, self.Y, self.nu = self.average(freqs, Nbins)
                self.compute_logl = self.wishart_logl
            else:
                self.freqs = freqs
                self.dtildedtilde =self.get_XtildeXtilde(dtilde)
                self.nu = 1
                self.compute_logl = self.whittle_logl

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

        @property
        def tc_container(self):
            return self._tc_container
        
        @tc_container.setter
        def tc_container(self, tc_container=[None, None, None, None]):
            self._tc_container = tc_container

    
    def __call__(self, args, groups=None, tc_container=None,**kwargs):
        '''
        TODO 
        -) check vectorization
        -) complete custom CUDA Kernel for spline interpolation
        -) check factors in front
        -) add response for individual sources

        #* The order that `args` has to follow is [(templates), (noise), (backgrounds), (foregrounds)]
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
        wf_groups_all, noise_groups_all, background_groups_all, foreground_groups_all = self.unpack_groups(groups)
        
        logl_all = []
        
        #breakpoint()

        subset = int(ngroups / self.nsubset) if ngroups > self.nsubset else ngroups
        
        try:
            inds_all = np.arange(0, ngroups+1, subset)
        except:
            breakpoint()
        
        if inds_all[-1] < ngroups:
            inds_all = np.concatenate([inds_all, np.array([ngroups])])

        for i in range(len(inds_all) - 1):

            wf_args, noise_args, background_args, foreground_args = [], [], [], []
            wf_groups, noise_groups, background_groups, foreground_groups = [], [], [], []

            for j in range(self.nsource_wf_gen):
                #inds = np.where((groups[j] >= inds_all[i]) & (groups[j] < inds_all[i + 1]))
                inds = np.where((wf_groups_all[j] >= inds_all[i]) & (wf_groups_all[j] < inds_all[i + 1]))
                wf_args += [wf_args_all[j][inds]]
                wf_groups += [wf_groups_all[j][inds]]

            for j in range(len(self.noisekeys)):
                inds = np.where((noise_groups_all[j] >= inds_all[i]) & (noise_groups_all[j] < inds_all[i + 1]))
                noise_args += [noise_args_all[j][inds]]
                noise_groups += [noise_groups_all[j][inds]]

            for j in range(len(self.backgroundkeys)):
                inds = np.where((background_groups_all[j] >= inds_all[i]) & (background_groups_all[j] < inds_all[i + 1]))
                background_args += [background_args_all[j][inds]]
                background_groups += [background_groups_all[j][inds]]
            
            for j in range(len(self.foregroundkeys)):
                inds = np.where((foreground_groups_all[j] >= inds_all[i]) & (foreground_groups_all[j] < inds_all[i + 1]))
                foreground_args += [foreground_args_all[j][inds]]
                foreground_groups += [foreground_groups_all[j][inds]]
        
            psd = self.compute_psd(self.freqs, 
                                   noise_args, 
                                   background_args, 
                                   foreground_args, 
                                   noise_groups,
                                   background_groups,
                                   foreground_groups,
                                   **kwargs)

            if self.nsource_wf_gen > 0:
                h = self.xp.zeros(shape=(self.freqs[0]))

                for source, wf_args_i in zip(self.source_wf_gen, wf_args):
                        
                    h += source(wf_args_i)

                n = self.d - h
                ntilde = self.get_Xtilde(n)
                ntildentilde = self.get_XtildeXtilde(ntilde)

            else:
                ntilde = self.dtilde[self.xp.newaxis, :, :]

            mempool = xp.get_default_memory_pool()
            mempool.free_all_blocks()

            logl = self.compute_logl(ntilde, psd)
            #logl = - self.xp.sum( self.xp.sum(ntildentilde / cov, axis = -1) + self.nu * xp.sum(self.xp.log(cov), axis = -1) , axis = -1)

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
        """
        Unpacks the arguments into separate components.

        Args:
            args (list): The list of arguments to be unpacked.

        Returns:
            tuple: A tuple containing the unpacked components: wf_args, noise_args, background_args, foreground_args.
        """
        wf_args, noise_args, background_args, foreground_args = [], [], [], []
        components = [wf_args, noise_args, background_args, foreground_args]
        indeces = self.indeces + [len(args)]
        #breakpoint()
        
        for i in range(len(components)):
            if self.tc_container[i] is not None:
                components[i] += [self.tc_container[i][j].transform_base_parameters(arg) for j,arg in enumerate(args[indeces[i] : indeces[i+1]])]
            else:
                components[i] += args[indeces[i] : indeces[i+1]]

        return wf_args, noise_args, background_args, foreground_args
    
    def unpack_groups(self, groups):
        wf_groups, noise_groups, background_groups, foreground_groups = [], [], [], []
        components = [wf_groups, noise_groups, background_groups, foreground_groups]
        indeces = self.indeces + [len(groups)]
        #breakpoint()
        
        for i in range(len(components)):
            components[i] += groups[indeces[i] : indeces[i+1]]

        return wf_groups, noise_groups, background_groups, foreground_groups
        

    def get_Xtilde(self, d=None):
        if d is None:
            d = self.d
        norm = 2
        Xtilde = self.xp.asarray([np.fft.rfft(d[:, i] * self.window)[self.frequencymask] for i in range(self.nchannels)]).T * np.sqrt(norm * self.dt / np.sum(self.window**2)) #ALREADY NORMALIZED, refer to arXiv:2302.12573
        return Xtilde

    
    def get_XtildeXtilde(self, dtilde=None):
        if dtilde is None:
            dtilde = self.dtilde
        if self.fullmatrix:
            return self.xp.einsum('...i,...j->...ij', self.xp.conj(dtilde), dtilde)[self.xp.newaxis, :, :, :] #vectorized over axis 0
            #return self.xp.real(self.xp.einsum('...i,...j->...ij', self.xp.conj(dtilde), dtilde))[self.xp.newaxis, :, :, :] #vectorized over axis 0
        else:
            return  self.xp.abs(self.xp.conj(dtilde) * dtilde)[self.xp.newaxis, :, :] #vectorized over axis 0
            #return self.xp.real(self.xp.conj(dtilde) * dtilde)[self.xp.newaxis, :, :] #vectorized over axis 0


    def average(self, freqs, Nbins):

        dtildedtilde = self.get_XtildeXtilde()
        #? not sure why I'm doing this, have to check 
        if freqs.shape[0] // 2 != 0:
            freqs = freqs[1:]
            dtildedtilde = dtildedtilde[:, 1:]

        edges = self.xp.linspace(freqs.min(), freqs.max(), Nbins + 1, endpoint=True) #edges of frequency bins
        #edges = self.xp.logspace(np.log10(freqs.min()), np.log10(freqs.max()), Nbins + 1, endpoint=True) #edges of frequency bins
        centers = self.xp.zeros(Nbins) #centers of frequency bins
        nu = self.xp.zeros(Nbins) #effective DoFs

        periodgram_shape = (1, Nbins,) + dtildedtilde.shape[2:]
        Y = self.xp.zeros(shape=periodgram_shape)

        for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (freqs >= start) & (freqs < stop)
            centers[i] = self.xp.median(freqs[mask])

            nu[i] = np.count_nonzero(freqs[mask]) / self.Nbw
 
            Y[:, i] = self.xp.mean(dtildedtilde[:, mask], axis = 1) * nu[i] # eq 29 in arXiv:2302.12573

        return centers, Y, nu
    

    def whittle_logl(self, ntilde, psd):
        
        if self.fullmatrix:
            cov = self.get_covariance(psd)
            ntilde_rep = self.xp.repeat(ntilde, cov.shape[0], axis=0)
            ntildeconj_invcov = self.xp.linalg.solve(cov, self.xp.conj(ntilde_rep)[:, :, :])
            detcov = self.xp.linalg.det(cov)
            del cov
            ntildentilde = self.xp.einsum('ijk,ijk -> ij', ntildeconj_invcov, ntilde)
            logl = - self.xp.sum(ntildentilde + self.xp.log(detcov), axis=-1)

        else:
            cov = psd
            ntildentilde = self.get_XtildeXtilde()
            logl = - self.xp.sum( ntildentilde / cov + self.xp.log(cov),  axis = (1, 2))

        return logl
    
    def wishart_logl(self, ntilde, psd):

        if self.fullmatrix:
            cov = self.get_covariance(psd)
            invcov = self.xp.linalg.inv(cov)
            detcov = self.xp.linalg.det(cov)
            del cov

            return -  self.xp.sum(self.xp.einsum('...ii', self.xp.einsum('...ij, ...jk->...ik', invcov, self.Y)) + self.nu * self.xp.log(detcov), axis=-1)

        else:
            cov = psd
            return -  self.xp.sum(self.xp.sum(self.Y / cov, axis=-1) + self.nu * self.xp.sum(self.xp.log(cov), axis=-1), axis=-1)

    
    def get_covariance(self, psd):
        nin, nfreqs = psd.shape[0], psd.shape[1]
        covariance = self.xp.zeros(shape=(nin, nfreqs, self.nchannels, self.nchannels))

        for i in range(self.nchannels):
            covariance[:,:,i,i] = psd[:,:,i]

        if self.correlated:
            covariance[:,:,0,1] = psd[:,:,3]  
            covariance[:,:,0,2] = psd[:,:,4]  
            covariance[:,:,1,2] = psd[:,:,5]

            covariance[:,:,1,0] = self.xp.conj(psd[:,:,3])
            covariance[:,:,2,0] = self.xp.conj(psd[:,:,4])
            covariance[:,:,2,1] = self.xp.conj(psd[:,:,5])

        return covariance

        


            


