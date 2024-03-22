# -*- coding: utf-8 -*-

from .baseclasses import GPUobject
import numpy as np
try:
    import cupy as xp
except:
    import numpy as xp

from .constants import YRSID_SI

#credits: Michael Katz, https://github.com/mikekatz04/lisa-on-gpu/blob/master/examples/fast_LISA_response_tutorial.ipynb
class GBWave(GPUobject):
    def __init__(self, use_gpu=False):

        super.__init__(self, use_gpu=use_gpu)

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):

        # get the t array 
        t = self.xp.arange(0.0, T * YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc

