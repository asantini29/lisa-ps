# -*- coding: utf-8 -*-

class likelihood:

    def __init__(self, rj=False, average=False, nu=1):
        self._rj = rj
        self._average = average
        if not self._average:
            assert nu == 1
        self._nu = nu

    