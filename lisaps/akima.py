# -*- coding: utf-8 -*-
try:
    import cupy as xp
    from numba import cuda
    cuda_available = True

except (ModuleNotFoundError, ImportError):
    import numpy as xp
    import numba
    cuda_available = False

import numpy as np
import math

if cuda_available:
    @cuda.jit(device=True)
    def linearslope(x, y, idx):
        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        m = dy / dx
        return m

    @cuda.jit(device=True)
    def splineslope(x, y, idx, start, stop):

        #! with these boundary conditions I ALWAYS NEED AT LEAST FOUR POINTS
        #? mi = linearslope(x, y, idx) #i
        #? mi1 = linearslope(x, y, idx+1) #i+1 

        #boundaries conditions
        if idx == start:
            return (3*linearslope(x, y, idx) - linearslope(x, y, idx+1)) / 2
        
        elif (idx == start + 1):
            #? mi_1 = linearslope(x, y, idx-1)
            return (abs(linearslope(x, y, idx+1) - linearslope(x, y, idx)) * linearslope(x, y, idx-1) \
            + abs(linearslope(x, y, idx) - linearslope(x, y, idx-1)) * linearslope(x, y, idx)) \
            / (abs(linearslope(x, y, idx+1) - linearslope(x, y, idx)) + abs(linearslope(x, y, idx) - linearslope(x, y, idx-1)))

        elif (idx == stop - 2):
            #? mi_1 = linearslope(x, y, idx-1)
            #? mi_2 = linearslope(x, y, idx-2) # i-2
            return (abs(linearslope(x, y, idx-2) - linearslope(x, y, idx-1)) * linearslope(x, y, idx)  \
            + abs(linearslope(x, y, idx-1) - linearslope(x, y, idx)) * linearslope(x, y, idx-1)) \
            / (abs(linearslope(x, y, idx-2) - linearslope(x, y, idx-1)) + abs(linearslope(x, y, idx-1) - linearslope(x, y, idx)))

        elif idx == stop - 1:
            #? mi_1 = linearslope(x, y, idx-1)
            #? mi_2 = linearslope(x, y, idx-2) # i-2
            return (3*linearslope(x, y, idx-1) - linearslope(x, y, idx-2)) / 2
        
        mi_2 = linearslope(x, y, idx-2) # i-2
        mi_1 = linearslope(x, y, idx-1) # i-1
        mi = linearslope(x, y, idx) #i
        mi1 = linearslope(x, y, idx+1) #i+1 
                
        thr = max(abs(mi1 - mi), abs(mi - mi_1), abs(mi_1 - mi_2), -1e99) * 1e-9

        if (abs(mi - mi1) + abs(mi_2 - mi_1)) < thr:
            return (mi_1 + mi) / 2
        else:
            return (abs(mi1 - mi) * mi_1 + abs(mi_1 - mi_2) * mi) / ( abs(mi1 - mi) + abs(mi_1 - mi_2) )
        

    @cuda.jit
    def akima_spline_kernel(x_new, x, y, n_in, ngroups, nnans, result):
        start1 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        increment1 = cuda.blockDim.x * cuda.gridDim.x

        start2 = cuda.blockIdx.y
        increment2 = cuda.gridDim.y

        for j in range(start2, n_in, increment2):
            
            for i in range(start1, x_new.shape[0], increment1):

                special_index = j * x_new.shape[0] + i
                idx = -1
                stop = (j+1)*ngroups - nnans[j] #! stop iterating when encounters NaNs

                if x_new[i] == x[stop - 1]: #* deal with extrema
                    result[special_index] = y[stop - 1] 
                
                else:
                    for k in range(j*ngroups, (j+1)*(ngroups) - 1):
                        
                        if (x_new[i] >= x[k] and x_new[i] < x[k + 1]):
                            idx = k
                            break
                    
                    #* Akima spline interpolation 
                    mi = linearslope(x, y, idx)
                    si = splineslope(x, y, idx, j*ngroups, stop)
                    si1 = splineslope(x, y, idx+1, j*ngroups, stop)
                
                    p0 = y[idx]

                    p1 = si

                    p2 = (3*mi - 2*si - si1) / (x[idx+1] - x[idx])

                    p3 = (si + si1 - 2*mi) / (x[idx+1] - x[idx])**2

                    result[special_index] = p0 + p1 * (x_new[i] - x[idx]) + p2 * (x_new[i] - x[idx])**2 + p3 * (x_new[i] - x[idx])**3


else:
    print("running on CPUs")

    @numba.jit(nopython=True)
    def linearslope(x, y, idx):
        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        m = dy / dx
        return m
    
    @numba.jit(nopython=True)
    def splineslope(x, y, idx, start, stop):

        #! with these boundary conditions I ALWAYS NEED AT LEAST FOUR POINTS
        #? mi = linearslope(x, y, idx) #i
        #? mi1 = linearslope(x, y, idx+1) #i+1 

        #boundaries conditions
        if idx == start:
            return (3*linearslope(x, y, idx) - linearslope(x, y, idx+1)) / 2
        
        elif (idx == start + 1):
            #? mi_1 = linearslope(x, y, idx-1)
            return (abs(linearslope(x, y, idx+1) - linearslope(x, y, idx)) * linearslope(x, y, idx-1) \
            + abs(linearslope(x, y, idx) - linearslope(x, y, idx-1)) * linearslope(x, y, idx)) \
            / (abs(linearslope(x, y, idx+1) - linearslope(x, y, idx)) + abs(linearslope(x, y, idx) - linearslope(x, y, idx-1)))

        elif (idx == stop - 2):
            #? mi_1 = linearslope(x, y, idx-1)
            #? mi_2 = linearslope(x, y, idx-2) # i-2
            return (abs(linearslope(x, y, idx-2) - linearslope(x, y, idx-1)) * linearslope(x, y, idx)  \
            + abs(linearslope(x, y, idx-1) - linearslope(x, y, idx)) * linearslope(x, y, idx-1)) \
            / (abs(linearslope(x, y, idx-2) - linearslope(x, y, idx-1)) + abs(linearslope(x, y, idx-1) - linearslope(x, y, idx)))

        elif idx == stop - 1:
            #? mi_1 = linearslope(x, y, idx-1)
            #? mi_2 = linearslope(x, y, idx-2) # i-2
            return (3*linearslope(x, y, idx-1) - linearslope(x, y, idx-2)) / 2
        
        mi_2 = linearslope(x, y, idx-2) # i-2
        mi_1 = linearslope(x, y, idx-1) # i-1
        mi = linearslope(x, y, idx) #i
        mi1 = linearslope(x, y, idx+1) #i+1 
                
        thr = max(abs(mi1 - mi), abs(mi - mi_1), abs(mi_1 - mi_2), -1e99) * 1e-9

        if (abs(mi - mi1) + abs(mi_2 - mi_1)) < thr:
            return (mi_1 + mi) / 2
        else:
            return (abs(mi1 - mi) * mi_1 + abs(mi_1 - mi_2) * mi) / ( abs(mi1 - mi) + abs(mi_1 - mi_2) )


    @numba.jit(nopython=True)
    def akima_spline_kernel(x_new, x, y, n_in, ngroups, nnans, result):
        for j in range(n_in):
            for i in range(x_new.shape[0]):
                special_index = j * x_new.shape[0] + i
                idx = -1
                stop = (j+1)*ngroups - nnans[j]
                if x_new[i] == x[stop - 1]: #* deal with extrema
                    result[special_index] = y[stop - 1] 
                
                else:
                    for k in range(j*ngroups, (j+1)*(ngroups) - 1):
                        
                        if (x_new[i] >= x[k] and x_new[i] < x[k + 1]):
                            idx = k
                            break
                    
                    #* Akima spline interpolation 
                    mi = linearslope(x, y, idx)
                    si = splineslope(x, y, idx, j*ngroups, stop)
                    si1 = splineslope(x, y, idx+1, j*ngroups, stop)
                
                    p0 = y[idx]

                    p1 = si

                    p2 = (3*mi - 2*si - si1) / (x[idx+1] - x[idx])

                    p3 = (si + si1 - 2*mi) / (x[idx+1] - x[idx])**2

                    result[special_index] = p0 + p1 * (x_new[i] - x[idx]) + p2 * (x_new[i] - x[idx])**2 + p3 * (x_new[i] - x[idx])**3

class AkimaInterpolant():
    def __init__(self, use_gpu=True, threadsperblock = 64):

        # TODO: add flexibility in output shape

        self.threadsperblock = threadsperblock
        self.xp = xp  
        if use_gpu and cuda_available:
            self.evaluate = self.evaluate_gpu
        else:
            self.evaluate = self.evaluate_cpu

    @property
    def use_numba(self):
        return True

    def __call__(self, x_new, x, y, **kwargs):

        ngroups = x.shape[-1]
        shape_out = x.shape[:-1] + x_new.shape

        x = x.reshape(-1, ngroups, order='C')
        y = y.reshape(-1, ngroups, order='C')

        nf = x_new.shape[0]
        nin = x.shape[0]
        result = self.xp.zeros(nin * nf)

        x_flat = x.flatten()
        y_flat = y.flatten()
        nnans = self.xp.count_nonzero(self.xp.isnan(x), axis=1)

        # blockspergrid = (nf + (self.threadsperblock - 1)) // self.threadsperblock
        # num_sets = nin
        # grid = (blockspergrid, num_sets, 1)

        self.evaluate(x_new, x_flat, y_flat, nin, ngroups, nnans, result)
        

        # mempool = xp.get_default_memory_pool()
        # mempool.free_all_blocks()


        return result.reshape(shape_out, order='C')
    
    def evaluate_gpu(self, x_new, x, y, nin, ngroups, nnans, result):

        blockspergrid = (x_new.shape[0] + (self.threadsperblock - 1)) // self.threadsperblock
        num_sets = nin
        grid = (blockspergrid, num_sets, 1)

        akima_spline_kernel[grid, self.threadsperblock](x_new, x, y, nin, ngroups, nnans, result)

        return result
    
    def evaluate_cpu(self, x_new, x, y, nin, ngroups, nnans, result):
            
        akima_spline_kernel(x_new, x, y, nin, ngroups, nnans, result)

        return result

    



'''
To generalize this code to also run on CPUs, you could use a combination of Numba's JIT compilation for CPUs and conditional logic to choose between the GPU and CPU code paths based on the availability of a CUDA-capable GPU. Here's a high-level plan:

Create a new version of each @cuda.jit decorated function that is decorated with @numba.jit instead. These will be the CPU versions of the functions.

In each of these new functions, replace CUDA-specific code with equivalent CPU code. For example, replace cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x with a simple loop over the range of data.

Create a function or method that checks if a CUDA-capable GPU is available. This could be as simple as trying to import numba.cuda and catching the ImportError if it's not available.

In the AkimaInterpolant class, use the result of the GPU availability check to decide whether to use the GPU or CPU versions of the functions. You could store the appropriate functions as attributes of the class instance, for example.

Also in the AkimaInterpolant class, replace the cuda.as_cuda_array call with a conditional that uses numpy.asarray if running on a CPU.

Finally, replace the xp module (which could be either cupy or numpy) with numpy if running on a CPU.

This approach allows you to maintain a single codebase that can run on both GPUs and CPUs, with the trade-off of having some duplication in the code. It also assumes that the CPU and GPU code are similar enough that they can be maintained in parallel. If the CPU and GPU code diverge significantly, it might be better to separate them into different modules or classes.

'''