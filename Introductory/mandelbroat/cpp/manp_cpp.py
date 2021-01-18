#!/usr/bin/env python
from scipy import *
from pylab import *
import time
import imanc
from timeit import default_timer as timer
from numba import jit
#%matplotlib qt

def MandPyth(ext, max_steps, Nx, Ny):
    data = ones( (Nx,Ny) )*max_steps
    for i in range(Nx):
        for j in range(Ny):
            x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.)
            y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.)
            z0 = x+y*1j
            z = 0j
            for itr in range(max_steps):
                if abs(z)>2.:
                    data[j,i]=itr
                    break
                z = z*z + z0
    return data

@jit(nopython=True)
def MandNumba(ext, max_steps, Nx, Ny):
    data = ones( (Nx,Ny) )*max_steps
    for i in range(Nx):
        for j in range(Ny):
            x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.)
            y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.)
            z0 = complex(x,y) #z0 = x+y*1j
            z = 0j
            for itr in range(max_steps):
                if z.real*z.real + z.imag*z.imag > 4.:
                    data[j,i]=itr
                    break
                z = z*z + z0
    return data

def MandPybind11(ext, max_steps, Nx, Ny):
    data = ones((Ny,Nx));
    imanc.mand(data, Nx, Ny, max_steps, ext)
    return data

    
if __name__ == '__main__':

    Nx = 1000
    Ny = 1000
    max_steps = 1000 # 50

    ext = [-2,1,-1,1]


    t0 = time.time()
    t0_ = time.process_time() # cpu time
    data = MandPybind11(ext, max_steps, Nx, Ny)
    t1 = time.time()
    t1_ = time.process_time() # cpu time
    print('pybind11: walltime: ', t1-t0, 'cputime: ', t1_-t0_)
    imshow(data, extent=ext)
    show()

    t0 = time.time()
    t0_ = time.process_time() # cpu time
    data = MandNumba(array(ext), max_steps, Nx, Ny)
    t1 = time.time()
    t1_ = time.process_time() # cpu time
    print('numba: walltime: ', t1-t0, 'cputime: ', t1_-t0_)
    imshow(data, extent=ext)
    show()
    
    #t0 = time.time()
    #data = MandPyth(array(ext), max_steps, Nx, Ny)
    #t1 = time.time()
    #print('Python: ', t1-t0)
    #imshow(data, extent=ext)
    #show()
