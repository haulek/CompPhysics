#!/usr/bin/env python
from scipy import *    # scientific python
from pylab import *    # plotting library
import time            # timeing
from numba import jit  # This is the new line with numba
        
@jit(nopython=True)    # This is the second new line with numba
def Mand(z0, max_steps):
    z = 0j  # no need to specify type. To initialize to complex number, just assign 0j==i*0
    for itr in range(max_steps):
        if abs(z)>2.:
            return itr
        z = z*z + z0
    return max_steps

if __name__ == '__main__':

    Nx = 1000
    Ny = 1000
    max_steps = 1000 #50

    ext = [-2,1,-1,1]  # no need to specify this is a python list, just initialize it
    t0 = time.time()
    
    data = zeros( (Nx,Ny) ) # initialize a 2D dynamic array (something C++ is still lacking)

    for i in range(Nx):
        for j in range(Ny):
            x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.)
            y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.)
            data[i,j] = Mand(x + y*1j, max_steps)  # creating complex number of the fly
            
    print ('clock time: '+str( time.time()-t0) )
    imshow(transpose(1./data), extent=ext)   # pylab's function for displaying 2D image
    show()                                   # showing the image.
    
