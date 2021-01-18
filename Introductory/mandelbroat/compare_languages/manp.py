#!/usr/bin/env python

from scipy import *
from pylab import *
import time

def Mand(z0, max_steps):
    z = 0j
    for itr in range(max_steps):
        if abs(z)>2.:
            return itr
        z = z*z + z0
    return max_steps

if __name__ == '__main__':

    Nx = 1000
    Ny = 1000
    max_steps = 1000 #50

    ext = [-2,1,-1,1]
    t0 = time.time()
    
    data = zeros( (Nx,Ny) )

    for i in range(Nx):
        for j in range(Ny):
            x = ext[0] + (ext[1]-ext[0])*i/(Nx-1.)
            y = ext[2] + (ext[3]-ext[2])*j/(Ny-1.)
            data[i,j] = Mand(x + y*1j, max_steps)
    print ('clock time: '+str( time.time()-t0) )
    imshow(transpose(1./data), extent=ext)
    show()
    
