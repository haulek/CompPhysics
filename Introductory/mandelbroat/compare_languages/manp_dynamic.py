#!/usr/bin/env python
from scipy import *
from numpy import *
from pylab import *
import time
from numba import jit  # This is the new line with numba
        
@jit(nopython=True)    # This is the second new line with numba
def MandNumba(ext, max_steps, Nx, Ny):
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

def ax_update(ax):  # actual plotting routine
    ax.set_autoscale_on(False) # Otherwise, infinite loop
    # Get the range for the new area
    xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
    xend = xstart + xdelta
    yend = ystart + ydelta
    ext=array([xstart,xend,ystart,yend])
    data = MandNumba(ext, max_steps, Nx, Ny) # actually producing new fractal
    
    # Update the image object with our new data and extent
    im = ax.images[-1]  # take the latest object
    im.set_data(data)   # update it with new data
    im.set_extent(ext)           # change the extent
    ax.figure.canvas.draw_idle() # finally redraw

if __name__ == '__main__':
    Nx = 1000
    Ny = 1000
    max_steps = 1000 # 50

    ext = [-2,1,-1,1]
    
    t0 = time.time()    
    data = MandNumba(array(ext), max_steps, Nx, Ny)
    t1 = time.time()
    print('Python: ', t1-t0)

    #imshow(1./data, extent=ext)
    fig,ax=subplots(1,1)
    ax.imshow(data, extent=ext,aspect='equal',origin='lower')
    
    ax.callbacks.connect('xlim_changed', ax_update)
    ax.callbacks.connect('ylim_changed', ax_update)
    
    show()
