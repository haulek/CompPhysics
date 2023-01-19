#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import pylab as pl

filename = 'mand.dat'
if len(sys.argv)>1:
    filename = sys.argv[1]
    
X,Y,Z = np.loadtxt(filename,unpack=True)
ext=[X[0],X[-1],Y[0],Y[-1]]
N = int(np.sqrt(len(Z)))
dat = Z.reshape(N,N)
print(ext)
pl.imshow(dat.T,extent=ext,cmap=cm.hot,norm=LogNorm())   # pylab's function for displaying 2D image
pl.show()
