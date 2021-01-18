#!/usr/bin/env python
from scipy import *  # for arrays
from pylab import *  # for plotting
import mandel  # importing module created by f2py
import time

# The range of the mandelbrot plot [x0,x1,y0,y1]
ext=[-2,1,-1,1]
#ext=[-1.8,-1.72,-0.05,0.05]

tc = time.process_time() # cpu time
tw = time.time()  # wall time
data = mandel.mandelb(ext,1000,1000).transpose()

print('# wall time : ', time.time()-tw, 's  clock time : ', time.process_time() - tc, 's')

# Using python's pylab, we display pixels to the screen!
imshow(data, interpolation='bilinear', origin='lower', extent=ext, aspect=1.)
#imshow(data, interpolation='bilinear', cmap=cm.hot, origin='lower', extent=ext, aspect=1.)
colorbar()
show()
