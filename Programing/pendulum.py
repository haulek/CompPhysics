from numpy import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import matplotlib.animation as animation # 

from numba import jit  # This is the new line with numba

g =  9.8 # acceleration due to gravity, in m/s^2
L = 1.0 # length of pendulums
m = 1.0 # mass of pendulums

@jit(nopython=True)
def dx(x,t):
    """
    The right-hand side of the pendulum ODE
    x=[x1,x2,x3,x4]
    """
    x1,x2,x3,x4 = x
    c1 = 1/(m*L**2)
    dx1 = 6.*c1 * (2*x3-3*cos(x1-x2)*x4)/(16.-9.*cos(x1-x2)**2)
    dx2 = 6.*c1 * (8*x4-3*cos(x1-x2)*x3)/(16.-9.*cos(x1-x2)**2)
    dx3 = -0.5/c1 * (dx1*dx2 * sin(x1-x2) + 3*g/L * sin(x1))
    dx4 = -0.5/c1 * (-dx1*dx2 * sin(x1-x2)+ g/L * sin(x2))
    return array([dx1,dx2,dx3,dx4])

# create a time array from 0..100 sampled at 0.1 second steps

# independent variable time
t = linspace(0,20.,800)
dt = t[1]-t[0]
# initial state
x0 = array([pi/2, -pi/2, 0, 0])

# integrate your ODE using scipy.integrate.
x = integrate.odeint(dx, x0, t)

x1 =  L*sin(x[:,0])
y1 = -L*cos(x[:,0])
x2 = x1 + L*sin(x[:,1])
y2 = y1 - L*cos(x[:,1])

#################################################### copy from jupyter

fig, ax = plt.subplots(1,1)
ax.set_xlim(-2*L,2*L)
ax.set_ylim(-2*L,2*L)

#### alternative way of doing the same thing
#fig = plt.figure()
#ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2*L, 2*L), ylim=(-2*L, 2*L))

ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text
    #pass

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

#interval = delay between frames in milliseconds: default: 200
ani = animation.FuncAnimation(fig, animate, arange(1, len(t)), interval=20, init_func=init, blit=True)

#ani.save('double_pendulum.mp4', fps=15, clear_temp=True)
plt.show()
