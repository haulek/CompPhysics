from numpy import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

g =  9.8 # acceleration due to gravity, in m/s^2
L = 1.0 # length of pendulums
m = 1.0 # mass of pendulums

def derivs(x,t):
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
dt = 0.03
t = arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
th2 = -10.0
w1 = 0.0
w2 = 0.0

# initial state
x0 = array([th1, th2, w1, w2])*pi/180.

# integrate your ODE using scipy.integrate.
x = integrate.odeint(derivs, x0, t)

x1 =  L*sin(x[:,0])
y1 = -L*cos(x[:,0])
x2 =  L*sin(x[:,1]) + x1
y2 = -L*cos(x[:,1]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, arange(1, len(t)),
    interval=25, blit=False, init_func=init)

#ani.save('double_pendulum.mp4', fps=15, clear_temp=True)
plt.show()
