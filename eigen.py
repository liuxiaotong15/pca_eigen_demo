from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 5  # how many seconds to simulate
history_len = 500  # how many trajectory points to display

# # create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.02


theta = np.radians(np.array(list(range(0, 360, 1))))
print(theta)

y1 = L1*sin(theta)
x1 = L1*cos(theta)

# A = np.array([[0, 2],
#      [2, 0]])

A = np.array([[1, 1/3],
     [4/3, 1]])



ev = np.array([x1,
     y1])

print(np.dot(A, ev))
print(np.dot(A, ev).shape)

x2 = np.dot(A, ev)[0]
y2 = np.dot(A, ev)[1]

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
ax.set_aspect('equal')
ax.grid()

line1, = ax.plot([], [], 'o-', lw=2)
trace1, = ax.plot([], [], ',-', lw=1)
line2, = ax.plot([], [], 'o-', lw=2)
trace2, = ax.plot([], [], ',-', lw=1)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x1, history_y1 = deque(maxlen=history_len), deque(maxlen=history_len)
history_x2, history_y2 = deque(maxlen=history_len), deque(maxlen=history_len)


def animate1(i):
    thisx = [0, x1[i]]
    thisy = [0, y1[i]]

    if i == 0:
        history_x1.clear()
        history_y1.clear()

    history_x1.appendleft(thisx[1])
    history_y1.appendleft(thisy[1])

    line1.set_data(thisx, thisy)
    trace1.set_data(history_x1, history_y1)
    # time_text.set_text(time_template % (i*dt))

    thisx2 = [0, x2[i]]
    thisy2 = [0, y2[i]]

    if i == 0:
        history_x2.clear()
        history_y2.clear()

    history_x2.appendleft(thisx2[1])
    history_y2.appendleft(thisy2[1])

    line2.set_data(thisx2, thisy2)
    trace2.set_data(history_x2, history_y2)
    time_text.set_text(time_template % (i*dt))

    return line1, line2, trace1, trace2, time_text


ani = animation.FuncAnimation(
    fig, animate1, len(theta), interval=dt*1000, blit=True)

plt.show()
