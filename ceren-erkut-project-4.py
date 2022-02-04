"""
Project-4: Semiclassical Hahn Echo
I, Ceren Erkut, declare that this code below solely belongs to me.
Email: ceren.erkut@ug.bilkent.edu.tr
Submitted as part of Phys-442/612: Quantum Optics course
Date: 13/04/20 | 5:10 a.m.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# defines the ODEs
# r1 for rx | r2 for ry
def ode_solv (r, t, t2, delta):
    dr1dt = -r[0]/t2 + delta*r[1]
    dr2dt = -r[1]/t2 - delta*r[0] 
    drdt = [dr1dt, dr2dt]
    return drdt

# 2D rotation matrix with respect to x-axis
def rot_matrix (angle, x, y):
    xx = x*np.cos(angle) - y*np.sin(angle)
    yy = x*np.sin(angle) + y*np.cos(angle)
    return [xx, yy]

# calculates the mean after solving the Bloch equations for each spin vector
def average (r0, t, t2, delta, angle):
    rx_mean = np.zeros(t.shape) # mean array of x component
    ry_mean = np.zeros(t.shape) # mean array of y component
    N = np.shape(r0)[0]
    num_spins = np.linspace(1, N, N)
    values_at_pulse = np.zeros((N,2)) # 2D array of spin values at the pulse instant | 1st col = x | 2nd col = y
    initial = np.empty(2)
    for d, i in zip(delta, num_spins):
        initial[0] = r0[int(i)-1, 0]
        initial[1] = r0[int(i)-1, 1]
        r = odeint(ode_solv,initial,t,args=(t2,d)) # solves the system specified as r for 1 spin at a time
        rx_mean = rx_mean + r[:,0]
        ry_mean = ry_mean + r[:,1]
        values_at_pulse[int(i)-1,:] = rot_matrix(angle[int(i)-1], r[len(t)-1,0], r[len(t)-1,1]) # rotation matrix applied at the pulse instant
    rx_mean = rx_mean / N
    ry_mean = ry_mean / N
    r_mean = [rx_mean, ry_mean, values_at_pulse]
    return r_mean


N = 500 # number of spins
t2 = 60 # observation duration
tR1 = 30 # pulse instant
tR2 = 50 # second pulse instant
angle_zero = np.repeat(0, repeats=N, axis=0)
# initial xy-component values for each spin vector
r0 = np.repeat([[0, -1]], repeats=N, axis=0)
# time points
t = np.linspace(0, t2, 300)

############################# QUESTION 1 #############################
# prepares the subplots area
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False, sharey=True)
fig.suptitle('No-Rotation/Pulse', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel('Spin Vector Components $R_x(t), R_y(t)$, Averaged',fontsize='large', labelpad=15)
# detuning parameters for spin vectors
delta = np.repeat(1, repeats=N, axis=0)
r = odeint(ode_solv,r0[0],t,args=(t2,1)) # solves the system specified as r | no need to calculate mean
axes[0].plot(t,r[:,0],'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
axes[0].plot(t,r[:,1],'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
axes[0].set(title='$\Delta_i$ = 1', xlabel='Time')
axes[0].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
# detuning parameters for spin vectors
delta = np.linspace(1, 1.5, N)
r_mean = average(r0, t, t2, delta, angle_zero) # calculates the mean considering varying detuning parameters
axes[1].plot(t,r_mean[0],'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
axes[1].plot(t,r_mean[1],'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
axes[1].set(title='$\Delta_i$ = [1, 1.5]', xlabel='Time')
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
fig.legend(loc='upper right')


############################# QUESTION 2 #############################
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Rotation at $t_R$ = 30 with Pulse = $\pi$ and $\Delta_i$ = [1, 1.5]', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t,r_mean[0],'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax2.plot(t,r_mean[1],'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax2.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax2.set(title='No-Rotation Until the End', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')
# second plot
tpulse_len = int(300*(t2-tR1)/t2)
tpulse =  np.linspace(0, int(t2-tR1), tpulse_len)
angle = np.repeat(np.pi, repeats=N, axis=0)
r_mean_pulse = average(r_mean[2], tpulse, t2, delta, angle) # new solution for r, upon the pulse instant
rx_mean_pulse = np.concatenate((r_mean[0][0:tpulse_len], r_mean_pulse[0])) # concatenates x values, from the beginning to the pulse instant
ry_mean_pulse = np.concatenate((r_mean[1][0:tpulse_len], r_mean_pulse[1])) # concatenates y values, from the beginning to the pulse instant
ax1.plot(t,rx_mean_pulse,'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax1.plot(t,ry_mean_pulse,'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax1.annotate(s='', xytext=(33,rx_mean_pulse[30]-0.25), xy=(30,0), arrowprops=dict(arrowstyle='->'))
ax1.set(title='Rotation at $t_R=30$ with Pulse = $\pi$', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')


############################# QUESTION 3 #############################
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Rotation at $t_R$ = 30 with Pulse = $\pi$ and $\Delta_i$ = [1, 5]', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
delta = np.linspace(1, 5, N)
r_mean2 = average(r0, t, t2, delta, angle_zero)
ax2.plot(t,r_mean2[0],'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax2.plot(t,r_mean2[1],'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax2.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax2.set(title='No-Rotation Until the End', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')
# second plot
r_mean_pulse2 = average(r_mean2[2], tpulse, t2, delta, angle) # r upon the pulse instant
rx_mean_pulse2 = np.concatenate((r_mean2[0][0:150], r_mean_pulse2[0])) # concatenates x values, from the beginning to the pulse instant
ry_mean_pulse2 = np.concatenate((r_mean2[1][0:150], r_mean_pulse2[1])) # concatenates y values, from the beginning to the pulse instant
ax1.plot(t,rx_mean_pulse2,'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax1.plot(t,ry_mean_pulse2,'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax1.annotate(s='', xytext=(33,rx_mean_pulse[30]-0.25), xy=(30,-0.01), arrowprops=dict(arrowstyle='->'))
ax1.set(title='Rotation at $t_R=30$ with Pulse = $\pi$', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')


############################# QUESTION 4 #############################
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Rotation at $t_R$ = 30 with varying pulse and $\Delta_i$ = [1, 1.5]', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
delta = np.linspace(1, 1.5, N)
angle_error = np.linspace(0.9*np.pi, 1.1*np.pi, N)
r_mean_pulse3 = average(r_mean[2], tpulse, t2, delta, angle_error) # new solution for r, upon the pulse instant
rx_mean_pulse3 = np.concatenate((r_mean[0][0:tpulse_len], r_mean_pulse3[0])) # concatenates x values, from the beginning to the pulse instant
ry_mean_pulse3 = np.concatenate((r_mean[1][0:tpulse_len], r_mean_pulse3[1])) # concatenates y values, from the beginning to the pulse instant
ax2.plot(t,rx_mean_pulse3,'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax2.plot(t,ry_mean_pulse3,'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax2.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax2.annotate(s='', xytext=(33,rx_mean_pulse[30]-0.25), xy=(30,0), arrowprops=dict(arrowstyle='->'))
ax2.set(title='Rotation at $t_R=30$ with Pulse = $[0.9\pi, 1.1\pi]$', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')
# second plot
ax1.plot(t,rx_mean_pulse,'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax1.plot(t,ry_mean_pulse,'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax1.annotate(s='', xytext=(33,rx_mean_pulse[30]-0.25), xy=(30,0), arrowprops=dict(arrowstyle='->'))
ax1.set(title='Rotation at $t_R=30$ with Pulse = $\pi$', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')


############################# QUESTION 5 #############################
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Different Pulses with $\Delta_i$ = [1, 1.5]', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
# defines the time vectors
tpulse_len2 = int(300*(tR2-tR1)/t2)
tpulse2 =  np.linspace(0, int(tR2-tR1), tpulse_len2)
tpulse_len3 = int(300*(t2-tR2)/t2)
tpulse3 =  np.linspace(0, int(t2-tR2), tpulse_len3)
angle_half = np.repeat(np.pi/2, repeats=N, axis=0)

r_mean_pulse4 = average(r_mean[2], tpulse2, t2, delta, angle_half) # new solution for r, from the first pulse to the second pulse
r_mean_pulse5 = average(r_mean_pulse4[2], tpulse3, t2, delta, angle_half) # new solution for r, from the second pulse to the end

rx_mean_pulse4 = np.concatenate((r_mean[0][0:tpulse_len], r_mean_pulse4[0])) # concatenate second part to first part for x values
ry_mean_pulse4 = np.concatenate((r_mean[1][0:tpulse_len], r_mean_pulse4[1])) # concatenate second part to first part for y values
rx_mean_pulse5 = np.concatenate((rx_mean_pulse4, r_mean_pulse5[0])) # concatenate third part to second part for x values
ry_mean_pulse5 = np.concatenate((ry_mean_pulse4, r_mean_pulse5[1])) # concatenate third part to second part for y values

ax2.plot(t,rx_mean_pulse5,'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax2.plot(t,ry_mean_pulse5 ,'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax2.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax2.annotate(s='', xytext=(tR1+3,rx_mean_pulse[tR1]-0.25), xy=(tR1,0), arrowprops=dict(arrowstyle='->'))
ax2.annotate(s='', xytext=(tR2+3,rx_mean_pulse[tR2]-0.25), xy=(tR2,0), arrowprops=dict(arrowstyle='->'))
ax2.set(title='Rotation at $t_R=30$ and $t_R=50$ with Pulse = $\\frac{\pi}{2}$', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')
# second plot
ax1.plot(t,rx_mean_pulse,'b-',linewidth=1,label=r'$\frac{dR_x(t)}{dt}=-\frac{R_x(t)}{T_1}-\Delta \; R_y(t)$')
ax1.plot(t,ry_mean_pulse,'r-',linewidth=1,label=r'$\frac{dR_y(t)}{dt}=-\frac{R_y(t)}{T_2}-\Delta \; R_x(t)$')
ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax1.annotate(s='', xytext=(33,rx_mean_pulse[30]-0.25), xy=(30,0), arrowprops=dict(arrowstyle='->'))
ax1.set(title='Rotation at $t_R=30$ with Pulse = $\pi$', xlabel='Time', ylabel='Spin Vector Components $R_x(t), R_y(t)$, Averaged')


plt.show()