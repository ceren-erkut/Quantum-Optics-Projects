"""
Project-5: Coherent Population Trapping
I, Ceren Erkut, declare that this code below solely belongs to me.
Email: ceren.erkut@ug.bilkent.edu.tr
The following references have been used in preparing this project:
https://github.com/WarrenWeckesser/odeintw (In order to solve ordinary differential equations with complex initial conditions)

Submitted as part of Phys-442/612: Quantum Optics course
Date: 20/04/20

Versions used:
Numpy__version__ : 1.17.2
Scipy__version__ : 1.3.1
Qutip__version__ : 4.4.1 
Matplotlib__version__ : 3.1.1
odeintw__version__ : 0.1.2.dev2

"""

import numpy as np
import matplotlib.pyplot as plt
from odeintw import odeintw

# defines the ODEs
# r1 for C_a | r2 for C_b | r3 for C_c
def ode_solv (r, t, omega):
    dr1dt = (omega[0] * r[1] + omega[1]*r[2]) * 1j / 2
    dr2dt = (omega[0]*r[0]) * 1j / 2
    dr3dt = (omega[1]*r[0]) * 1j / 2
    drdt = [dr1dt, dr2dt, dr3dt]
    return drdt


############################# QUESTION 1 #############################
omega = [1,1] # for R1 and R2
theta = np.pi/2
beta = np.pi
c0 = [0+0j, np.cos(theta/2)+0j, np.sin(theta/2)*np.exp(-1j*beta)] # initial conditions for C_a, C_b, C_c 
t = np.linspace(0,5,100) # x-axis of the plot
c = odeintw(ode_solv,c0,t,args=(omega,)) # the solution matrix containing C_a, C_b, C_c as column vectors

fig = plt.figure(figsize=(15, 10))
plt.plot(t, np.abs(c[:,0])**2, 'm', linestyle = 'dashed', label = '$|C_a(t)|^2$') # plots the amplitude squared of C_a
plt.plot(t, np.abs(c[:,1])**2,'g-',linewidth=2, linestyle = 'dashed' , label = '$|C_b(t)|^2$') # plots the amplitude squared of C_b
plt.plot(t, np.abs(c[:,2])**2,'r-', linestyle = 'dotted', label = '$|C_c(t)|^2$') # plots the amplitude squared of C_c
plt.plot(t, np.abs(c[:,0])**2 + np.abs(c[:,1])**2 + np.abs(c[:,2])**2,'k',linewidth=2, label = '$|C_a(t)|^2$ + $|C_b(t)|^2$ + $|C_c(t)|^2$') # plots the total amplitude squared
plt.xlabel('Time')
plt.ylabel('Population Magnitudes')
plt.title('Amplitude Squared Solutions ($\\Omega_{R1}=\\Omega_{R2}=1$, $\\theta = \\frac{\pi}{2}$, $\\beta = \\pi$)')
plt.yticks(np.arange(0, 1.2, step=0.1))  # Set label locations.
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.legend()


############################# QUESTION 2 #############################
c0 = [1+0j, 0+0j, 0+0j] # initial conditions for C_a, C_b, C_c 
c = odeintw(ode_solv,c0,t,args=(omega,)) # the solution matrix containing C_a, C_b, C_c as column vectors

fig2 = plt.figure(figsize=(15, 10))
plt.plot(t, np.abs(c[:,0])**2 + np.abs(c[:,1])**2 + np.abs(c[:,2])**2,'k',linewidth=2, label = '$|C_a(t)|^2$ + $|C_b(t)|^2$ + $|C_c(t)|^2$') # plots the total amplitude squared
plt.plot(t, np.abs(c[:,0])**2, 'y', linestyle = 'dashed', label = '$|C_a(t)|^2$') # plots the amplitude squared of C_a
plt.plot(t, np.abs(c[:,1])**2,'g-',linewidth=2, linestyle = 'dashed' , label = '$|C_b(t)|^2$') # plots the amplitude squared of C_b
plt.plot(t, np.abs(c[:,2])**2,'r-', linestyle = 'dotted', label = '$|C_c(t)|^2$') # plots the amplitude squared of C_c
plt.xlabel('Time')
plt.ylabel('Population Magnitudes')
plt.title('Amplitude Squared Solutions ($\\Omega_{R1}=\\Omega_{R2}=1$, $C_a(0) = 1$, $C_b(0) = 0$, $C_c(0) = 0$)')
plt.yticks(np.arange(0, 1.2, step=0.1))  # Set label locations.
plt.xticks(np.arange(min(t), max(t)+0.2, step=0.2))  # Set label locations.
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.legend()


############################# QUESTION 3 #############################
beta = np.pi/2
c0 = [0+0j, np.cos(theta/2)+0j, np.sin(theta/2)*np.exp(-1j*beta)] # initial conditions for C_a, C_b, C_c
c = odeintw(ode_solv,c0,t,args=(omega,)) # the solution matrix containing C_a, C_b, C_c as column vectors

fig3 = plt.figure(figsize=(15, 10))
plt.plot(t, np.abs(c[:,0])**2 + np.abs(c[:,1])**2 + np.abs(c[:,2])**2,'k',linewidth=2, label = '$|C_a(t)|^2$ + $|C_b(t)|^2$ + $|C_c(t)|^2$') # plots the total amplitude squared
plt.plot(t, np.abs(c[:,0])**2, 'b', label = '$|C_a(t)|^2$') # plots the amplitude squared of C_a
plt.plot(t, np.abs(c[:,1])**2,'g-',linewidth=2, linestyle = 'dashed' , label = '$|C_b(t)|^2$') # plots the amplitude squared of C_b
plt.plot(t, np.abs(c[:,2])**2,'y-', linestyle = 'dashed', label = '$|C_c(t)|^2$') # plots the amplitude squared of C_c
plt.title('Amplitude Squared Solutions ($\\Omega_{R1}=\\Omega_{R2}=1$, $\\theta = \\frac{\pi}{2}$, $\\beta = \\frac{\pi}{2}$)')
plt.xlabel('Time')
plt.ylabel('Population Magnitudes')
plt.yticks(np.arange(0, 1.2, step=0.1))  # Set label locations.
plt.xticks(np.arange(min(t), max(t)+0.2, step=0.2))  # Set label locations.
plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
plt.legend()

plt.show()