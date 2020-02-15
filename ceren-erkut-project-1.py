"""
This script solves a set of coupled ODEs
    using built-in Python libraries SciPy, NumPy and Matplotlib.
It produces 5 different subplots for the parameters given.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# defines the ODEs
def model (r, t, t1, t2, delta):
    dr1dt = -r[0]/t2 + delta*r[1]
    dr2dt = -r[1]/t2 - delta*r[0] - r[2]
    dr3dt = -(r[2] + 1)/t1 + r[1]
    drdt = [dr1dt, dr2dt, dr3dt]
    return drdt


# initial condition
r0 = [0,0,-1]
# time points
t = np.linspace(0, 30)
# prepares the subplots area
fig = plt.figure(figsize=(3, 2))
fig.suptitle('Solution to System of ODE with Parameters', fontsize=10, fontweight='bold')
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
plt.ylabel('Functions $R_1(t), R_2(t), R_3(t)$',labelpad=10)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 1, 3)

# first plot with parameters [100, 100, 0]
t1 = 100
t2 = 100
delta = 0
r = odeint(model,r0,t,args=(t1,t2,delta)) # solves the system specified as r
ax1.plot(t,r[:,0],'b-',linewidth=1,label=r'$\frac{dR_1(t)}{dt}=-\frac{R_1(t)}{T_1}-\Delta \; R_2(t)$')
ax1.plot(t,r[:,1],'r-',linewidth=1,label=r'$\frac{dR_2(t)}{dt}=-\frac{R_2(t)}{T_2}-\Delta \; R_1(t)-R_3(t)$')
ax1.plot(t,r[:,2],'g-',linewidth=1,label=r'$\frac{dR_3(t)}{dt}=-\frac{R_3(t)+1}{T_1}+R_2(t)$')
# adjustments for properly displaying
ax1.yaxis.set_ticks(np.arange(-1., 1.1, 0.5))
ax1.set_title('[$T_1,T_2,\Delta$] = [100, 100, 0]',fontsize='small')
ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)


# second plot with parameters [100, 10, 0]
t1 = 100
t2 = 10
delta = 0
r = odeint(model,r0,t,args=(t1,t2,delta)) # solves the system specified as r
ax2.plot(t,r[:,0],'b-',linewidth=1,label=r'$\frac{dR_1(t)}{dt}=-\frac{R_1(t)}{T_1}-\Delta \; R_2(t)$')
ax2.plot(t,r[:,1],'r-',linewidth=1,label=r'$\frac{dR_2(t)}{dt}=-\frac{R_2(t)}{T_2}-\Delta \; R_1(t)-R_3(t)$')
ax2.plot(t,r[:,2],'g-',linewidth=1,label=r'$\frac{dR_3(t)}{dt}=-\frac{R_3(t)+1}{T_1}+R_2(t)$')
# adjustments for properly displaying
ax2.yaxis.set_ticks(np.arange(-1., 1.1, 0.5))
ax2.set_title('[$T_1,T_2,\Delta$] = [100, 10, 0]',fontsize='small')
ax2.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)


# third plot with parameters [10, 10, 0]
t1 = 10
t2 = 10
delta = 0
r = odeint(model,r0,t,args=(t1,t2,delta)) # solves the system specified as r
ax3.plot(t,r[:,0],'b-',linewidth=1,label=r'$\frac{dR_1(t)}{dt}=-\frac{R_1(t)}{T_1}-\Delta \; R_2(t)$')
ax3.plot(t,r[:,1],'r-',linewidth=1,label=r'$\frac{dR_2(t)}{dt}=-\frac{R_2(t)}{T_2}-\Delta \; R_1(t)-R_3(t)$')
ax3.plot(t,r[:,2],'g-',linewidth=1,label=r'$\frac{dR_3(t)}{dt}=-\frac{R_3(t)+1}{T_1}+R_2(t)$')
# adjustments for properly displaying
ax3.yaxis.set_ticks(np.arange(-1., 1.1, 0.5))
ax3.set_title('[$T_1,T_2,\Delta$] = [10, 10, 0]',fontsize='small')
ax3.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)


# forth plot with parameters [10, 10, 0.2]
t1 = 10
t2 = 10
delta = 0.2
r = odeint(model,r0,t,args=(t1,t2,delta)) # solves the system specified as r
ax4.plot(t,r[:,0],'b-',linewidth=1,label=r'$\frac{dR_1(t)}{dt}=-\frac{R_1(t)}{T_1}-\Delta \; R_2(t)$')
ax4.plot(t,r[:,1],'r-',linewidth=1,label=r'$\frac{dR_2(t)}{dt}=-\frac{R_2(t)}{T_2}-\Delta \; R_1(t)-R_3(t)$')
ax4.plot(t,r[:,2],'g-',linewidth=1,label=r'$\frac{dR_3(t)}{dt}=-\frac{R_3(t)+1}{T_1}+R_2(t)$')
# adjustments for properly displaying
ax4.yaxis.set_ticks(np.arange(-1., 1.1, 0.5))
ax4.set_title('[$T_1,T_2,\Delta$] = [10, 10, 0.2]',fontsize='small')
ax4.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)


# fifth plot with parameters [10, 10, 0.75]
t1 = 10
t2 = 10
delta = 0.75
r = odeint(model,r0,t,args=(t1,t2,delta)) # solves the system specified as r
ax5.plot(t,r[:,0],'b-',linewidth=1,label=r'$\frac{dR_1(t)}{dt}=-\frac{R_1(t)}{T_1}-\Delta \; R_2(t)$')
ax5.plot(t,r[:,1],'r-',linewidth=1,label=r'$\frac{dR_2(t)}{dt}=-\frac{R_2(t)}{T_2}-\Delta \; R_1(t)-R_3(t)$')
ax5.plot(t,r[:,2],'g-',linewidth=1,label=r'$\frac{dR_3(t)}{dt}=-\frac{R_3(t)+1}{T_1}+R_2(t)$')
# adjustments for properly displaying
ax5.yaxis.set_ticks(np.arange(-1., 1.1, 0.5))
ax5.set_title('[$T_1,T_2,\Delta$] = [10, 10, 0.75]',fontsize='small')
ax5.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)


plt.xlabel('Time',labelpad=10)
plt.legend(loc='best')
plt.show()