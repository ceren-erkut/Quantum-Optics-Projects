"""
Project-2: Photonic Distributions Using QuTiP
I, Ceren Erkut declare that this code below solely belongs to me.
Email: ceren.erkut@ug.bilkent.edu.tr
The following references have been used in preparing this project:

Submitted as part of Phys-442/612: Quantum Optics course
Date: 28/02/20
"""
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from qutip import *


######## QUESTION 1 ########
# Initial adjustments with the figure
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(1, 2, 1)
plt.ylabel('P(n)')
plt.xlabel('n')
ax2 = fig.add_subplot(1, 2, 2)
plt.ylabel('P(n)')
plt.xlabel('n')
ax1.set_title('$\\bar{n}$ = 0.1',fontsize='small')
ax2.set_title('$\\bar{n}$ = 2',fontsize='small')
ax1.set_xlim([-0.4,15])
ax1.set_ylim([0,1])
ax2.set_xlim([-0.4,15])
ax2.set_ylim([0,1])
fig.suptitle('Fock Distribution for Thermal Photon Ensemble', fontsize=10, fontweight='bold')
x = np.array(range(15))
# Actual computation
rho_thermal = thermal_dm(15, 0.1)
ax1.bar(x,rho_thermal.diag())
rho_thermal2 = thermal_dm(15, 2)
ax2.bar(x,rho_thermal2.diag())



######## QUESTION 2 ########
# Initial adjustments with the figure
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(1, 2, 1)
plt.ylabel('P(n)')
plt.xlabel('n')
ax2 = fig.add_subplot(1, 2, 2)
plt.ylabel('P(n)')
plt.xlabel('n')
ax1.set_title('$\\bar{n}$ = 2',fontsize='small')
ax2.set_title('$\\bar{n}$ = 10',fontsize='small')
ax1.set_xlim([0,24])
ax1.set_ylim([0,0.275])
ax2.set_xlim([0,24])
ax2.set_ylim([0,0.275])
fig.suptitle('Fock Distribution for Coherent State', fontsize=10, fontweight='bold')
x = np.array(range(24))
# Actual computation
rho_coherent = coherent_dm(24, np.sqrt(2))
ax1.bar(x,rho_coherent.diag())
rho_coherent2 = coherent_dm(24, np.sqrt(10))
ax2.bar(x,rho_coherent2.diag())



######## QUESTION 3 ########
""" 
No explicit equation has been found to solve this problem. 
The phase distribution of a coherent state is given as an approximation by the equation (3.29) in Gerry & Knight.
This is, in essence, a Gaussian distribution centered at a constant.
A similar Gaussin expression is given by the equation (3.144) for the Q-function of a coherent state.
Therefore, the phase distribution formula has been derived using Q-function, which is available with QuTiP.
Analytical derivation has been done for this purpose.
"""
# Initial adjustments with the figure
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(top=0.85,hspace=0.8,wspace=0.4)
ax1 = fig.add_subplot(1, 2, 1)
plt.ylabel('P($\phi$)')
plt.xlabel('$\phi$')
ax2 = fig.add_subplot(1, 2, 2)
plt.ylabel('P($\phi$)')
plt.xlabel('$\phi$')
ax1.set_title('$\\bar{n}$ = 2',fontsize='small')
ax2.set_title('$\\bar{n}$ = 10',fontsize='small')
fig.suptitle('Phase Distribution for Coherent States', fontsize=10, fontweight='bold')
phivec = np.linspace(-3.5,3.5,100)
yvec = np.linspace(-3.5,3.5,100)
# Actual computation
n = 2
phase_coherent = coherent_dm(20, np.sqrt(n))
Q_coherent = qfunc(phase_coherent, phivec, yvec)
Q_coherent = Q_coherent[:, [99]]
Q_coherent = np.sqrt(2*n)*((np.pi)**(2*n-0.3))*(Q_coherent**(n))
ax1.plot(phivec, Q_coherent)
ax1.set_xlim(-4,4)
ax1.set_ylim(0, 1.5)
n = 10
phase_coherent = coherent_dm(20, np.sqrt(n))
Q_coherent = qfunc(phase_coherent, phivec, yvec)
Q_coherent = Q_coherent[:, [99]]
Q_coherent = np.sqrt(2*n)*((np.pi)**(2*n-0.3))*(Q_coherent**(n))
ax2.plot(phivec, Q_coherent)
ax2.set_xlim(-4,4)
ax2.set_ylim(0, 3)



######## QUESTION 4 ########
# Initial adjustments with the figure
fig = plt.figure(figsize=(24, 6))
fig.subplots_adjust(top=0.85,hspace=0.4,wspace=0.05)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title('Q-Function for Coherent State, $\\bar{n}$ = 10', fontsize=10, fontweight='bold')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-4, 4)
plt.ylabel('Im($\\alpha$)',labelpad=10)
plt.xlabel('Re($\\alpha$)',labelpad=10)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title('Q-Function for Number State, n = 3', fontsize=10, fontweight='bold')
fig.suptitle('Q-Functions', fontsize=15, fontweight='bold')
plt.ylabel('Im($\\alpha$)',labelpad=10)
plt.xlabel('Re($\\alpha$)',labelpad=10)
### PART A ###
xvec = np.linspace(-5,5,200)
yvec = np.linspace(-4,4,200)
# Actual computation
rho_coherent = coherent_dm(20, np.sqrt(10))
Q_coherent = qfunc(rho_coherent, xvec, yvec)
xvec = np.outer(xvec, np.ones(200)) # for a proper display, vectors are reformulated
yvec = yvec.T
surf = ax1.plot_surface(xvec, yvec, Q_coherent, cmap='viridis', edgecolor='none')
### PART B ###
xvec = np.linspace(-5,5,200)
# Actual computation
rho_fock = fock_dm(20, 3)
Q_number = qfunc(rho_fock, xvec, xvec)
xvec = np.outer(xvec, np.ones(200)) # for a proper display, vectors are reformulated
yvec = xvec.copy().T
surf = ax2.plot_surface(xvec, yvec, Q_number, cmap='viridis', edgecolor='none')
fig.colorbar(surf, shrink=0.7, aspect=5)



######## QUESTION 5 ########
# Initial adjustments with the figure
fig = plt.figure(figsize=(24, 6))
fig.subplots_adjust(top=0.85,hspace=0.4,wspace=0.05)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlim(2, 6)
ax1.set_ylim(-6, 6)
ax1.set_title('Wigner Function for Coherent State, $\\bar{n}$ = 10', fontsize=10, fontweight='bold')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title('Wigner Function for Number State, n = 3', fontsize=10, fontweight='bold')
fig.suptitle('Wigner Functions', fontsize=15, fontweight='bold')
### PART A ###
# Actual computation
rho_coherent = coherent_dm(20, np.sqrt(10))
plot_wigner(rho_coherent, fig=fig, ax=ax1, cmap=None, alpha_max=6, colorbar=True, method='iterative', projection='3d')
### PART B ###
# Actual computation
rho_fock = fock_dm(20, 3)
plot_wigner(rho_fock, fig=fig, ax=ax2, cmap=None, alpha_max=2, colorbar=True, method='iterative', projection='3d')
plt.show()