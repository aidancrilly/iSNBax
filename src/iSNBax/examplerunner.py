from solver import *
import matplotlib.pyplot as plt

def tanh_profile(x,width,T0,Tmax):
    return T0 + (Tmax-T0)*0.5*(1.0+jnp.tanh((x-jnp.mean(x))/width))

Nx = 100
xs = jnp.linspace(0.0,100e-6,Nx+1)
Z  = 1.0*jnp.ones(Nx)
source_term = jnp.zeros(Nx)

Ee_max = 1e4 
Ngrp = 10

fig = plt.figure(dpi=100)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ni = 1e30*jnp.ones(Nx)
args = assemble_args('Cartesian',xs,ni,Z,source_term,Ee_max,Ngrp)
Te0 = tanh_profile(args['xc'],10.0e-6,1e2,5e3)
Nt = 5
ts = jnp.linspace(0.0,1.0e-10,Nt)
dt0 = 0.1*(ts[1]-ts[0])

Te_SH = solve_SpitzerHarm(Te0, dt0, ts, args)
Te_iSNB = solve_iSNB(Te0, dt0, ts, args)
ax1.plot(args['xc']*1e6,Te_iSNB.T)
ax1.plot(args['xc']*1e6,Te_SH.T,c='k',ls='--')
ax1.plot(args['xc']*1e6,jnp.zeros_like(args['xc']),c='b',label='SNB')
ax1.plot(args['xc']*1e6,jnp.zeros_like(args['xc']),c='k',ls='--',label='SNB')
ax1.legend(frameon=False)
ax1.set_title(r'Z = 1, $n_e$ = 10$^{30}$ 1 / m$^3$')
ax1.set_xlabel('x (um)')
ax1.set_ylabel('Te (eV)')
ax1.set_ylim(9e1,5.2e3)

ni = 1e28*jnp.ones(Nx)
args = assemble_args('Cartesian',xs,ni,Z,source_term,Ee_max,Ngrp)
Nt = 5
ts = jnp.linspace(0.0,1.0e-12,Nt)
dt0 = 0.1*(ts[1]-ts[0])

Te_SH = solve_SpitzerHarm(Te0, dt0, ts, args)
Te_iSNB = solve_iSNB(Te0, dt0, ts, args)
ax2.plot(args['xc']*1e6,Te_iSNB.T)
ax2.plot(args['xc']*1e6,Te_SH.T,c='k',ls='--')
ax2.plot(args['xc']*1e6,jnp.zeros_like(args['xc']),c='b',label='SNB')
ax2.plot(args['xc']*1e6,jnp.zeros_like(args['xc']),c='k',ls='--',label='SNB')
ax2.legend(frameon=False)
ax2.set_title(r'Z = 1, $n_e$ = 10$^{28}$ 1 / m$^3$')
ax2.set_xlabel('x (um)')
ax2.set_ylabel('Te (eV)')
ax2.set_ylim(9e1,5.2e3)

fig.tight_layout()

plt.show()