import scipy.constants as sc
import jax
import jax.numpy as jnp
from functools import partial

def assemble_args(geometry,xs,ni,Z,source_term,Ee_max,Ngrp):
    Nx = xs.shape[0]-1
    if(geometry == 'Cartesian'):
        area    = jnp.ones(Nx+1)
        vol     = xs[1:]-xs[:-1]
        xc      = 0.5*(xs[1:]+xs[:-1])
        delta_x = xc[1:]-xc[:-1]
        delta_x = jnp.concatenate([delta_x[:1],delta_x,delta_x[-1:]])

    Cv = 1.5*ni*Z*sc.e

    Egb = jnp.linspace(0.0,Ee_max,Ngrp+1)
    Egc = 0.5*(Egb[1:]+Egb[:-1])

    args = {'xs' : xs, 'xc' : xc,
            'Egb' : Egb, 'Egc' : Egc, 'Ngrp' : Ngrp,
            'area' : area, 'vol' : vol, 'delta_x' : delta_x,
            'Z' : Z, 'ne' : Z*ni, 'S' : source_term, 'Cv' : Cv}
    return args

def calc_transport_coeffs(Z, ne, Te):
    ln_lambda = calc_coloumb_log()
    tau_e = calc_tau_e(Z, ne, Te, ln_lambda)
    kappa_e = 3.2*ne*sc.e**2*Te*tau_e/(sc.m_e)
    mfp_e = tau_e*jnp.sqrt(2*Te*sc.e/sc.m_e)
    return kappa_e,mfp_e

def calc_tau_e(Z, ne, Te, coulomb_log):
    return 3.44e11 * Te**1.5 / ne / Z / coulomb_log

def calc_coloumb_log():
    return 10.0

def calc_eta_grp(Eg_lower,Eg_upper,Te,Ntrapz=100):
    beta_lower = Eg_lower[:,None]/Te[None,:]
    beta_upper = Eg_upper[:,None]/Te[None,:]
    beta = jnp.linspace(beta_lower,beta_upper,Ntrapz)
    integrand = (beta**4*jnp.exp(-beta))/24.0
    return jnp.trapz(integrand,x=beta,axis=0)

def calc_mfp_grp(Egc,Te,thermal_mfp):
    return 2*(Egc[:,None]/Te[None,:])**2*thermal_mfp[None,:]