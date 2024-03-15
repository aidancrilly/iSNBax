import scipy.constants as sc
import jax
import jax.numpy as jnp
import EpperleinHaines as EH
from functools import partial

class BetaIntegralInterp():

    def __init__(self,beta_grid):
        self.beta_grid = beta_grid
        self.compute_cumulative_integral()

    def integral(self,beta_lower,beta_upper,Ntrapz=100):
        beta = jnp.linspace(beta_lower,beta_upper,Ntrapz)
        I = jnp.trapz(self.integrand(beta),x=beta,axis=0)
        return I

    def integrand(self,beta):
        return beta**4 * jnp.exp(-beta) / 24.0
    
    def compute_cumulative_integral(self):
        def cumulative_integral(carry,beta):
            beta_prev,I_prev = carry
            I = self.integral(beta_prev,beta)
            return (beta,I+I_prev),I+I_prev

        carry = 0.0,0.0
        _,self.I_grid = jax.lax.scan(cumulative_integral,carry,self.beta_grid)

    def __call__(self,beta):
        return jnp.interp(beta,self.beta_grid,self.I_grid,right=1.0)

eta_interp = BetaIntegralInterp(jnp.linspace(0.0,25.0,500))

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
    kappa_e = EH.gamma0(Z)*ne*sc.e**2*Te*tau_e/(sc.m_e)
    mfp_ei = tau_e*jnp.sqrt(2*Te*sc.e/sc.m_e)
    effective_mfp_e = SNB_effective_mfp_e(Z, ne, Te, ln_lambda)
    return kappa_e,mfp_ei,effective_mfp_e

def SNB_effective_mfp_e(Z, ne, Te, coulomb_log):
    phi = (Z+4.2)/(Z+0.25)
    return 3.8378e16 * Te**2 / ne / coulomb_log / jnp.sqrt(Z * phi)

def calc_tau_e(Z, ne, Te, coulomb_log):
    return 3.44e11 * Te**1.5 / ne / Z / coulomb_log

def calc_coloumb_log():
    return 7.1

def calc_eta_grp(Eg_lower,Eg_upper,Te):
    beta_lower = Eg_lower[:,None]/Te[None,:]
    beta_upper = Eg_upper[:,None]/Te[None,:]
    I_lower = eta_interp(beta_lower)
    I_upper = eta_interp(beta_upper)
    return I_upper-I_lower

def calc_mfp_grp(Egc,Te,thermal_mfp):
    return 2*jnp.sqrt(2)*(Egc[:,None]/Te[None,:])**2*(thermal_mfp[None,:])