import scipy.constants as sc
import jax.numpy as jnp

Z_eh = jnp.array([1,2,3,4,5,6,7,8,10,12,14,20,30,60])
gamma0_eh = jnp.array([3.203,4.931,6.115,6.995,7.680,8.231,8.685,9.067,9.673,10.13,10.50,11.23,11.90,12.67])

def gamma0(Z):
    return jnp.interp(Z,Z_eh,gamma0_eh)