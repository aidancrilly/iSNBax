import diffrax
import lineax as lx
from utils import *

class dummy_term(diffrax.AbstractTerm):
    """
    
    To use diffrax.diffeqsolve we need to pass a "term" but we don't need this
    
    """
    def vf(self, t, y, args):
        return None
    
    def contr(self, t0, t1):
        return t1 - t0
    
    def prod(self, vf, control):
        return None

def calc_minus_div_op(Nx,d):
    """
    
    Calculates matrix representation of - div d grad (x) operator
    
    """
    A = jnp.zeros(Nx)
    # Diagonal
    A = A.at[1:-1].add(d[2:-1]+d[1:-2])
    # Right hand boundary
    A = A.at[-1].add(d[-2])
    # Left hand boundary
    A = A.at[0].add(d[1])
    # Lower diagonal
    B = -d[1:-1]
    # Upper diagonal
    C = -d[1:-1]

    return lx.TridiagonalLinearOperator(A,B,C),A,B,C

class iSNB_Solver(diffrax.AbstractSolver):
    """
    
    Cao et al.'s method for implicitly solving the SNB model
    
    """
    term_structure = dummy_term
    interpolation_cls = diffrax.LocalLinearInterpolation
    atol = 1e-2
    rtol = 1e-3
    itermax = 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0

        # Define variables that are static during iterative solve
        area, vol, delta_x  = args['area'],args['vol'],args['delta_x']
        Nx = vol.shape[0]
        Ngrp = args['Ngrp']
        IdentityOperator = lx.DiagonalLinearOperator(jnp.ones(Nx))

        S, Cv = args['S'], args['Cv']
        heat_capacity_e = Cv*vol

        def keep_iterating(val):
            _,_,iter,convergence_criterion = val
            
            return convergence_criterion # & iter < self.itermax

        def SNB_iteration(val):
            Te_prev, div_Q_correction, iter, _ = val
            
            Te_new, div_Q_correction, convergence_criterion = single_implicit_iteration(Te_prev,div_Q_correction)
            iter = iter + 1

            return Te_new, div_Q_correction, iter, convergence_criterion

        def single_implicit_iteration(Te_km1,div_Q_correction):
            kappae,thermal_mfp = calc_transport_coeffs(args['Z'],args['ne'],Te_km1)
            # Conductivities on the faces
            ke_ghost = jnp.concatenate([kappae[:1],kappae,kappae[-1:]])
            ke_face = 0.5*(ke_ghost[:-1]+ke_ghost[1:])
            rate_e = area*ke_face/delta_x
            
            SpitzerHarmQOperator,Ash,Bsh,Csh = calc_minus_div_op(Nx,rate_e)

            # Diagonal
            AD = δt*Ash/heat_capacity_e
            # Lower diagonal
            BD = δt*Bsh/heat_capacity_e[1:]
            # Upper diagonal
            CD = δt*Csh/heat_capacity_e[:-1]

            DiffusionOperator = lx.TridiagonalLinearOperator(AD,BD,CD)

            LocalHeatOperator = IdentityOperator + DiffusionOperator

            vector = Te_0 + (δt / heat_capacity_e) * (S + div_Q_correction)
            
            Te_k = lx.linear_solve(LocalHeatOperator, vector).value

            diff = jnp.abs(Te_k - Te_km1)
            max_Te = jnp.maximum(jnp.abs(Te_km1), jnp.abs(Te_k))
            scale = self.atol + self.rtol * max_Te
            convergence_criterion = jnp.any(diff > scale)

            minusDivQsh = SpitzerHarmQOperator.mv(Te_k)

            minus_div_Q_nonlocal = jax.lax.cond(convergence_criterion,
                                                calc_nonlocal_Q,
                                                lambda x,y,z : jnp.zeros_like(x),
                                                Te_k,kappae,thermal_mfp)

            div_Q_correction = minus_div_Q_nonlocal-minusDivQsh

            return Te_k, div_Q_correction, convergence_criterion
        
        def calc_nonlocal_Q(Te,kappae,thermal_mfp):
            """
            
            Calculate the non-local heat flow term, Eq. 10 of Cao et al.
            
            """
            # Can vmap over energy groups
            def calc_Hg_over_lambdag(eta_g,mfp_g):
                """
                
                Calculate H_g/lambda_g for a single energy energy group
                
                """

                kappae_SNB = kappae*eta_g
                # Conductivities on the faces
                ke_ghost = jnp.concatenate([kappae_SNB[:1],kappae_SNB,kappae_SNB[-1:]])
                ke_SNB_face = 0.5*(ke_ghost[:-1]+ke_ghost[1:])
                rate_e_SNB = area*ke_SNB_face/delta_x

                DivUgOperator,_,_,_ = calc_minus_div_op(Nx,rate_e_SNB)

                minusDivUg = DivUgOperator.mv(Te)
                
                # Mean free paths on the faces
                mfp_ghost = jnp.concatenate([mfp_g[:1],mfp_g,mfp_g[-1:]])
                mfp_SNB_face = 0.5*(mfp_ghost[:-1]+mfp_ghost[1:])
                Hgdivcoeff = area*mfp_SNB_face/delta_x/3.0

                # Assemble system of Hg solve
                HgDivOperator,_,_,_ = calc_minus_div_op(Nx,Hgdivcoeff)

                HgOperator = lx.DiagonalLinearOperator(vol/mfp_g) + HgDivOperator

                # Thomas Algorithm can be unstable, swap to LU
                HgOperator = lx.MatrixLinearOperator(HgOperator.as_matrix())

                # Hg = lx.linear_solve(HgOperator, minusDivUg).value

                Hg = jnp.matmul(jnp.linalg.pinv(HgOperator.as_matrix(),rcond=1e-3),minusDivUg)

                return vol*Hg/mfp_g

            eta_g = calc_eta_grp(args['Egb'][:-1],args['Egb'][1:],Te)
            # Make sure sum to 1
            norm_eta_g = jnp.sum(eta_g,axis=0)
            norm_eta_g = jnp.where(norm_eta_g > 0, norm_eta_g, 1.0)
            eta_g = eta_g/norm_eta_g[None,:]

            mfp_g = calc_mfp_grp(args['Egc'],Te,thermal_mfp)
            # mfp_g = jnp.where(mfp_g > 1e-6, mfp_g, 1e-6)
            # mfp_g = jnp.where(mfp_g < 1e-5, mfp_g, 1e-5)
            # mfp_g = 10e-6*jnp.ones_like(mfp_g)

            Hg_over_lambdag = jax.vmap(calc_Hg_over_lambdag,in_axes=(0,0),out_axes=0)(eta_g,mfp_g)

            minus_div_Q_nonlocal = jnp.sum(Hg_over_lambdag,axis=0)

            return minus_div_Q_nonlocal

        Te_0 = y0
        div_Q_correction = jnp.zeros(Nx)
        Te_1, _, _, _ = jax.lax.while_loop(keep_iterating, SNB_iteration, (Te_0, div_Q_correction, 0, True))

        Te_error = jnp.zeros_like(Te_0)
        dense_info = dict(y0=Te_0, y1=Te_1)

        solver_state = None
        result = diffrax.RESULTS.successful
        return Te_1, Te_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        return None
    
def solve_iSNB(y0, dt0, ts, args, max_steps = 1000000):
    """
    
    Uses diffrax to integrate iSNB model in time
    
    """
    solution = diffrax.diffeqsolve(
            dummy_term(),
            solver=iSNB_Solver(),
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps
        )
    return solution.ys.reshape(ts.shape[0],args['vol'].shape[0])

# Assume batched axis is last axis
BatchediSNBSolve = jax.vmap(solve_iSNB,in_axes=(-1,-1,-1,-1),out_axes=-1)

class SpitzerHarm_Solver(diffrax.AbstractSolver):
    """
    
    Implicitly solving the local heat flow Spitzer-Harm model
    
    """
    term_structure = dummy_term
    interpolation_cls = diffrax.LocalLinearInterpolation
    atol = 1e-2
    rtol = 1e-3
    itermax = 10

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0

        # Define variables that are static during iterative solve
        area, vol, delta_x  = args['area'],args['vol'],args['delta_x']
        Nx = vol.shape[0]
        Ngrp = args['Ngrp']
        IdentityOperator = lx.DiagonalLinearOperator(jnp.ones(Nx))

        S, Cv = args['S'], args['Cv']
        heat_capacity_e = Cv*vol

        def keep_iterating(val):
            _, iter, convergence_criterion = val
            return convergence_criterion # & iter < self.itermax # uncommenting this makes it really slow....

        def SH_iteration(val):
            Te_prev, iter, _ = val
            
            Te_new, convergence_criterion = single_implicit_iteration(Te_prev)
            iter = iter + 1

            return Te_new, iter, convergence_criterion

        def single_implicit_iteration(Te_km1):
            kappae,_ = calc_transport_coeffs(args['Z'],args['ne'],Te_km1)
            # Conductivities on the faces
            ke_ghost = jnp.concatenate([kappae[:1],kappae,kappae[-1:]])
            ke_face = 0.5*(ke_ghost[:-1]+ke_ghost[1:])
            rate_e = area*ke_face/delta_x
            
            _,Ash,Bsh,Csh = calc_minus_div_op(Nx,rate_e)

            # Diagonal
            AD = δt*Ash/heat_capacity_e
            # Lower diagonal
            BD = δt*Bsh/heat_capacity_e[1:]
            # Upper diagonal
            CD = δt*Csh/heat_capacity_e[:-1]

            DiffusionOperator = lx.TridiagonalLinearOperator(AD,BD,CD)

            LocalHeatOperator = IdentityOperator + DiffusionOperator

            vector = Te_0 + δt * S
            
            Te_k = lx.linear_solve(LocalHeatOperator, vector).value

            diff = jnp.abs(Te_k - Te_km1)
            max_Te = jnp.maximum(jnp.abs(Te_km1), jnp.abs(Te_k))
            scale = self.atol + self.rtol * max_Te
            convergence_criterion = jnp.any(diff > scale)

            return Te_k, convergence_criterion

        Te_0 = y0
        Te_1, _, _ = jax.lax.while_loop(keep_iterating, SH_iteration, (Te_0, 0, True))

        Te_error = jnp.zeros_like(Te_0)
        dense_info = dict(y0=Te_0, y1=Te_1)

        solver_state = None
        result = diffrax.RESULTS.successful
        return Te_1, Te_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        return None
    
def solve_SpitzerHarm(y0, dt0, ts, args, max_steps = 1000000):
    """
    
    Uses diffrax to integrate iSNB model in time
    
    """
    solution = diffrax.diffeqsolve(
            dummy_term(),
            solver=SpitzerHarm_Solver(),
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps
        )
    return solution.ys.reshape(ts.shape[0],args['vol'].shape[0])

# Assume batched axis is last axis
BatchedSpitzerHarmSolve = jax.vmap(solve_SpitzerHarm,in_axes=(-1,-1,-1,-1),out_axes=-1)