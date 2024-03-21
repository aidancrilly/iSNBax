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

def calc_div_FicksLaw_op(Nx,d):
    """
    
    Calculates matrix representation of div ( - d grad (x)) operator
    
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

def divQ_to_Q(divQ,area):
    """
    
    Convert from div Q to Q assuming zero flux boundary
    N.B. input divQ is volume integral

    """
    Q = jnp.zeros_like(divQ)

    def Qintegral(carry,divQi):
        ix, Q_im1h = carry
        Q_ip1h = (divQi+area[ix]*Q_im1h)/area[ix+1]
        carry = ix+1,Q_ip1h
        return carry,Q_ip1h

    carry = 0, 0.0
    _,Q = jax.lax.scan(Qintegral,carry,divQ)

    return Q

def calc_local_SpitzerHarm_divQ(kappae,Te,args):
    area, vol, delta_x  = args['area'],args['vol'],args['delta_x']
    Nx = vol.shape[0]
    # Conductivities on the faces
    ke_ghost = jnp.concatenate([kappae[:1],kappae,kappae[-1:]])
    ke_face = 0.5*(ke_ghost[:-1]+ke_ghost[1:])
    rate_e = area*ke_face/delta_x
    
    SpitzerHarmQOperator,_,_,_ = calc_div_FicksLaw_op(Nx,rate_e)

    DivQsh = SpitzerHarmQOperator.mv(Te)

    return DivQsh

def calc_nonlocal_SNB_divQ(Te,kappae,mfp,args):
    """
    
    Calculate the non-local heat flow term, Eq. 10 of Cao et al.
    
    """
    area, vol, delta_x  = args['area'],args['vol'],args['delta_x']
    Nx = vol.shape[0]
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

        DivUgOperator,_,_,_ = calc_div_FicksLaw_op(Nx,rate_e_SNB)

        minusDivUg = DivUgOperator.mv(Te)
        
        # Mean free paths on the faces
        mfp_ghost = jnp.concatenate([mfp_g[:1],mfp_g,mfp_g[-1:]])
        mfp_SNB_face = 0.5*(mfp_ghost[:-1]+mfp_ghost[1:])
        Hgdivcoeff = area*(mfp_SNB_face/3.0)/delta_x

        # Assemble system of Hg solve
        HgDivOperator,_,_,_ = calc_div_FicksLaw_op(Nx,Hgdivcoeff)

        HgOperator = lx.DiagonalLinearOperator(vol/mfp_g) + HgDivOperator

        # Thomas Algorithm can be unstable, swap to LU
        HgOperator = lx.MatrixLinearOperator(HgOperator.as_matrix())

        Hg = lx.linear_solve(HgOperator, minusDivUg).value

        # Hg = jnp.matmul(jnp.linalg.pinv(HgOperator.as_matrix(),rcond=1e-5),minusDivUg)

        return Hg/mfp_g

    eta_gs = calc_eta_grp(args['Egb'][:-1],args['Egb'][1:],Te)
    # Make sure sum to 1
    norm_eta_g = jnp.sum(eta_gs,axis=0)
    norm_eta_g = jnp.where(norm_eta_g > 0, norm_eta_g, 1.0)
    eta_gs = eta_gs/norm_eta_g[None,:]

    mfp_gs = calc_mfp_grp(args['Egc'],Te,mfp)

    Hg_over_lambdag = jax.vmap(calc_Hg_over_lambdag,in_axes=(0,0),out_axes=0)(eta_gs,mfp_gs)

    minus_div_Q_nonlocal = vol*jnp.sum(Hg_over_lambdag,axis=0)

    return minus_div_Q_nonlocal

class iSNB_Solver(diffrax.AbstractSolver):
    """
    
    Cao et al.'s method for implicitly solving the SNB model
    
    """
    term_structure = dummy_term
    interpolation_cls = diffrax.LocalLinearInterpolation
    atol = 1e-4
    rtol = 1e-1
    itermax = 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0

        # Define variables that are static during iterative solve
        area, vol, delta_x  = args['area'],args['vol'],args['delta_x']
        Nx = vol.shape[0]
        IdentityOperator = lx.DiagonalLinearOperator(jnp.ones(Nx))

        S, Cv = args['S'], args['Cv']
        heat_capacity_e = Cv*vol

        def keep_iterating(val):
            _,_,_,iter,convergence_criterion = val
            
            return convergence_criterion # & iter < self.itermax

        def SNB_iteration(val):
            Te_prev, div_Q_correction, divQsh_prev, iter, _ = val
            
            Te_new, div_Q_correction, divQsh_new, convergence_criterion = single_implicit_iteration(Te_prev,div_Q_correction,divQsh_prev)
            iter = iter + 1

            return Te_new, div_Q_correction, divQsh_new, iter, convergence_criterion

        def single_implicit_iteration(Te_km1,div_Q_correction,divQsh_km1):
            kappae,thermal_mfp,effective_mfp = calc_transport_coeffs(args['Z'],args['ne'],Te_km1)
            # Conductivities on the faces
            ke_ghost = jnp.concatenate([kappae[:1],kappae,kappae[-1:]])
            ke_face = 0.5*(ke_ghost[:-1]+ke_ghost[1:])
            rate_e = area*ke_face/delta_x
            
            SpitzerHarmQOperator,Ash,Bsh,Csh = calc_div_FicksLaw_op(Nx,rate_e)

            # Diagonal
            AD = δt*Ash/heat_capacity_e
            # Lower diagonal
            BD = δt*Bsh/heat_capacity_e[1:]
            # Upper diagonal
            CD = δt*Csh/heat_capacity_e[:-1]

            DiffusionOperator = lx.TridiagonalLinearOperator(AD,BD,CD)

            LocalHeatOperator = IdentityOperator + DiffusionOperator

            vector = Te_0 + (δt / heat_capacity_e) * (vol * S + div_Q_correction)
            
            Te_k = lx.linear_solve(LocalHeatOperator, vector).value

            divQsh_k = SpitzerHarmQOperator.mv(Te_k)

            diff = jnp.abs(divQsh_k - divQsh_km1)
            scale = self.rtol * heat_capacity_e * Te_k / δt
            convergence_criterion = jnp.any(diff > scale)

            minus_div_Q_nonlocal = jax.lax.cond(convergence_criterion,
                                                calc_nonlocal_SNB_divQ,
                                                lambda x1,x2,x3,x4 : -div_Q_correction+divQsh_k,
                                                Te_k,kappae,effective_mfp,args)

            div_Q_correction = -minus_div_Q_nonlocal+divQsh_k

            return Te_k, div_Q_correction, divQsh_k, convergence_criterion

        Te_0 = y0
        div_Q_correction,divQsh = jnp.zeros(Nx),jnp.zeros(Nx)
        Te_1, div_Q_correction, _, _, _ = jax.lax.while_loop(keep_iterating, SNB_iteration, (Te_0, div_Q_correction, divQsh, 0, True))

        y1 = Te_1

        y_error = jnp.zeros_like(y0)
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        return None
    
    def save_function(self, t, y, args):

        kappae,_,mfp = calc_transport_coeffs(args['Z'],args['ne'],y)
        divQ = calc_nonlocal_SNB_divQ(y,kappae,mfp,args)
        Q = divQ_to_Q(divQ,args['area'])
        save_ys = jnp.column_stack([y,Q])

        return save_ys

def solve_iSNB(T0, dt0, ts, args, max_steps = 1000000):
    """
    
    Uses diffrax to integrate iSNB model in time
    
    """
    y0 = T0
    SNB_solver = iSNB_Solver()
    solution = diffrax.diffeqsolve(
            dummy_term(),
            solver=SNB_solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(ts=ts,fn=SNB_solver.save_function),
            max_steps=max_steps
        )
    return solution.ys[:,:,0],solution.ys[:,:,1]

# Assume batched axis is last axis
BatchediSNBSolve = jax.vmap(solve_iSNB,in_axes=(-1,-1,-1,-1),out_axes=-1)

class SpitzerHarm_Solver(diffrax.AbstractSolver):
    """
    
    Implicitly solving the local heat flow Spitzer-Harm model
    
    """
    term_structure = dummy_term
    interpolation_cls = diffrax.LocalLinearInterpolation
    atol = 1e-4
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
        IdentityOperator = lx.DiagonalLinearOperator(jnp.ones(Nx))

        S, Cv = args['S'], args['Cv']
        heat_capacity_e = Cv*vol

        def keep_iterating(val):
            _, _, iter, convergence_criterion = val
            return convergence_criterion # & iter < self.itermax # uncommenting this makes it really slow....

        def SH_iteration(val):
            Te_prev, Qsh_prev, iter, _ = val
            
            Te_new, Qsh_new, convergence_criterion = single_implicit_iteration(Te_prev,Qsh_prev)
            iter = iter + 1

            return Te_new, Qsh_new, iter, convergence_criterion

        def single_implicit_iteration(Te_km1,divQsh_km1):
            kappae,_,_ = calc_transport_coeffs(args['Z'],args['ne'],Te_km1)
            # Conductivities on the faces
            ke_ghost = jnp.concatenate([kappae[:1],kappae,kappae[-1:]])
            ke_face = 0.5*(ke_ghost[:-1]+ke_ghost[1:])
            rate_e = area*ke_face/delta_x
            
            SpitzerHarmQOperator,Ash,Bsh,Csh = calc_div_FicksLaw_op(Nx,rate_e)

            # Diagonal
            AD = δt*Ash/heat_capacity_e
            # Lower diagonal
            BD = δt*Bsh/heat_capacity_e[1:]
            # Upper diagonal
            CD = δt*Csh/heat_capacity_e[:-1]

            DiffusionOperator = lx.TridiagonalLinearOperator(AD,BD,CD)

            LocalHeatOperator = IdentityOperator + DiffusionOperator

            vector = Te_0 + δt * (vol * S / heat_capacity_e)
            
            Te_k = lx.linear_solve(LocalHeatOperator, vector).value

            divQsh_k = SpitzerHarmQOperator.mv(Te_k)

            diff = jnp.abs(divQsh_k - divQsh_km1)
            scale = self.rtol * heat_capacity_e * Te_k / δt
            convergence_criterion = jnp.any(diff > scale)

            return Te_k, divQsh_k, convergence_criterion
        
        Te_0 = y0
        divQsh = jnp.zeros_like(Te_0)
        Te_1, _, _, _ = jax.lax.while_loop(keep_iterating, SH_iteration, (Te_0, divQsh, 0, True))

        y1 = Te_1

        y_error = jnp.zeros_like(y0)
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        return None
    
    def save_function(self, t, y, args):

        kappae,_,_ = calc_transport_coeffs(args['Z'],args['ne'],y)
        divQ = calc_local_SpitzerHarm_divQ(kappae,y,args)
        Q = divQ_to_Q(divQ,args['area'])
        save_ys = jnp.column_stack([y,Q])

        return save_ys

def solve_SpitzerHarm(T0, dt0, ts, args, max_steps = 1000000):
    """
    
    Uses diffrax to integrate iSNB model in time
    
    """
    y0 = T0
    SH_solver = SpitzerHarm_Solver()
    solution = diffrax.diffeqsolve(
            dummy_term(),
            solver=SH_solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(ts=ts,fn=SH_solver.save_function),
            max_steps=max_steps
        )
    return solution.ys[:,:,0],solution.ys[:,:,1]

# Assume batched axis is last axis
BatchedSpitzerHarmSolve = jax.vmap(solve_SpitzerHarm,in_axes=(-1,-1,-1,-1),out_axes=-1)