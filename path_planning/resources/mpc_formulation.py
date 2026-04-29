"""
MPC formulation for kinematic bicycle model trajectory tracking.

Model
-----
States  : x, y, theta   (world-frame position and heading)
Input   : delta          (front-wheel steering angle)
Speed   : fixed constant v (set at construction time)

Dynamics (discrete, Euler integration at step dt)
    x[k+1]     = x[k]     + v * cos(theta[k]) * dt
    y[k+1]     = y[k]     + v * sin(theta[k]) * dt
    theta[k+1] = theta[k] + v * tan(delta[k]) / L * dt

A discrete model is used instead of continuous+collocation to keep the
NLP small and solvable in real-time (<50 ms per step).

Reference
---------
Provided as time-varying parameters (TVP): x_ref, y_ref, theta_ref
for each step k = 0 … N in the prediction horizon.

Usage
-----
    from path_planning.resources.mpc_formulation import build_mpc

    mpc, tvp_template = build_mpc(
        wheelbase=0.34, speed=1.0, max_steer=0.34, N=20, dt=0.1
    )

    # Before each solve, write your reference into tvp_template:
    #   tvp_template['_tvp', k, 'x_ref']     = ...
    #   tvp_template['_tvp', k, 'y_ref']     = ...
    #   tvp_template['_tvp', k, 'theta_ref'] = ...
    # then call mpc.make_step(x0).
"""

import do_mpc
from casadi import cos, sin, tan


def build_mpc(
    wheelbase: float,
    speed: float,
    max_steer: float,
    N: int,
    dt: float,
    Q_pos: float = 10.0,
    Q_theta: float = 2.0,
    R_delta: float = 4.0,
) -> tuple:
    """
    Build and return a configured do_mpc MPC controller.

    Parameters
    ----------
    wheelbase : float
        Distance between front and rear axles [m].
    speed : float
        Fixed longitudinal speed used in the model [m/s].
    max_steer : float
        Maximum steering angle magnitude [rad].
    N : int
        Prediction horizon (number of steps).
    dt : float
        Time step per horizon step [s].
    Q_pos : float
        Weight on (x, y) position tracking error.
    Q_theta : float
        Weight on heading tracking error (uses 1-cos for proper wrapping).
    R_delta : float
        Input rate penalty weight on delta.

    Returns
    -------
    mpc : do_mpc.controller.MPC
        Fully configured and setup MPC controller.
    tvp_template : do_mpc.core.StructureBase
        TVP template to fill before each solve call.
    """

    # ------------------------------------------------------------------
    # 1. Model  (discrete — avoids collocation overhead)
    # ------------------------------------------------------------------
    model = do_mpc.model.Model('discrete')

    # States
    px    = model.set_variable('_x', 'x')
    py    = model.set_variable('_x', 'y')
    theta = model.set_variable('_x', 'theta')

    # Input
    delta = model.set_variable('_u', 'delta')

    # Time-varying reference parameters
    x_ref  = model.set_variable('_tvp', 'x_ref')
    y_ref  = model.set_variable('_tvp', 'y_ref')
    th_ref = model.set_variable('_tvp', 'theta_ref')

    # Euler-discretised bicycle kinematics
    model.set_rhs('x',     px    + speed * cos(theta) * dt)
    model.set_rhs('y',     py    + speed * sin(theta) * dt)
    model.set_rhs('theta', theta + speed * tan(delta) / wheelbase * dt)

    model.setup()

    # ------------------------------------------------------------------
    # 2. MPC controller
    # ------------------------------------------------------------------
    mpc = do_mpc.controller.MPC(model)

    mpc.set_param(
        n_horizon           = N,
        t_step              = dt,
        state_discretization= 'discrete',
        store_full_solution = False,
        nlpsol_opts         = {
            'ipopt.print_level'              : 0,
            'ipopt.max_iter'                 : 100,
            'ipopt.warm_start_init_point'    : 'yes',
            'ipopt.warm_start_bound_push'    : 1e-8,
            'ipopt.warm_start_mult_bound_push': 1e-8,
            'print_time'                     : 0,
        },
    )

    # ------------------------------------------------------------------
    # 3. Cost function
    # ------------------------------------------------------------------
    # Position error (squared Euclidean)
    pos_err = Q_pos * ((px - x_ref)**2 + (py - y_ref)**2)

    # Heading error: (1 - cos(angle_diff)) is bounded [0, 2], wraps correctly
    head_err = Q_theta * (1 - cos(theta - th_ref))

    lterm = pos_err + head_err   # stage cost
    mterm = pos_err + head_err   # terminal cost

    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(delta=R_delta)

    # ------------------------------------------------------------------
    # 4. Constraints
    # ------------------------------------------------------------------
    mpc.bounds['lower', '_u', 'delta'] = -max_steer
    mpc.bounds['upper', '_u', 'delta'] =  max_steer

    # ------------------------------------------------------------------
    # 5. TVP function — caller fills template in-place before each solve
    # ------------------------------------------------------------------
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(_t_now):
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc, tvp_template
