import os
import numpy as np

from functions import (
    v_to_qdot,
    R_to_quat,
    phase_composer,
    is_in_contact,
    node_is_in_contact,
    Parameters,
    Model,
    Environment,
    addFrictionConeConstraint,
)

from model_configs import G1_29DOF_PARAMS, TALOS_PARAMS

import casadi as cs
import pinocchio as pin
from pinocchio import casadi as cpin

## Parameter Setup ##

params = G1_29DOF_PARAMS
# params = TALOS_PARAMS

params.assert_validity()

## Load Model ##
model = Model(params)

## Create environment
env = Environment()

nq = model.nq()
nv = model.nv()
nx = nq + nv

frame_phase_sequence = {
    "left_foot_upper_right": params.ee_phase_sequence["lf"],
    "left_foot_lower_right": params.ee_phase_sequence["lf"],
    "left_foot_upper_left": params.ee_phase_sequence["lf"],
    "left_foot_lower_left": params.ee_phase_sequence["lf"],
    "right_foot_upper_right": params.ee_phase_sequence["rf"],
    "right_foot_lower_right": params.ee_phase_sequence["rf"],
    "right_foot_upper_left": params.ee_phase_sequence["rf"],
    "right_foot_lower_left": params.ee_phase_sequence["rf"],
}

frame_is_init_contact = {
    "left_foot_upper_right": params.is_init_contact["lf"],
    "left_foot_lower_right": params.is_init_contact["lf"],
    "left_foot_upper_left": params.is_init_contact["lf"],
    "left_foot_lower_left": params.is_init_contact["lf"],
    "right_foot_upper_right": params.is_init_contact["rf"],
    "right_foot_lower_right": params.is_init_contact["rf"],
    "right_foot_upper_left": params.is_init_contact["rf"],
    "right_foot_lower_left": params.is_init_contact["rf"],
}

a0 = [0.0 for _ in range(nv)]
fc0 = [0.0, 0.0, model.total_mass() * env.grav / 8]

q_init = model.q_init
v_init = [0.0 for _ in range(nv)]
q_last = model.q_last
v_last = [0.0 for _ in range(nv)]

## End Model Stuff ##

## Durations ##
dt0 = params.total_duration / params.num_shooting_states / params.num_rollout_states
dt_min = 0.02
dt_max = 0.1

## End Durations ## ## Add variables

q_min = model.lower_pos_lim()
q_max = model.upper_pos_lim()
v_min = model.lower_vel_lim()
v_max = model.upper_vel_lim()
a_min = [-cs.inf for _ in range(nv)]
a_max = [cs.inf for _ in range(nv)]

tau_min = model.lower_joint_effort_lim()
tau_max = model.upper_joint_effort_lim()

fc_min = [-cs.inf for _ in range(params.dim_f_ext)]
fc_max = [cs.inf for _ in range(params.dim_f_ext)]

## Start with an empty NLP
w = []  # variables
w0 = []  # variable initial guess
lbw = []  # variable lower bound
ubw = []  # varialble upper bound
J = 0  # cost
g = []  # constraints
lbg = []  # lower bound for contstraints
ubg = []  # upper bound for contraints

## Initial state
x = cs.SX.sym("x0", nx, 1)
w += [x]
lbw += q_init + v_init
ubw += q_init + v_init
w0 += q_init + v_init
for i in range(params.num_shooting_states):
    ## State related costs
    # J += 1e-5 * cs.dot(x, x)  # Regularisation
    # J += 1e-10 * cs.sumsqr(x[7:nq] - q_init[7:])  # Postural
    for j in range(params.num_rollout_states):
        ## Control acceleration variable
        a = cs.SX.sym("a_" + str(i) + "_" + str(j), nv, 1)
        w += [a]
        lbw += a_min
        ubw += a_max
        w0 += a0
        ## Control acceleration related costs
        J += cs.dot(a, a)  # Regularisation
        # J += cs.sumsqr(model.angular_momentum(x[:nq], x[nq:], a))  # Centroidal dynamics

        JtF_sum = cs.SX.zeros(model.nv(), 1)
        for frame_name, phase_seq in frame_phase_sequence.items():
            # Determine if ee is in contact
            in_contact = node_is_in_contact(
                i, phase_seq, frame_is_init_contact[frame_name]
            )

            if in_contact:
                ## Control contact force variable
                fc = cs.SX.sym(
                    "fc_" + frame_name + "_" + str(i) + "_" + str(j),
                    params.dim_f_ext,
                    1,
                )
                w += [fc]
                lbw += fc_min
                ubw += fc_max
                w0 += fc0

                ## Force related costs
                J += 1e-6 * cs.dot(fc, fc)  # Regularisation

                ## End effector on ground
                ee_pos = model.frame_dist_from_ground(frame_name, x[:nq])
                g += [ee_pos[2] - env.ground_z]
                lbg += [0.0]
                ubg += [0.0]

                ## Zero velocity in contact
                jac = model.frame_jacobian(frame_name, x[:nq])
                # g += [jac[0:3, :] @ x[nq:]]
                g += [model.frame_velocity(frame_name, x[:nq]).linear]
                lbg += [0.0 for _ in range(3)]
                ubg += [0.0 for _ in range(3)]

                ## Zero acceleration in contact
                # jacdot = model.frame_jacobian_time_var(frame_name, x[:nq])
                # g += [jac[0:3, :] @ a + jacdot[0:3, :] @ x[nq:]]
                # g += [model.frame_acceleration(frame_name, x[:nq]).linear]
                # lbg += [0.0 for _ in range(3)]
                # ubg += [0.0 for _ in range(3)]

                addFrictionConeConstraint(env, fc, g, lbg, ubg)

                JtF = cs.mtimes(jac[0:3, :].T, fc)
                JtF_sum += JtF
            else:
                # End effector above ground
                ee_pos = model.frame_dist_from_ground(frame_name, x[:nq])
                g += [ee_pos[2] - env.ground_z]
                lbg += [0.0]
                ubg += [cs.inf]

        ## Add dynamics constraint
        g += [model.inverse_dynamics(x[:nq], x[nq:], a, JtF_sum)]
        lbg += tau_min
        ubg += tau_max

        ## Dt variable
        dt = dt0
        if params.opt_dt:
            dt = cs.SX.sym("dt_" + str(i), 1)
            w += [dt]
            lbw += [dt_min]
            ubw += [dt_max]
            w0 += [dt0]

        # Semi-implicit Euler integration
        v = x[nq:] + a * dt
        # p = cpin.integrate(model.cmodel, x[:nq], v * dt)
        p = model.integrate(x[:nq], v * dt)
        x = cs.vertcat(p, v)

    ## Add intermediate state as variable
    xk = cs.SX.sym("x" + str(i + 1), nx, 1)
    w += [xk]
    if i == params.num_shooting_states - 1:
        lbw += q_last + v_last
        ubw += q_last + v_last
    # elif i == params.num_shooting_states / 2:
    #     lbw += q_init[0:2] + [q_init[2] - 0.05] + q_min[3:] + v_last
    #     ubw += q_init[0:2] + [q_init[2] - 0.05] + q_max[3:] + v_last
    else:
        lbw += q_min + v_min
        ubw += q_max + v_max
    w0 += q_init + v_init

    ## Defect constraint
    g += [x - xk]
    # g += [model.difference(xk[:nq], x[:nq])]
    # g += [xk[nq:] - x[nq:]]
    lbg += [0.0 for _ in range(nx)]
    ubg += [0.0 for _ in range(nx)]

    ## Current variable = next variable
    x = xk

# g += [dt_sum - total_duration]
# lbg += [0.0]
# ubg += [0.0]


## Assert vector sizes are correct
assert cs.vertcat(*w).shape[0] == len(w0) == len(lbw) == len(ubw)
assert cs.vertcat(*g).shape[0] == len(lbg) == len(ubg)

## Create an NLP solver
opts = {
    "ipopt.linear_solver": "ma57",
    "ipopt.tol": 0.0001,
    "ipopt.constr_viol_tol": 0.0001,
    "ipopt.max_iter": 3000,
}
prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g)}
solver = cs.nlpsol("solver", "ipopt", prob, opts)

## Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol["x"].full().flatten()

## Save to csv files
total_ee_contact_points = 0
for ee in params.contact_ee_names:
    total_ee_contact_points += params.num_contact_points_per_ee[ee]

x = np.zeros((nx, params.num_shooting_states + 1))
a = np.zeros((nv, params.num_shooting_states * params.num_rollout_states))
fc = np.zeros(
    (
        total_ee_contact_points * params.dim_f_ext,
        params.num_shooting_states * params.num_rollout_states,
    )
)
dts = np.zeros((1, params.num_shooting_states * params.num_rollout_states))


w_opt_idx = 0
fc_idx = 0
col_idx = 0

for i in range(params.num_shooting_states):
    # x solution
    x[:, col_idx] = w_opt[w_opt_idx : w_opt_idx + nx]
    w_opt_idx += nx

    for j in range(params.num_rollout_states):
        # a solution
        a[:, col_idx] = w_opt[w_opt_idx : w_opt_idx + nv]
        w_opt_idx += nv

        fc_idx = 0
        for frame_name, phase_seq in frame_phase_sequence.items():
            # Determine if ee is in contact
            in_contact = node_is_in_contact(
                i, phase_seq, frame_is_init_contact[frame_name]
            )
            if in_contact:
                fc[fc_idx : fc_idx + params.dim_f_ext, col_idx] = w_opt[
                    w_opt_idx : w_opt_idx + params.dim_f_ext
                ]
                fc_idx += params.dim_f_ext
                w_opt_idx += params.dim_f_ext
            else:
                fc[fc_idx : fc_idx + params.dim_f_ext, col_idx] = np.zeros(
                    params.dim_f_ext
                )
                fc_idx += params.dim_f_ext

    ## TODO: Add integrated states in between

    # dts[:, col_idx] = w_opt[w_opt_idx : w_opt_idx + 1]
    # w_opt_idx += nv
    dts[:, col_idx] = dt0

    col_idx += 1

# Final x solution
x[:, col_idx] = w_opt[w_opt_idx:]

np.savetxt("csv/x.csv", x, delimiter="\t,")
np.savetxt("csv/a.csv", a, delimiter=",")
np.savetxt("csv/fc.csv", fc, delimiter=",")
np.savetxt("csv/dts.csv", dts, delimiter=",")
