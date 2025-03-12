import os
import numpy as np

from functions import (
    v_to_qdot,
    R_to_quat,
    Phase,
    phase_composer,
    is_in_contact,
    node_is_in_contact,
    Parameters,
    Model,
    Environment,
)

import casadi as cs
import pinocchio as pin
from pinocchio import casadi as cpin

## Parameter Setup ##

# Find urdf file
path_to_curr_folder = os.path.dirname(os.path.realpath(__file__))
urdffile = os.path.join(path_to_curr_folder, "urdf/g1_23dof.urdf")

params = Parameters()
params.total_duration = 0.1
params.num_shooting_states = 10
params.num_rollout_states = 1
params.model_urdf_path = urdffile
params.contact_ee_names = ["lf", "rf"]
params.ee_phase_sequence = {"lf": list((10,)), "rf": list((10,))}
params.is_init_contact = {"lf": True, "rf": True}
params.num_contact_points_per_ee = {
    "lf": 4,
    "rf": 4,
}
params.dim_f_ext = 3
params.q0 = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    -0.6,
    0.0,
    0.0,
    1.2,
    -0.6,
    0.0,
    -0.6,
    0.0,
    0.0,
    1.2,
    -0.6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

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

q0 = model.q0
q1 = [0.0, 0.0]
# q_mid = [0.0, 0.0, 1.2]

v0 = [0.0 for _ in range(nv)]
a0 = [0.0 for _ in range(nv)]
fc0 = [0.0, 0.0, 34.0 * env.grav / 8]

q_init = q0
v_init = v0
q_last = q1 + q0[2:]
v_last = v0

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

tau_min = [0.0 for _ in range(7)] + model.lower_joint_effort_lim()
tau_max = [0.0 for _ in range(7)] + model.upper_joint_effort_lim()

fc_min = [-cs.inf for _ in range(params.dim_f_ext)]
fc_max = [cs.inf for _ in range(params.dim_f_ext)]

# Start with an empty NLP
w = []  # variables
w0 = []  # variable initial guess
lbw = []  # variable lower bound
ubw = []  # varialble upper bound
J = 0  # cost
g = []  # constraints
lbg = []  # lower bound for contstraints
ubg = []  # upper bound for contraints

# Initial state
x = cs.SX.sym("x0", nx, 1)
w += [x]
lbw += q_init + v_init
ubw += q_init + v_init
# lbw += q_init[:7] + q_min[7:] + v_init
# ubw += q_init[:7] + q_max[7:] + v_init
w0 += q0 + v0

t = 0.0
for i in range(params.num_shooting_states):

    # Add postural cost
    # J += 1e3 * cs.sumsqr(x[7:nq] - q_init[7:])

    # Integrate for num_rollout_states
    for j in range(params.num_rollout_states):
        # Add control variable
        a = cs.SX.sym("a_" + str(i) + "_" + str(j), nv, 1)
        w += [a]
        lbw += a_min
        ubw += a_max
        w0 += a0

        # J += 1.0 * cs.dot(a, a)
        # Add centroidal dynamics in cost
        # J += 1e3 * cs.sumsqr(model.angular_momentum(x[:nq], x[nq:], a))

        JtF_sum = cs.SX.zeros(29, 1)
        for frame_name, phase_seq in frame_phase_sequence.items():
            # Determine if ee is in contact
            in_contact = node_is_in_contact(
                i, phase_seq, frame_is_init_contact[frame_name]
            )
            # print(frame_name, "(", i, "): ", ("contact" if in_contact else "swing"))

            # if not in_contact:
            #     J += 3e5 * cs.sumsqr(x[2] - (q0[2] + 0.2))

            if in_contact:

                # Create force variable
                fc = cs.SX.sym(
                    "fc_" + frame_name + "_" + str(i) + "_" + str(j),
                    params.dim_f_ext,
                    1,
                )
                w += [fc]
                lbw += fc_min
                ubw += fc_max
                w0 += fc0

                # J += cs.dot(fc, fc)

                # End effector on ground
                ee_pos = model.frame_dist_from_ground(frame_name, x[:nq])
                g += [ee_pos[2] - env.ground_z]
                lbg += [0.0]
                ubg += [0.0]

                # Zero velocity in contact
                jac = model.frame_jacobian(frame_name, x[:nq])
                g += [jac[0:3, :] @ x[nq:]]
                lbg += [0.0 for _ in range(3)]
                ubg += [0.0 for _ in range(3)]

                ## Zero acceleration ##
                # jacdot = model.frame_jacobian_time_var(frame_name, x[:nq])
                # g += [jac[0:3, :] @ a + jacdot[0:3, :] @ x[nq:]]
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

        g += [model.inverse_dynamics(x[:nq], x[nq:], a, JtF_sum)]
        lbg += tau_min
        ubg += tau_max

        # Contact constraints

        # Semi-implicit Euler integration
        dt = dt0
        # dt = cs.SX.sym("dt_" + str(i), 1)
        # w += [dt]
        # lbw += [dt_min]
        # ubw += [dt_max]
        # w0 += [dt0]

        v = x[nq:] + a * dt
        p = x[:nq] + v_to_qdot(nq, x[:nq], v) * dt
        x = cs.vertcat(p, v)

        # t += dt0

    # Add intermediate state as variable
    xk = cs.SX.sym("x" + str(i + 1), nx, 1)
    w += [xk]
    if i == params.num_shooting_states - 1:
        lbw += q1 + q0[2:] + v_last
        ubw += q1 + q0[2:] + v_last
        # lbw += q_min + v_last
        # ubw += q_max + v_last
    # elif i == num_shooting_states / 2:
    #     lbw += q_mid + q_min[3:] + v_last
    #     ubw += q_mid + q_max[3:] + v_last
    else:
        lbw += q_min + v_min
        ubw += q_max + v_max
    w0 += q0 + v0

    # Add defect constraint
    g += [x - xk]
    lbg += [0.0 for _ in range(nx)]
    ubg += [0.0 for _ in range(nx)]

    # Perform same iteration for new node
    x = xk

# g += [dt_sum - total_duration]
# lbg += [0.0]
# ubg += [0.0]

# Assert vector sizes are correct

assert cs.vertcat(*w).shape[0] == len(w0) == len(lbw) == len(ubw)
assert cs.vertcat(*g).shape[0] == len(lbg) == len(ubg)

# Create an NLP solver
opts = {
    "ipopt.linear_solver": "ma57",
    "ipopt.tol": 0.0001,
    "ipopt.constr_viol_tol": 0.0001,
    "ipopt.max_iter": 2000,
}
prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g)}
solver = cs.nlpsol("solver", "ipopt", prob, opts)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol["x"].full().flatten()

total_ee_contact_points = 0
for ee in params.contact_ee_names:
    total_ee_contact_points += params.num_contact_points_per_ee[ee]

nc = total_ee_contact_points * params.dim_f_ext
nw = nx + params.num_rollout_states * (nv + nc) + 1

c_idx = 0
x = np.zeros((nx, params.num_shooting_states + 1))
for i in range(params.num_shooting_states + 1):
    start = i * nw
    end = start + nx
    x[:, c_idx] = w_opt[start:end]
    c_idx += 1
np.savetxt("x.csv", x, fmt="%.2f", delimiter="\t,")

# Acceleration
a = np.zeros((nv, params.num_shooting_states * params.num_rollout_states))

c_idx = 0
for i in range(params.num_shooting_states):
    for j in range(params.num_rollout_states):
        start = i * nw + nx
        end = start + j * (nv + nc) + nv
        a[:, c_idx] = w_opt[start:end]
        c_idx += 1

np.savetxt("a.csv", a, delimiter=",")

# Contact forces
fc = np.zeros((nc, params.num_shooting_states * params.num_rollout_states))

c_idx = 0
for i in range(params.num_shooting_states):
    for j in range(params.num_rollout_states):
        start = i * nw + nx + nv
        end = start + j * (nv + nc) + nc
        fc[:, c_idx] = w_opt[start:end]
        c_idx += 1

np.savetxt("fc.csv", fc, delimiter=",")


# Dts
dts = np.zeros((1, params.num_shooting_states * params.num_rollout_states))

c_idx = 0
for i in range(params.num_shooting_states):
    for j in range(params.num_rollout_states):
        start = i * nw + nw - 1
        end = start + 1
        dts[:, c_idx] = w_opt[start:end]
        c_idx += 1

np.savetxt("dt.csv", dts, delimiter=",")
