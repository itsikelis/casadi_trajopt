import os
import numpy as np

from functions import (
    v_to_qdot,
    R_to_quat,
    Phase,
    phase_composer,
    is_in_contact,
    node_is_in_contact,
)

import casadi as cs
import pinocchio as pin
from pinocchio import casadi as cpin

## Begin Parameter Setup ##

# Trajectory parameters
total_duration = 1.0
num_states = 100  # number of states
num_intermediate_steps = 1  # number of intermediate steps in multiple shooting

contact_ee_names = [
    "lf",
    "rf",
]  # End-effectors in contact (names arbitrary)

# Phases
# MAKE SURE THE TOTAL DURATIONS FOR EACH EE MATCH UP
node_phase_sequence = {
    "lf": list((35, 30, 35)),
    "rf": list((100,)),
}
# contact_durations = {
#     "lf": list((0.5,)),
#     "rf": list((0.5,)),
#     # "lf": list((0.5, 1.0)),
#     # "rf": list((1.2, 0.5)),
# }
# swing_durations = {
#     "lf": list(()),
#     "rf": list(()),
#     # "lf": list((0.5,)),
#     # "rf": list((0.3,)),
# }
is_init_contact = {
    "lf": True,
    "rf": True,
}

# Optimization model parameters
num_contact_points_per_ee = {
    "lf": 4,
    "rf": 4,
}

# phase_sequence = {}
# for ee_name in contact_ee_names:
#     phase_sequence[ee_name] = phase_composer(
#         contact_durations[ee_name], swing_durations[ee_name], is_init_contact[ee_name]
#     )


dim_f_ext = 3  # Dimensions of external forces

## End Parameter Setup ##

## Load Model ##

path_to_curr_folder = os.path.dirname(os.path.realpath(__file__))
urdffile = os.path.join(path_to_curr_folder, "g1_23dof.urdf")

model = pin.buildModelFromUrdf(urdffile)
data = model.createData()

cmodel = cpin.Model(model)
cdata = cmodel.createData()

## End Load Model ##

## Environment Parameters ##

ground_mu = 0.9

ground_n = [0.0, 0.0, 1.0]
ground_b = [0.0, 1.0, 0.0]
ground_t = [1.0, 0.0, 0.0]

ground_z = 0.0

grav = 9.81

## End Environment Parameters ##

nq = model.nq
nv = model.nv
nx = nq + nv

## Model Stuff ##

contact_frame_names = {
    "lf": [
        "left_foot_upper_right",
        "left_foot_lower_right",
        "left_foot_upper_left",
        "left_foot_lower_left",
    ],
    "rf": [
        "right_foot_upper_right",
        "right_foot_lower_right",
        "right_foot_upper_left",
        "right_foot_lower_left",
    ],
}

q0 = [
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

q1 = [0.0, 0.0]
q_mid = [0.0, 0.0, 1.2]

# Calculate base offset from ground
pin.framesForwardKinematics(model, data, np.array(q0))
frame_id = model.getFrameId("right_foot_point_contact")
base_pos_offset = data.oMf[frame_id].translation
base_rot_offset = R_to_quat(data.oMf[frame_id].rotation)
q0[2] = -base_pos_offset[2]
q0[3:7] = (base_rot_offset).tolist()

v0 = [0.0 for _ in range(nv)]
a0 = [0.0 for _ in range(nv)]
fc0 = [0.0, 0.0, 34.0 * grav / 8]

q_init = q0
v_init = v0
q_last = q1 + q0[2:]
v_last = v0

## End Model Stuff ##

## Durations ##
# total_duration = sum(contact_durations["lf"]) + sum(swing_durations["lf"])

dt0 = total_duration / num_states / num_intermediate_steps
dt_min = 0.001
dt_max = 0.05

# print("Total duration: ", total_duration)
# print("Num dts: ", num_states / num_intermediate_steps)
# print("dt: ", dt0)

## End Durations ##

## Add variables

q_min = model.lowerPositionLimit.tolist()
q_max = model.upperPositionLimit.tolist()
v_min = (-model.velocityLimit).tolist()
v_max = model.velocityLimit.tolist()
a_min = [-cs.inf for _ in range(nv)]
a_max = [cs.inf for _ in range(nv)]

tau_min = [0.0 for _ in range(7)] + (-model.effortLimit).tolist()[7:]
tau_max = [0.0 for _ in range(7)] + model.effortLimit.tolist()[7:]

fc_min = [-cs.inf for _ in range(dim_f_ext)]
fc_max = [cs.inf for _ in range(dim_f_ext)]

# print("Limits:")
# print("q_min:", q_min)
# print("q_max:", q_max)
# print("v_min:", v_min)
# print("v_max:", v_max)
# print("a_min:", a_min)
# print("a_max:", a_max)

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
for i in range(num_states):

    # Add state to cost
    # J += 10.0 * cs.sumsqr(x[7:nq] - q_init[7:])
    # J += cs.dot(x, x)

    # Integrate for num_intermediate_steps
    for j in range(num_intermediate_steps):
        # Add control variable
        a = cs.SX.sym("a" + str(i) + "_" + str(j), nv, 1)
        w += [a]
        lbw += a_min
        ubw += a_max
        w0 += a0

        J += 10.0 * cs.dot(a, a)
        # Add centroidal dynamics in cost
        # cpin.computeCentroidalMomentumTimeVariation(cmodel, cdata, x[:nq], x[nq:], a)
        # J += 1.0 * cs.sumsqr(cdata.hg.angular)

        cpin.computeJointJacobians(cmodel, cdata, x[:nq])
        cpin.computeJointJacobiansTimeVariation(cmodel, cdata, x[:nq], x[nq:])
        JtF_sum = 0.0
        for ee_name in contact_ee_names:
            # Determine if ee is in contact
            # in_contact = is_in_contact(t, phase_sequence[ee_name])
            in_contact = node_is_in_contact(
                i, node_phase_sequence[ee_name], is_init_contact[ee_name]
            )
            # print("Node: ", i, " in contact:", in_contact)

            for frame_name in contact_frame_names[ee_name]:
                frame_id = model.getFrameId(frame_name)
                ref_frame = pin.LOCAL_WORLD_ALIGNED
                jac = cpin.getFrameJacobian(cmodel, cdata, frame_id, ref_frame)
                jacdot = cpin.getFrameJacobianTimeVariation(
                    cmodel, cdata, frame_id, ref_frame
                )

                # Create force variable
                fc = cs.SX.sym(
                    "fc_" + frame_name + "_" + str(i) + "_" + str(j), dim_f_ext, 1
                )
                w += [fc]
                lbw += fc_min
                ubw += fc_max
                w0 += fc0

                J += cs.dot(fc, fc)

                ## End-effector distance from ground ##
                cpin.framesForwardKinematics(cmodel, cdata, x[:nq])
                ee_pos = cdata.oMf[frame_id].translation

                # if not in_contact:
                # J += cs.sumsqr(ee_pos[2] - (ground_z + 0.1))

                g += [ee_pos[2] - ground_z]
                lbg += [0.0]
                ubg += [0.0 if in_contact else cs.inf]
                ## End end-effector distance from ground ##

                ## Zero velocity and acceleration in contact ##
                if in_contact:
                    ## Zero velocity ##
                    g += [jac[0:3, :] @ x[nq:]]
                    lbg += [0.0 for _ in range(3)]
                    ubg += [0.0 for _ in range(3)]

                    ## Zero acceleration ##
                    # g += [jac[0:3, :] @ a + jacdot[0:3, :] @ x[nq:]]
                    # lbg += [0.0 for _ in range(3)]
                    # ubg += [0.0 for _ in range(3)]
                ## End zero velocity and acceleration in contact ##

                if in_contact:
                    ## Friction cone constraint ##
                    # Normal component
                    g += [cs.dot(fc, ground_n)]
                    lbg += [0.0]
                    ubg += [cs.inf]
                    # Tangentials
                    lim = (ground_mu / cs.sqrt(2.0)) * cs.dot(fc, ground_n)
                    g += [
                        cs.dot(fc, ground_b) + lim,
                        -cs.dot(fc, ground_b) + lim,
                        cs.dot(fc, ground_t) + lim,
                        -cs.dot(fc, ground_t) + lim,
                    ]
                    lbg += [0.0, 0.0, 0.0, 0.0]
                    ubg += [cs.inf, cs.inf, cs.inf, cs.inf]
                    ## End friction cone constraint ##
                else:
                    ## Zero force when not in contact ##
                    g += [fc]
                    lbg += [0.0 for _ in range(dim_f_ext)]
                    ubg += [0.0 for _ in range(dim_f_ext)]
                    ## End zero force when not in contact ##

                ## Calculate JtF sum
                # Get contact Jacobian
                JtF = cs.mtimes(jac[0:3, :].T, fc)
                JtF_sum += JtF

        g += [cpin.rnea(cmodel, cdata, x[:nq], x[nq:], a) - JtF_sum]
        lbg += tau_min
        ubg += tau_max

        # Contact constraints

        # Semi-implicit Euler integration
        dt = cs.SX.sym("dt_" + str(i), 1)
        w += [dt]
        lbw += [dt_min]
        ubw += [dt_max]
        w0 += [dt0]

        v = x[nq:] + a * dt
        p = x[:nq] + v_to_qdot(nq, x[:nq], v) * dt
        x = cs.vertcat(p, v)

        # t += dt0

    # Add intermediate state as variable
    xk = cs.SX.sym("x" + str(i + 1), nx, 1)
    w += [xk]
    if i == num_states - 1:
        lbw += q_last + v_last
        ubw += q_last + v_last
    # elif i == num_states / 2:
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

num_contact_ees = 2
num_contact_points_per_ee = 4
nc = num_contact_ees * num_contact_points_per_ee * dim_f_ext
nw = nx + num_intermediate_steps * (nv + nc) + 1

c_idx = 0
x = np.zeros((nx, num_states + 1))
for i in range(num_states + 1):
    start = i * nw
    end = start + nx
    x[:, c_idx] = w_opt[start:end]
    c_idx += 1
np.savetxt("x.csv", x, fmt="%.2f", delimiter="\t,")

# Acceleration
a = np.zeros((nv, num_states * num_intermediate_steps))

c_idx = 0
for i in range(num_states):
    for j in range(num_intermediate_steps):
        start = i * nw + nx
        end = start + j * (nv + nc) + nv
        a[:, c_idx] = w_opt[start:end]
        c_idx += 1

np.savetxt("a.csv", a, delimiter=",")

# Contact forces
fc = np.zeros((nc, num_states * num_intermediate_steps))

c_idx = 0
for i in range(num_states):
    for j in range(num_intermediate_steps):
        start = i * nw + nx + nv
        end = start + j * (nv + nc) + nc
        fc[:, c_idx] = w_opt[start:end]
        c_idx += 1

np.savetxt("fc.csv", fc, delimiter=",")

# Save results to csv
# dts = np.array([dt for _ in range(num_states)])
# np.savetxt("dt.csv", dts, delimiter=",")

# num_contact_ees = 2
# num_contact_points_per_ee = 4
# nc = num_contact_ees * num_contact_points_per_ee * dim_f_ext
# nw = nx + num_intermediate_steps * (nv + nc)
#
# c_idx = 0
# x = np.zeros((nx, num_states + 1))
# for i in range(num_states + 1):
#     start = i * nw
#     end = start + nx
#     print("Iter: ", c_idx, " (start: ", start, " ,end: ", end, ")")
#     x[:, c_idx] = w_opt[start:end]
#     c_idx += 1
# np.savetxt("x.csv", x, fmt="%.2f", delimiter="\t,")
#
# # Acceleration
# a = np.zeros((nv, num_states * num_intermediate_steps))
#
# c_idx = 0
# for i in range(num_states):
#     for j in range(num_intermediate_steps):
#         start = i * nw + nx
#         end = start + j * (nv + nc) + nv
#         a[:, c_idx] = w_opt[start:end]
#         c_idx += 1
#
# np.savetxt("a.csv", a, delimiter=",")
#
# # Contact forces
# fc = np.zeros((nc, num_states * num_intermediate_steps))
#
# c_idx = 0
# for i in range(num_states):
#     for j in range(num_intermediate_steps):
#         start = i * nw + nx + nv
#         end = start + j * (nv + nc) + nc
#         fc[:, c_idx] = w_opt[start:end]
#         c_idx += 1
#
# np.savetxt("fc.csv", fc, delimiter=",")
