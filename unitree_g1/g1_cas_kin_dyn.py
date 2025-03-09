import os
import numpy as np
import casadi as cs
from functions import *
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn

# from pinocchio import casadi as cpin

## Begin Parameter Setup ##

# Trajectory parameters
num_knot_points = 10  # number of knot points
num_col_points = 12  # number of collocation points

contact_ee_names = [
    "lf",
    "rf",
]  # End-effectors in contact (names arbitrary)

# Phased parameters
# MAKE SURE THE TOTAL DURATIONS FOR EACH EE MATCH UP
contact_durations = {}
contact_durations["lf"] = list((0.2, 0.1))
contact_durations["rf"] = list((0.4,))

swing_durations = {}
swing_durations["lf"] = list((0.1,))
swing_durations["rf"] = list(())

is_init_contact = {}
is_init_contact["lf"] = True
is_init_contact["rf"] = True

num_splines_per_contact = 5

# Optimization model parameters
num_contact_points_per_ee = {}  # Contact points per end-effector
num_contact_points_per_ee["lf"] = 4
num_contact_points_per_ee["rf"] = 4

dim_f_ext = 3  # Dimensions of external forces

## End Parameter Setup ##

## Model stuff ##

# Find and load the urdf
path_to_curr_folder = os.path.dirname(os.path.realpath(__file__))
urdffile = os.path.join(path_to_curr_folder, "g1_23dof.urdf")
urdf = open(urdffile, "r").read()

# model = cpin.buildModelFromUrdf(urdffile)
# data = model.createData()

kindyn = cas_kin_dyn.CasadiKinDyn(urdf)
nq = kindyn.nq()
nv = kindyn.nv()

contact_frame_names = {}
contact_frame_names["lf"] = [
    "left_foot_upper_right",
    "left_foot_lower_right",
    "left_foot_upper_left",
    "left_foot_lower_left",
]

contact_frame_names["rf"] = [
    "right_foot_upper_right",
    "right_foot_lower_right",
    "right_foot_upper_left",
    "right_foot_lower_left",
]

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

v0 = [0.0 for _ in range(nv)]
q_init = q0
v_init = v0
q_last = q0
v_last = v0

## End Model Stuff ##

## Fround normal vectors ##
ground_mu = 0.9

ground_n = [0.0, 0.0, 1.0]
ground_b = [0.0, 1.0, 0.0]
ground_t = [1.0, 0.0, 0.0]

ground_z = 0.0
## End ground normal vectors ##

# Create phased arrays
spline_is_contact = {}
phased_spline_durations = {}
num_contact_phases = {}
num_swing_phases = {}
num_phases_total = {}
is_final_contact = {}

for ee_name in contact_ee_names:
    num_contact_phases[ee_name] = len(contact_durations[ee_name])
    num_swing_phases[ee_name] = len(swing_durations[ee_name])
    num_phases_total[ee_name] = num_contact_phases[ee_name] + num_swing_phases[ee_name]

    is_final_contact[ee_name] = (
        True
        if (is_init_contact[ee_name] and (not num_phases_total[ee_name] % 2 == 0))
        or ((not is_init_contact[ee_name]) and num_phases_total[ee_name] % 2 == 0)
        else False
    )

    spline_is_contact[ee_name] = get_spline_phases(
        num_contact_phases=num_contact_phases[ee_name],
        num_swing_phases=num_swing_phases[ee_name],
        num_splines_per_contact=num_splines_per_contact,
        is_init_contact=is_init_contact[ee_name],
    )
    phased_spline_durations[ee_name] = get_phased_spline_durations(
        contact_durations=contact_durations[ee_name],
        swing_durations=swing_durations[ee_name],
        num_splines_per_contact=num_splines_per_contact,
        is_init_contact=is_init_contact[ee_name],
    )

# print(phased_spline_durations)
# print(spline_is_contact)
# print(["contact" if is_contact else "swing" for is_contact in spline_is_contact])
# exit()

grav = 9.81

total_duration = sum(phased_spline_durations["lf"])
state_spline_T = total_duration / (num_knot_points - 1)
state_spline_durations = [
    state_spline_T for _ in range(num_knot_points - 1)
]  # The durations of the state splines (equal)
collocation_times = [
    k * (total_duration / (num_col_points - 1)) for k in range(num_col_points)
]

# print("Total duration: " + str(total_duration))
# print("State spline durations: " + str(state_spline_durations))
# print("Collocation times: " + str(collocation_times))
# exit()

# print(total_duration)
# print(collocation_times)

## Add variables

q_min = kindyn.q_min()
q_max = kindyn.q_max()
v_min = [-cs.inf for _ in range(nv)]
v_max = [cs.inf for _ in range(nv)]

# Start with an empty NLP
w = []  # variables
w0 = []  # variable initial guess
lbw = []  # variable lower bound
ubw = []  # varialble upper bound
J = 0  # cost
g = []  # constraints
lbg = []  # lower bound for contstraints
ubg = []  # upper bound for contraints

# Add position and velocity variables
for k in range(num_knot_points):
    xk = cs.MX.sym("x_" + str(k), nq + nv)
    w += [xk]
    w0 += q0 + v0

    if k == 0:
        lbw += [0.0, 0.0] + q_min[2:7] + q_init[7:] + v_init
        ubw += [0.0, 0.0] + q_max[2:7] + q_init[7:] + v_init
    elif k == num_knot_points - 1:
        lbw += [0.0, 0.0] + q_min[2:7] + q_last[7:] + v_last
        ubw += [0.0, 0.0] + q_max[2:7] + q_last[7:] + v_last
    else:
        lbw += q_min + v_min
        ubw += q_max + v_max

# f_min = [0.0, 0.0, -34.0 * grav / (num_contact_points_per_ee)]
# f_max = [0.0, 0.0, 34.0 * grav / (num_contact_points_per_ee)]

tau_min = (-kindyn.effortLimits()).tolist()
tau_max = (kindyn.effortLimits()).tolist()

w_phased = {}
for ee_name in contact_ee_names:
    f0 = []
    f_min = []
    f_max = []
    for _ in range(num_contact_points_per_ee[ee_name]):
        f_min += [-cs.inf for _ in range(dim_f_ext)]
        f_max += [cs.inf for _ in range(dim_f_ext)]
        # f0 += [0.0, 0.0, -34.0 * grav / (num_contact_points_per_ee)]
        f0 += [0.0 for _ in range(dim_f_ext)]

    ee_vars = []
    for k in range(num_contact_phases[ee_name]):
        init = k == 0
        last = k == num_contact_phases[ee_name] - 1
        iters = 0

        if (is_init_contact[ee_name] and init) or (is_final_contact[ee_name] and last):
            iters = num_splines_per_contact
        else:
            iters = num_splines_per_contact - 1

        # Edge case if there is just a single contact phase
        if num_swing_phases[ee_name] == 0:
            iters = num_splines_per_contact + 1

        for i in range(iters):
            name = "fc_" + ee_name + "_" + str(k) + "_" + str(i)
            f = cs.MX.sym(name, num_contact_points_per_ee[ee_name] * dim_f_ext)
            ee_vars += [f]
            w0 += f0
            lbw += f_min
            ubw += f_max
    w_phased[ee_name] = ee_vars

## Add constraints ##

# print("w: " + str(len(w)))
# for ee in contact_ee_names:
#     print("w_phased " + ee + ": " + str(w_phased[ee]))
# exit()

## Dynamics constraints ##

inv_dyn = kindyn.rnea()  # cs function of rnea
for t in collocation_times:
    [idx, t_norm] = get_state_spline_idx(state_spline_durations, t)
    x0 = w[idx]
    x1 = w[idx + 1]

    # print("Time: " + str(t) + " t_norm: " + str(t_norm))
    # print("x0: " + str(x0) + " x1: " + str(x1))

    # Get state at collocation points
    q = collocation_q(kindyn, state_spline_T, x0, x1, t_norm)
    qdot = collocation_qdot(kindyn, state_spline_T, x0, x1, t_norm)
    qddot = collocation_qddot(kindyn, state_spline_T, x0, x1, t_norm)

    # Transform qdot and qddot to v and a
    v = cs.MX.zeros(nv)
    qdot_to_v(v, q, qdot)
    a = cs.MX.zeros(nv)
    qddot_to_a(a, q, qdot, qddot)

    JtF_sum = 0
    for ee_name in contact_ee_names:
        [_, fc] = get_contact_forces(
            w_phased[ee_name],
            t,
            phased_spline_durations[ee_name],
            spline_is_contact[ee_name],
            num_contact_points_per_ee[ee_name],
            dim_f_ext,
        )

        for k in range(num_contact_points_per_ee[ee_name]):
            frame = contact_frame_names[ee_name][k]
            jac = kindyn.jacobian(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
            force_var_idx = k * dim_f_ext
            force = fc[force_var_idx : force_var_idx + dim_f_ext]
            # print("t: " + str(t) + ": " + str(force))
            # print(force)
            JtF = cs.mtimes(jac(q)[0:3, :].T, force)
            JtF_sum += JtF

    g += [inv_dyn(q=q, v=v, a=a)["tau"] - JtF_sum]
    lbg += tau_min
    ubg += tau_max
    idx += 1

## End dynamics constraints ##

## End-effector in contact constraint ##

for ee_name in contact_ee_names:
    for frame_name in contact_frame_names[ee_name]:
        forw_kin = kindyn.fk(frame_name)
        for t in collocation_times:
            [idx, t_norm] = get_state_spline_idx(state_spline_durations, t)
            x0 = w[idx]
            x1 = w[idx + 1]

            # print("Time: " + str(t) + " t_norm: " + str(t_norm))
            # print("x0: " + str(x0) + " x1: " + str(x1))

            q = collocation_q(kindyn, state_spline_T, x0, x1, t_norm)
            g += [forw_kin(q=q)["ee_pos"][2]]

            is_curr_contact = is_in_contact(
                phased_spline_durations[ee_name], spline_is_contact[ee_name], t
            )

            lower = [ground_z]
            upper = [ground_z if is_curr_contact else cs.inf]

            lbg += lower
            ubg += upper

            # curr_phase = "contact" if is_curr_contact else "swing"
            # print(
            #     str(round(t, 2))
            #     + ": "
            #     + " ( "
            #     + curr_phase
            #     + ") ==> "
            #     + str(lower)
            #     + " | "
            #     + str(upper)
            # )

            idx += 1

## End end-effector in contact constraint ##

## Zero velocity and acceleration in contact constraint ##

# for ee_name in contact_ee_names:
#     for frame_name in contact_frame_names[ee_name]:
#         fv = kindyn.frameVelocity(
#             frame_name, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
#         )
#         # fa = kindyn.frameAcceleration(
#         #     frame_name, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
#         # )
#         for t in collocation_times:
#             is_curr_contact = is_in_contact(
#                 phased_spline_durations[ee_name], spline_is_contact[ee_name], t
#             )
#             if is_curr_contact:
#                 [idx, t_norm] = get_state_spline_idx(state_spline_durations, t)
#                 x0 = w[idx]
#                 x1 = w[idx + 1]
#                 q = collocation_q(kindyn, state_spline_T, x0, x1, t_norm)
#                 qdot = collocation_qdot(kindyn, state_spline_T, x0, x1, t_norm)
#                 qddot = collocation_qddot(kindyn, state_spline_T, x0, x1, t_norm)
#                 v = cs.MX.zeros(nv)
#                 # a = cs.MX.zeros(nv)
#                 qdot_to_v(v, q, qdot)
#                 # qddot_to_a(v, q, qdot, qddot)
#
#                 # jac = kindyn.jacobian(
#                 #     frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
#                 # )
#
#                 g += [fv(q=q, qdot=v)["ee_vel_linear"]]
#                 lbg += [0.0 for _ in range(3)]
#                 ubg += [0.0 for _ in range(3)]
#
#                 # g += [fa(q=q, qdot=v, qddot=a)["ee_acc_linear"]]
#                 # lbg += [0.0 for _ in range(3)]
#                 # ubg += [0.0 for _ in range(3)]
#
#             idx += 1

## End zero velocity and acceleration in contact constraint ##

## Friction cone constraint ##

for ee_name in contact_ee_names:
    for t in collocation_times:
        [is_contact, fc] = get_contact_forces(
            w_phased[ee_name],
            t,
            phased_spline_durations[ee_name],
            spline_is_contact[ee_name],
            num_contact_points_per_ee[ee_name],
            dim_f_ext,
        )

        if is_contact:
            for k in range(num_contact_points_per_ee[ee_name]):
                force_var_idx = k * dim_f_ext
                force = fc[force_var_idx : force_var_idx + dim_f_ext]
                # Normal component
                g += [cs.dot(force, ground_n)]
                lbg += [0.0]
                ubg += [cs.inf]
                # Tangentials
                lim = (ground_mu / cs.sqrt(2.0)) * cs.dot(force, ground_n)
                g += [
                    cs.dot(force, ground_b) + lim,
                    -cs.dot(force, ground_b) + lim,
                    cs.dot(force, ground_t) + lim,
                    -cs.dot(force, ground_t) + lim,
                ]
                lbg += [0.0, 0.0, 0.0, 0.0]
                ubg += [cs.inf, cs.inf, cs.inf, cs.inf]

## End friction cone constraint ##

## Equal acceleration constraint ##

for k in range(num_knot_points - 2):
    # In each intermittent knot point, add an acceleration constraint
    x0 = w[k]
    x1 = w[k + 1]
    x2 = w[k + 2]
    T0 = state_spline_durations[k]
    T1 = state_spline_durations[k + 1]

    qddot0 = collocation_qddot(kindyn, T0, x0, x1, T0)
    qddot1 = collocation_qddot(kindyn, T0, x1, x2, 0.0)

    g += [qddot0 - qddot1]
    lbg += [0.0 for _ in range(nq)]
    ubg += [0.0 for _ in range(nq)]

## End equal acceleration constraint ##

# Add force to cost
# for ee_name in contact_ee_names:
#     for var in w_phased[ee_name]:
#         J += cs.dot(var, var)

# Merge the variables vector
for ee in contact_ee_names:
    w += w_phased[ee]

# for var in w:
#     J += cs.dot(var, var)

# Assert vector sizes are correct
assert cs.vertcat(*w).shape[0] == len(w0) == len(lbw) == len(ubw)
assert cs.vertcat(*g).shape[0] == len(lbg) == len(ubg)

# Create an NLP solver
opts = {}
# opts["ipopt.linear_solver"] = "ma57"
prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g)}
solver = cs.nlpsol("solver", "ipopt", prob, opts)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol["x"].full().flatten()
print(w_opt[0 : num_knot_points * (nq + nv)])

# # Save results to csv
# dts = np.array(state_spline_durations)
# np.savetxt("dt.csv", dts, delimiter=",")
#
# knot_points = np.array(w_opt[0 : num_knot_points * (nq + nv)]).reshape(
#     (
#         num_knot_points,
#         nq + nv,
#     )
# )
# np.savetxt("x.csv", knot_points.T, delimiter=",")
#
# fc = np.array(w_opt[num_knot_points * (nq + nv) :])
# contact_forces = {}
# var_end = 0
# for ee_name in contact_ee_names:
#     var_start = var_end
#     var_end = var_start + len(w_phased[ee_name]) * (
#         dim_f_ext * num_contact_points_per_ee[ee_name]
#     )
#     contact_forces[ee_name] = np.array(fc[var_start:var_end])
#     # .reshape(
#     #    (dim_f_ext * num_contact_points_per_ee[ee_name], -1)
#     # )
#     # np.savetxt("fc_" + ee_name + ".csv", contact_forces[ee_name], delimiter=",")
