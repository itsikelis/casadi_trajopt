import os
import numpy as np
import casadi as cs
from functions import (
    get_force_spline_phases,
    get_force_spline_durations,
    get_ee_spline_phases,
    get_ee_spline_durations,
    inertia_tensor,
)
from srbd_model import SRBDModel

## Begin Parameter Setup ##

# Trajectory parameters
num_knot_points = 30  # number of knot points
num_col_points = 34  # number of collocation points

contact_ee_names = [
    "lf",
    "rf",
]  # End-effectors in contact (names arbitrary)

# Phased parameters
contact_durations = {}
contact_durations["lf"] = list((1.2,))
contact_durations["rf"] = list((0.5, 0.5))

swing_durations = {}
swing_durations["lf"] = list(())
swing_durations["rf"] = list((0.2,))

# Assert the total durations for each ee match up
sums = [
    round(sum(contact_durations[ee]) + sum(swing_durations[ee]), 3)
    for ee in contact_ee_names
]
assert len(set(sums)) <= 1

is_init_contact = {}
is_init_contact["lf"] = True
is_init_contact["rf"] = True

num_ee_splines_per_contact = 2
num_force_splines_per_contact = 4

# Optimization model parameters

# Contact points per end-effector
num_contact_points_per_ee = {ee_name: 1 for ee_name in contact_ee_names}

dim_f_ext = 3  # Dimensions of external forces

## End Parameter Setup ##

## Model stuff ##

# Anymal
# model_mass = 30.4213964625
# model_inertia = inertia_tensor(
#     0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922
# )

# Biped
model_mass = 20.0
model_inertia = inertia_tensor(1.209, 5.583, 6.056, 0.005, -0.190, -0.012)
num_contact_ees = 2

model = SRBDModel(model_mass, model_inertia, num_contact_ees)

nq = model.nq()
nv = model.nv()

## End Model Stuff ##

## Fround normal vectors ##
ground_mu = 0.9

ground_n = [0.0, 0.0, 1.0]
ground_b = [0.0, 1.0, 0.0]
ground_t = [1.0, 0.0, 0.0]

ground_z = 0.0
## End ground normal vectors ##

# Create phased arrays
force_spline_is_contact = {}
force_phased_spline_durations = {}

ee_pos_spline_is_contact = {}
ee_pos_phased_spline_durations = {}

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

    force_spline_is_contact[ee_name] = get_force_spline_phases(
        num_contact_phases=num_contact_phases[ee_name],
        num_swing_phases=num_swing_phases[ee_name],
        num_splines_per_contact=num_force_splines_per_contact,
        is_init_contact=is_init_contact[ee_name],
    )
    force_phased_spline_durations[ee_name] = get_force_spline_durations(
        contact_durations=contact_durations[ee_name],
        swing_durations=swing_durations[ee_name],
        num_splines_per_contact=num_force_splines_per_contact,
        is_init_contact=is_init_contact[ee_name],
    )

    ee_pos_spline_is_contact[ee_name] = get_ee_spline_phases(
        num_contact_phases=num_contact_phases[ee_name],
        num_swing_phases=num_swing_phases[ee_name],
        num_splines_per_contact=num_ee_splines_per_contact,
        is_init_contact=is_init_contact[ee_name],
    )

    ee_pos_phased_spline_durations[ee_name] = get_ee_spline_durations(
        contact_durations=contact_durations[ee_name],
        swing_durations=swing_durations[ee_name],
        num_splines_per_contact=num_ee_splines_per_contact,
        is_init_contact=is_init_contact[ee_name],
    )

# print(force_phased_spline_durations)
# print(force_spline_is_contact)
# # print(["contact" if is_contact else "swing" for is_contact in force_spline_is_contact])
# print()
# print(ee_pos_phased_spline_durations)
# print(ee_pos_spline_is_contact)
# # exit()

grav = 9.81

total_duration = sum(force_phased_spline_durations["lf"])
state_spline_T = total_duration / (num_knot_points - 1)
state_spline_durations = [
    state_spline_T for _ in range(num_knot_points - 1)
]  # The durations of the state splines (equal)
collocation_times = [
    k * (total_duration / num_col_points) for k in range(num_col_points)
]

# print("\nTotal duration: " + str(total_duration))
# print(state_spline_durations)
# print(collocation_times)
# exit()

## Add variables
q0 = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]
v0 = [0.0 for _ in range(nv)]

q_init = q0
v_init = v0
q_last = q0
v_last = v0

q_min = [-cs.inf, -cs.inf, -cs.inf, -1.01, -1.01, -1.01, -1.01]
q_max = [cs.inf, cs.inf, cs.inf, 1.01, 1.01, 1.01, 1.01]
v_min = [-cs.inf for _ in range(nv)]
v_max = [cs.inf for _ in range(nv)]

f_min = [-cs.inf for _ in range(3)]
f_max = [cs.inf for _ in range(3)]

q_init = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
v_init = [0.0 for _ in range(6)]
q_target = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
v_target = [0.0 for _ in range(6)]

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
        lbw += q_init + v_init
        ubw += q_init + v_init
    elif k == num_knot_points - 1:
        lbw += q_target + v_target
        ubw += q_target + v_target
    else:
        lbw += q_min + v_min
        ubw += q_max + v_max

# f_min = [0.0, 0.0, -34.0 * grav / (num_contact_points_per_ee)]
# f_max = [0.0, 0.0, 34.0 * grav / (num_contact_points_per_ee)]

w_force = {}
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
            iters = num_force_splines_per_contact
        else:
            iters = num_force_splines_per_contact - 1

        # Edge case if there is just a single contact phase
        if num_swing_phases[ee_name] == 0:
            iters = num_force_splines_per_contact + 1

        for i in range(iters):
            name = "fc_" + ee_name + "_" + str(k) + "_" + str(i)
            f = cs.MX.sym(name, num_contact_points_per_ee[ee_name] * dim_f_ext)
            ee_vars += [f]
            w0 += f0
            lbw += f_min
            ubw += f_max
    w_force[ee_name] = ee_vars
    print(ee_vars)

pos_min = [-cs.inf for _ in range(3)]
pos_max = [cs.inf for _ in range(3)]
pos0 = [0.0 for _ in range(dim_f_ext)]

w_ee_pos = {}
for ee_name in contact_ee_names:
    ee_vars = []

    num_phases = num_contact_phases[ee_name] + num_swing_phases[ee_name]
    is_contact = is_init_contact[ee_name]
    for i in range(num_phases):
        if is_contact == True:
            name = "contact_" + ee_name + "_" + str(i)
            ee_pos = cs.MX.sym(name, 3)
            ee_vars += [ee_pos]
            w0 += pos0
            lbw += pos_min
            ubw += pos_max
        else:
            iters = 0

            if (i == 0) or (i == num_phases - 1):
                iters = num_ee_splines_per_contact
            else:
                iters = num_ee_splines_per_contact - 1

            # Edge case if there is just a single swing phase
            if num_swing_phases[ee_name] == 0:
                iters = num_ee_splines_per_contact + 1

            for k in range(iters):
                name = "swing_" + ee_name + "_" + str(i) + "_" + str(k)
                pos = cs.MX.sym(name, 3)
                ee_vars += [pos]
                w0 += pos0
                lbw += pos_min
                ubw += pos_max

        is_contact = not is_contact

    w_ee_pos[ee_name] = ee_vars
    print(ee_vars)

## Add constraints ##

# print("w: " + str(len(w)))
# for ee in contact_ee_names:
#     print("w_phased " + ee + ": " + str(w_phased[ee]))
exit()

## Dynamics constraints ##

for t in collocation_times:
    idx = get_state_spline_idx(state_spline_durations, t)
    x0 = w[idx]
    x1 = w[idx + 1]

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
        t_norm = state_spline_T / 2.0
        for t in collocation_times:
            idx = get_state_spline_idx(state_spline_durations, t)
            x0 = w[idx]
            x1 = w[idx + 1]
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

for ee_name in contact_ee_names:
    for frame_name in contact_frame_names[ee_name]:
        fv = kindyn.frameVelocity(
            frame_name, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        )
        # fa = kindyn.frameAcceleration(
        #     frame_name, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
        # )
        t_norm = state_spline_T / 2.0
        for t in collocation_times:
            is_curr_contact = is_in_contact(
                phased_spline_durations[ee_name], spline_is_contact[ee_name], t
            )
            if is_curr_contact:
                idx = get_state_spline_idx(state_spline_durations, t)
                x0 = w[idx]
                x1 = w[idx + 1]
                q = collocation_q(kindyn, state_spline_T, x0, x1, t_norm)
                qdot = collocation_qdot(kindyn, state_spline_T, x0, x1, t_norm)
                qddot = collocation_qddot(kindyn, state_spline_T, x0, x1, t_norm)
                v = cs.MX.zeros(nv)
                # a = cs.MX.zeros(nv)
                qdot_to_v(v, q, qdot)
                # qddot_to_a(v, q, qdot, qddot)

                # jac = kindyn.jacobian(
                #     frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED
                # )

                g += [fv(q=q, qdot=v)["ee_vel_linear"]]
                lbg += [0.0 for _ in range(3)]
                ubg += [0.0 for _ in range(3)]

                # g += [fa(q=q, qdot=v, qddot=a)["ee_acc_linear"]]
                # lbg += [0.0 for _ in range(3)]
                # ubg += [0.0 for _ in range(3)]

            idx += 1

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
for ee_name in contact_ee_names:
    for var in w_phased[ee_name]:
        J += cs.dot(var, var)

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
# print(w_opt[0 : num_knot_points * (nq + nv)])

# Save results to csv
dts = np.array(state_spline_durations)
np.savetxt("dt.csv", dts, delimiter=",")

knot_points = np.array(w_opt[0 : num_knot_points * (nq + nv)]).reshape(
    (
        num_knot_points,
        nq + nv,
    )
)
np.savetxt("x.csv", knot_points.T, delimiter=",")

fc = np.array(w_opt[num_knot_points * (nq + nv) :])
contact_forces = {}
var_end = 0
for ee_name in contact_ee_names:
    var_start = var_end
    var_end = var_start + len(w_phased[ee_name]) * (
        dim_f_ext * num_contact_points_per_ee[ee_name]
    )
    contact_forces[ee_name] = np.array(fc[var_start:var_end])
    # .reshape(
    #    (dim_f_ext * num_contact_points_per_ee[ee_name], -1)
    # )
    print(contact_forces[ee_name])
    # np.savetxt("fc_" + ee_name + ".csv", contact_forces[ee_name], delimiter=",")
