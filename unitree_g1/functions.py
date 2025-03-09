import numpy as np
import casadi as cs
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn


# Return t_norm and spline index
def normalise_time(durations, spline_is_contact, t):
    spline_idx = 0
    prev_swing_phases = 0
    sum = 0.0
    prev_sum = 0.0
    t_norm = 0.0

    for i in range(len(durations)):
        # Record previous swing phases
        if not spline_is_contact[i]:
            prev_swing_phases += 1

        # Sum all durations up to now to determine if t is before or after
        sum += durations[i]
        if t <= sum - 1e-3:
            # Normalise by subtracting all previous durations
            t_norm = t - prev_sum
            # Record spline index
            spline_idx = i
            break

        prev_sum = sum

    var_start_idx = spline_idx - 2 * prev_swing_phases
    return (spline_idx, var_start_idx, t_norm)


# Return get state spline idx
def get_state_spline_idx(durations, t):
    total_duration = np.sum(durations)
    if t < total_duration:
        sum = 0.0
        prev_sum = 0.0
        for i in range(len(durations)):
            sum += durations[i]
            if t <= sum - 1e-3:
                t_norm = t - prev_sum
                return [i, t_norm]
            prev_sum = sum
    # Keep this outside an else statement, because for loop may fail to return due to floating point error.
    return [len(durations) - 1, total_duration]


# Quaternion dot to angular velocity transform
def G(q):
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    G = cs.MX.zeros(3, 4)
    G[0, 0] = -q1
    G[0, 1] = q0
    G[0, 2] = q3
    G[0, 3] = -q2

    G[1, 0] = -q2
    G[1, 1] = -q3
    G[1, 2] = q0
    G[1, 3] = -q2

    G[2, 0] = -q3
    G[2, 1] = q2
    G[2, 2] = -q1
    G[2, 3] = q0

    return G


# Angular velocity to quaternion dot transform
def G_T(q):
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    G_T = cs.MX.zeros(4, 3)
    G_T[0, 0] = q1
    G_T[0, 1] = -q2
    G_T[0, 2] = -q3

    G_T[1, 0] = q0
    G_T[1, 1] = -q3
    G_T[1, 2] = q2

    G_T[2, 0] = q3
    G_T[2, 1] = q0
    G_T[2, 2] = -q1

    G_T[3, 0] = -q2
    G_T[3, 1] = q1
    G_T[3, 2] = q0

    return G_T


def v_to_qdot(nq, q, v):
    qdot[0:3] = v[0:3]
    qdot[3:7] = 0.5 * G_T(q[3:7]) @ v[3:6]
    qdot[7:] = v[6:]


def qdot_to_v(v, q, qdot):
    v[0:3] = qdot[0:3]
    v[3:6] = 2 * G(q[3:7]) @ qdot[3:7]
    v[6:] = qdot[7:]


def qddot_to_a(a, q, qdot, qddot):
    a[0:3] = qddot[0:3]
    a[3:6] = 2 * (G(qdot[3:7]) @ qdot[3:7] + G(q[3:7]) @ qddot[3:7])
    a[6:] = qddot[7:]


# Return a python list with the phase of each spline in the trajectory
def get_spline_phases(**args):
    col_point_phases = []

    num_phases = args["num_contact_phases"] + args["num_swing_phases"]
    is_contact = args["is_init_contact"]
    for i in range(num_phases):
        if is_contact == True:
            num_splines = args["num_splines_per_contact"]
            col_point_phases += [True for _ in range(num_splines)]
        else:
            col_point_phases += [False]
        is_contact = not is_contact

    return col_point_phases


# Return a list of the surations of each spline in the trajectory
def get_phased_spline_durations(**args):
    durations = []

    contact_phases = len(args["contact_durations"])
    swing_phases = len(args["swing_durations"])
    num_phases = contact_phases + swing_phases

    is_contact = args["is_init_contact"]
    contact_idx = 0
    swing_idx = 0
    for i in range(num_phases):
        if is_contact == True:
            num_splines = args["num_splines_per_contact"]
            durations += [
                args["contact_durations"][contact_idx] / num_splines
                for _ in range(num_splines)
            ]
            contact_idx += 1
        else:
            durations += [args["swing_durations"][swing_idx]]
            swing_idx += 1
        is_contact = not is_contact
    return durations


# Collocation point position from cubic hermite spline interpolation
def first_order_interpolation(T, f0, f1, t):
    a = (1 / T**3) @ (f1 - f0)
    b = f0
    # print("time: " + str(t) + " -> " + str(cs.MX((a * t) + b)))
    return cs.MX((a * t) + b)


def is_in_contact(phased_spline_durations, spline_is_contact, t):
    [spl_idx, _, _] = normalise_time(phased_spline_durations, spline_is_contact, t)
    is_curr_contact = spline_is_contact[spl_idx]
    return is_curr_contact


def get_contact_forces(
    w_phased,
    t,
    phased_spline_durations,
    spline_is_contact,
    num_contact_points_per_ee,
    dim_f_ext,
):
    fc = cs.MX.zeros(num_contact_points_per_ee * dim_f_ext)

    [spl_idx, var_start_idx, t_n] = normalise_time(
        phased_spline_durations, spline_is_contact, t
    )
    is_curr_contact = spline_is_contact[spl_idx]

    if is_curr_contact:
        is_not_last = spl_idx < len(spline_is_contact) - 1
        is_next_contact = spline_is_contact[spl_idx + 1] if is_not_last else True
        is_prev_contact = spline_is_contact[spl_idx - 1]

        # fc0 = w_phased[var_start_idx]
        fc0 = cs.MX.zeros(num_contact_points_per_ee * dim_f_ext)
        fc1 = cs.MX.zeros(num_contact_points_per_ee * dim_f_ext)
        if is_prev_contact:
            fc0 = w_phased[var_start_idx]
        if is_next_contact:
            fc1 = w_phased[var_start_idx + 1]

        fc = first_order_interpolation(phased_spline_durations[spl_idx], fc0, fc1, t_n)

    # curr_phase = "contact" if is_curr_contact else "swing"
    # next_phase = "contact" if is_next_contact else "swing"
    # print(
    #     str(round(t, 2))
    #     + ": "
    #     + str(spl_idx)
    #     + " (curr: "
    #     + curr_phase
    #     + ", next: "
    #     + next_phase
    #     + ") ==> "
    #     + str(fc0)
    #     + " - "
    #     + str(fc1)
    # )

    return [is_curr_contact, fc]


# Collocation point position from cubic hermite spline interpolation
def collocation_q(kindyn, T, x0, x1, t):
    nq = kindyn.nq()
    nv = kindyn.nv()

    q0 = x0[0:nq]
    v0 = x0[nq:]
    q1 = x1[0:nq]
    v1 = x1[nq:]

    # Transform v to q_dot
    qdot0 = cs.MX.zeros(nq)
    v_to_qdot(qdot0, q0, v0)
    qdot1 = cs.MX.zeros(nq)
    v_to_qdot(qdot1, q1, v1)

    a = (2 / T**3) * q0 + (1 / T**2) * qdot0 - (2 / T**3) * q1 + (1 / T**2) * qdot1
    b = -(3 / T**2) * q0 - (2 / T) * qdot0 + (3 / T**2) * q1 - (1 / T) * qdot1
    c = qdot0
    d = q0

    return cs.MX(a * t**3 + b * t**2 + c * t + d)


# Collocation point velocity from cubic hermite spline interpolation
def collocation_qdot(kindyn, T, x0, x1, t):
    nq = kindyn.nq()
    nv = kindyn.nv()

    q0 = x0[0:nq]
    v0 = x0[nq:]
    q1 = x1[0:nq]
    v1 = x1[nq:]

    # Transform v to q_dot
    qdot0 = cs.MX.zeros(nq)
    v_to_qdot(qdot0, q0, v0)
    qdot1 = cs.MX.zeros(nq)
    v_to_qdot(qdot1, q1, v1)

    a = (2 / T**3) * q0 + (1 / T**2) * qdot0 - (2 / T**3) * q1 + (1 / T**2) * qdot1
    b = -(3 / T**2) * q0 - (2 / T) * qdot0 + (3 / T**2) * q1 - (1 / T) * qdot1
    c = qdot0
    d = q0

    return cs.MX(3 * a * t**2 + 2 * b * t + c)


# Collocation point acceleration from cubic hermite spline interpolation
def collocation_qddot(kindyn, T, x0, x1, t):
    nq = kindyn.nq()

    q0 = x0[0:nq]
    v0 = x0[nq:]
    q1 = x1[0:nq]
    v1 = x1[nq:]

    # Transform v to q_dot
    qdot0 = cs.MX.zeros(nq)
    v_to_qdot(qdot0, q0, v0)
    qdot1 = cs.MX.zeros(nq)
    v_to_qdot(qdot1, q1, v1)

    a = (2 / T**3) * q0 + (1 / T**2) * qdot0 - (2 / T**3) * q1 + (1 / T**2) * qdot1
    b = -(3 / T**2) * q0 - (2 / T) * qdot0 + (3 / T**2) * q1 - (1 / T) * qdot1
    c = qdot0
    d = q0

    return cs.MX(6 * a * t + 2 * b)


# # Midpoint collocation point position from cubic hermite spline interpolation
# def MidpointCollocationPos(dt, q0, v0, q1, v1):
#     # Transform v to q_dot
#     ang_vel0 = v0[3:6]
#     ang_vel1 = v1[3:6]
#
#     quat0 = q0[3:7]
#     quat1 = q1[3:7]
#
#     qdot0 = q0
#     qdot0[0:3] = v0[0:3]
#     qdot0[3:7] = vel_to_q_dot(quat0, ang_vel0)
#     qdot0[7:] = v0[6:]
#
#     qdot1 = q1
#     qdot1[0:3] = v1[0:3]
#     qdot1[3:7] = vel_to_q_dot(quat1, ang_vel1)
#     qdot1[7:] = v1[6:]
#
#     return cs.Function(
#         "collocation_pos",
#         [q0, v0, q1, v1],
#         [
#             (dt**3 / 6.0)
#             * (
#                 (2.0 / dt**3) * q0
#                 + (1.0 / dt**2) * qdot0
#                 - (2.0 / dt**3) * q1
#                 + (1 / dt**2) * qdot1
#             )
#             + (dt**2 / 4)
#             * (
#                 -(3.0 / dt**2) * q0
#                 - (2.0 / dt) * qdot0
#                 + (3.0 / dt**2) * q1
#                 - (1.0 / dt) * qdot1
#             )
#             + (dt / 2) * qdot0
#             + q0
#         ],
#     )
#
#
# # Midpoint Collocation point velocity from cubic hermite spline interpolation
# def MidpointCollocationVel(dt, q0, v0, q1, v1):
#     # Transform v to q_dot
#     ang_vel0 = v0[3:6]
#     ang_vel1 = v1[3:6]
#
#     quat0 = q0[3:7]
#     quat1 = q1[3:7]
#
#     F0 = vel_to_q_dot(quat0, ang_vel0)
#     F1 = vel_to_q_dot(quat1, ang_vel1)
#
#     qdot0 = q0
#     qdot0[0:3] = v0[0:3]
#     qdot0[3:7] = F0(quat0, ang_vel0)
#     qdot0[7:] = v0[6:]
#
#     qdot1 = q1
#     qdot1[0:3] = v1[0:3]
#     qdot1[3:7] = F1(quat1, ang_vel1)
#     qdot1[7:] = v1[6:]
#
#     return cs.Function(
#         "collocation_vel",
#         [q0, v0, q1, v1],
#         [
#             (3 * dt**2 / 4.0)
#             * (
#                 (2.0 / dt**3) * q0
#                 + (1.0 / dt**2) * qdot0
#                 - (2.0 / dt**3) * q1
#                 + (1 / dt**2) * qdot1
#             )
#             + dt
#             * (
#                 -(3.0 / dt**2) * q0
#                 - (2.0 / dt) * qdot0
#                 + (3.0 / dt**2) * q1
#                 - (1.0 / dt) * qdot1
#             )
#             + v0
#         ],
#     )
#
#
# def MidpointCollocationAcc(dt, q0, v0, q1, v1):
#     # Transform v to q_dot
#     ang_vel0 = v0[3:6]
#     ang_vel1 = v1[3:6]
#
#     quat0 = q0[3:7]
#     quat1 = q1[3:7]
#
#     F0 = vel_to_q_dot(quat0, ang_vel0)
#     F1 = vel_to_q_dot(quat1, ang_vel1)
#
#     qdot0 = q0
#     qdot0[0:3] = v0[0:3]
#     qdot0[3:7] = F0(quat0, ang_vel0)
#     qdot0[7:] = v0[6:]
#
#     qdot1 = q1
#     qdot1[0:3] = v1[0:3]
#     qdot1[3:7] = F1(quat1, ang_vel1)
#     qdot1[7:] = v1[6:]
#
#     return cs.Function(
#         "collocation_acc",
#         [x0, x1],
#         [
#             3
#             * dt
#             * (
#                 (2.0 / dt**3) * q0
#                 + (1.0 / dt**2) * qdot0
#                 - (2.0 / dt**3) * q1
#                 + (1 / dt**2) * qdot1
#             )
#             - (3.0 / dt**2) * q0
#             - (2.0 / dt) * qdot0
#             + (3.0 / dt**2) * q1
#             - (1.0 / dt) * qdot1
#         ],
#     )
#
#
# def MidpointCollocationPosOnStance(dt, p0):
#     return cs.Function(
#         "collocation_pos_on_stance",
#         [p0],
#         [
#             (dt**3 / 6.0) * ((2.0 / dt**3) * p0 - (2.0 / dt**3) * p0)
#             + (dt**2 / 4) * (-(3.0 / dt**2) * p0 + (3.0 / dt**2) * p0)
#             + p0
#         ],
#     )
#
#
# def MidpointCollocationPosPrevStance(dt, p0, p1, v1):
#     return cs.Function(
#         "collocation_pos_prev_stance",
#         [p0, p1, v1],
#         [
#             (dt**3 / 6.0) * ((2.0 / dt**3) * p0 - (2.0 / dt**3) * p1 + (1 / dt**2) * v1)
#             + (dt**2 / 4) * (-(3.0 / dt**2) * p0 + (3.0 / dt**2) * p1 - (1.0 / dt) * v1)
#             + p0
#         ],
#     )
#
#
# def MidpointCollocationPosNextSwing(dt, p0, v0, p1, v1):
#     return cs.Function(
#         "collocation_pos_next_swing",
#         [p0, v0, p1, v1],
#         [
#             (dt**3 / 6.0)
#             * (
#                 (2.0 / dt**3) * p0
#                 + (1.0 / dt**2) * v0
#                 - (2.0 / dt**3) * p1
#                 + (1 / dt**2) * v1
#             )
#             + (dt**2 / 4)
#             * (
#                 -(3.0 / dt**2) * p0
#                 - (2.0 / dt) * v0
#                 + (3.0 / dt**2) * p1
#                 - (1.0 / dt) * v1
#             )
#             + (dt / 2) * v0
#             + p0
#         ],
#     )
#
#
# def MidpointCollocationPosNextStance(dt, p0, v0, p1):
#     return cs.Function(
#         "collocation_pos_next_stance",
#         [p0, v0, p1],
#         [
#             (dt**3 / 6.0)
#             * ((2.0 / dt**3) * p0 + (1.0 / dt**2) * v0 - (2.0 / dt**3) * p1)
#             + (dt**2 / 4) * (-(3.0 / dt**2) * p0 - (2.0 / dt) * v0 + (3.0 / dt**2) * p1)
#             + (dt / 2) * v0
#             + p0
#         ],
#     )
