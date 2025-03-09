import numpy as np
import pinocchio as pin


# Quaternion dot to angular velocity transform
def G(q):
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    G = np.zeros(3, 4)
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


def v_to_qdot(qdot, q, v):
    qdot[0:3] = v[0:3]
    qdot[3:7] = 0.5 * G_T(q[3:7]) @ v[3:6]
    qdot[7:] = v[6:]


# Collocation point position from cubic hermite spline interpolation
def collocation_q(pin_model, T, x0, x1, t):
    nq = pin_model.nq
    nv = pin_model.nv

    q0 = x0[0:nq]
    v0 = x0[nq:]
    q1 = x1[0:nq]
    v1 = x1[nq:]

    # Transform v to q_dot
    qdot0 = np.zeros(nq)
    v_to_qdot(qdot0, q0, v0)
    qdot1 = np.zeros(nq)
    v_to_qdot(qdot1, q1, v1)

    a = (2 / T**3) * q0 + (1 / T**2) * qdot0 - (2 / T**3) * q1 + (1 / T**2) * qdot1
    b = -(3 / T**2) * q0 - (2 / T) * qdot0 + (3 / T**2) * q1 - (1 / T) * qdot1
    c = qdot0
    d = q0

    return a * t**3 + b * t**2 + c * t + d


# Collocation point velocity from cubic hermite spline interpolation
def collocation_qdot(pin_model, T, x0, x1, t):
    nq = pin_modl.nq
    nv = pin_modl.nv

    q0 = x0[0:nq]
    v0 = x0[nq:]
    q1 = x1[0:nq]
    v1 = x1[nq:]

    # Transform v to q_dot
    qdot0 = no.zeros(nq)
    v_to_qdot(qdot0, q0, v0)
    qdot1 = np.zeros(nq)
    v_to_qdot(qdot1, q1, v1)

    a = (2 / T**3) * q0 + (1 / T**2) * qdot0 - (2 / T**3) * q1 + (1 / T**2) * qdot1
    b = -(3 / T**2) * q0 - (2 / T) * qdot0 + (3 / T**2) * q1 - (1 / T) * qdot1
    c = qdot0
    d = q0

    return 3 * a * t**2 + 2 * b * t + c


# Get normalised time of spline
def normalise_time(durations, t):
    total_duration = np.sum(durations)
    if t < total_duration:
        sum = 0.0
        prev_sum = 0.0
        for i in range(len(durations)):
            sum += durations[i]
            if t <= sum - 1e-3:
                t_norm = t - prev_sum
                return (i, t_norm)
            prev_sum = sum
    # Keep this outside an else statement, because for loop may fail to return due to floating point error.
    return (len(durations) - 1, durations[-1])


# Get desired u from zero order interpolation
def u_desired(u_nodes, durations, t):
    idx, _ = normalise_time(durations, t)
    u_des = u_nodes[:, idx]
    return u_des


# Get desired state from integrating u with semi-implicit Euler
def x_desired(q_nodes, qdot_nodes, qddot, durations, t):
    idx, t_norm = normalise_time(durations, t)
    q_k = q_nodes[:, idx]
    qdot_k = qdot_nodes[:, idx]

    qdot_des = qdot_k + qddot * t_norm
    q_des = q_k + qdot_des * t_norm
    return (q_des, qdot_des)


# Calculate inverse dynamics using pinocchio and return the torques.
def inverse_dynamics(pin_model, pin_data, q, v, a, f, contact_frame_names):
    JtF_sum = np.zeros(pin_model.nq - 7)

    for frame in contact_frame_names:
        # Calculate contact jacobian wrt to the world frame
        frame_idx = pin_model.getFrameId(frame)

        pin.computeJointJacobians(pin_model, pin_data, q)
        pin.framesForwardKinematics(pin_model, pin_data, q)
        jac = pin.getFrameJacobian(
            pin_model, pin_data, frame_idx, pin.ReferenceFrame(pin.LOCAL_WORLD_ALIGNED)
        )

        JtF_sum += jac[:3, 6:].T @ f[frame]

    tau = pin.rnea(pin_model, pin_data, q, v, a)[6:] - JtF_sum

    return tau
