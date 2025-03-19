import numpy as np
import casadi as cs

import pinocchio as pin
from pinocchio import casadi as cpin


class Parameters:
    def __init__(self) -> None:
        self.total_duration: float = 0.0
        self.num_shooting_states: int = 0
        self.num_rollout_states: int = 0

        self.opt_dt: bool = False

        self.model_urdf_path: string = ""
        self.model_foot_sole_link_name: string = ""

        self.contact_ee_names: list[str] = []
        self.ee_phase_sequence: dict[str, int] = {}
        self.is_init_contact: dict[str, bool] = {}

        self.num_contact_points_per_ee: dict[str, int] = {}

        self.dim_f_ext: int = 0

        self.q0: list[float] = []

    def assert_validity(self) -> None:
        for ee in self.ee_phase_sequence:
            num_states = np.sum(np.array(self.ee_phase_sequence[ee]))
            assert self.num_shooting_states == num_states


class Model:
    def __init__(self, params: Parameters):
        self.model: pin.Model = pin.buildModelFromUrdf(params.model_urdf_path)
        self.data: pin.Data = self.model.createData()

        self.cmodel: cpin.Model = cpin.Model(self.model)
        self.cdata: cpin.Model = self.cmodel.createData()

        self.q0: list[float] = params.q0.copy()
        pin.framesForwardKinematics(self.model, self.data, np.array(self.q0))
        frame_id = self.model.getFrameId(params.model_foot_sole_link_name)
        base_pos_offset = self.data.oMf[frame_id].translation
        base_rot_offset = R_to_quat(self.data.oMf[frame_id].rotation)
        self.q0[2] = float(-base_pos_offset[2])
        self.q0[3:7] = (base_rot_offset).tolist()

    def nq(self) -> int:
        return self.model.nq

    def nv(self) -> int:
        return self.model.nv

    def lower_pos_lim(self) -> list[float]:
        return self.model.lowerPositionLimit.tolist()

    def upper_pos_lim(self) -> list[float]:
        return self.model.upperPositionLimit.tolist()

    def lower_vel_lim(self) -> list[float]:
        return (-self.model.velocityLimit).tolist()

    def upper_vel_lim(self) -> list[float]:
        return self.model.velocityLimit.tolist()

    def lower_joint_effort_lim(self) -> list[float]:
        return (-self.model.effortLimit).tolist()[7:]

    def upper_joint_effort_lim(self) -> list[float]:
        return self.model.effortLimit.tolist()[7:]

    def total_mass(self) -> float:
        pin.computeTotalMass(self.model, self.data)
        return self.data.mass[0]

    def frame_dist_from_ground(self, frame_name: str, q: cs.SX) -> cs.SX:
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        frame_id = self.model.getFrameId(frame_name)
        ee_pos = self.cdata.oMf[frame_id].translation
        return ee_pos

    def frame_jacobian(self, frame_name: str, q: cs.SX) -> cs.SX:
        cpin.computeJointJacobians(self.cmodel, self.cdata, q)
        frame_id = self.model.getFrameId(frame_name)
        ref_frame = pin.LOCAL_WORLD_ALIGNED
        jac = cpin.getFrameJacobian(self.cmodel, self.cdata, frame_id, ref_frame)
        return jac

    def frame_jacobian_time_var(self, frame_name: str, q: cs.SX) -> cs.SX:
        cpin.computeJointJacobians(self.cmodel, self.cdata, q)
        frame_id = self.model.getFrameId(frame_name)
        ref_frame = pin.LOCAL_WORLD_ALIGNED
        jac = cpin.getFrameJacobianTimeVariation(
            self.cmodel, self.cdata, frame_id, ref_frame
        )
        return jac

    def inverse_dynamics(self, q: cs.SX, v: cs.SX, a: cs.SX, JtF_sum: cs.SX) -> cs.SX:
        return cpin.rnea(self.cmodel, self.cdata, q, v, a) - JtF_sum

    def angular_momentum(self, q: cs.SX, v: cs.SX, a: cs.SX) -> cs.SX:
        cpin.computeCentroidalMomentumTimeVariation(self.cmodel, self.cdata, q, v, a)
        return self.cdata.hg.angular

        return q0


class Environment:
    def __init__(self):
        self.ground_mu: float = 0.8
        self.ground_z: float = 0.0

        self.ground_n: list[float] = [0.0, 0.0, 1.0]
        self.ground_b: list[float] = [0.0, 1.0, 0.0]
        self.ground_t: list[float] = [1.0, 0.0, 0.0]

        self.grav: float = 9.81


def addFrictionConeConstraint(
    env: Environment, fc: cs.SX, g: list[cs.SX], lbg: list[float], ubg: list[float]
):
    ## Friction cone constraint ##
    # Normal component
    g += [cs.dot(fc, env.ground_n)]
    lbg += [0.0]
    ubg += [cs.inf]
    # Tangentials
    lim = (env.ground_mu / cs.sqrt(2.0)) * cs.dot(fc, env.ground_n)
    g += [
        cs.dot(fc, env.ground_b) + lim,
        -cs.dot(fc, env.ground_b) + lim,
        cs.dot(fc, env.ground_t) + lim,
        -cs.dot(fc, env.ground_t) + lim,
    ]
    lbg += [0.0, 0.0, 0.0, 0.0]
    ubg += [cs.inf, cs.inf, cs.inf, cs.inf]


def phase_composer(contact_durations, swing_durations, is_init_contact):
    phase_sequence = []
    type = ""
    duration = 0.0

    is_contact = is_init_contact
    num_phases = len(contact_durations + swing_durations)

    c_idx = 0
    s_idx = 0
    for _ in range(num_phases):
        if is_contact:
            type = "contact"
            duration = contact_durations[c_idx]
            c_idx += 1
        else:
            type = "swing"
            duration = swing_durations[s_idx]
            s_idx += 1
        phase_sequence += [Phase(type, duration)]
        is_contact = not is_contact

    return phase_sequence


def is_in_contact(t: float, phase_sequence: list) -> bool:
    sum = 0.0
    prev_sum = 0.0
    for phase in phase_sequence:
        sum += phase.duration
        if t <= sum + 1e-6:
            return True if phase.type == "contact" else False


def node_is_in_contact(
    k: int, node_contact_sequence: list, is_init_contact: bool
) -> bool:
    in_contact = is_init_contact
    sum = 0
    for num_nodes in node_contact_sequence:
        sum += num_nodes
        if k <= sum - 1:
            return in_contact
        in_contact = not in_contact


def G_T(q: cs.SX) -> cs.SX:
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    G_T = cs.SX.zeros(4, 3)
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


def v_to_qdot(nq: int, q: cs.SX, v: cs.SX) -> cs.SX:
    qdot = cs.SX.zeros(nq)

    qdot[0:3] = v[0:3]
    qdot[3:7] = 0.5 * G_T(q[3:7]) @ v[3:6]
    qdot[7:] = v[6:]

    return qdot


def R_to_quat(R: cs.SX) -> cs.SX:
    q = np.zeros(4)
    q[3] = 0.5 * cs.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2])
    q[0] = 0.5 * (R[2, 1] - R[1, 2]) / (4.0 * q[3])
    q[1] = 0.5 * (R[0, 2] - R[2, 0]) / (4.0 * q[3])
    q[2] = 0.5 * (R[1, 0] - R[0, 1]) / (4.0 * q[3])
    return q
