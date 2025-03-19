import os
from functions import Parameters

G1_29DOF_PARAMS = Parameters()
G1_29DOF_PARAMS.total_duration = 0.5
G1_29DOF_PARAMS.num_shooting_states = 10
G1_29DOF_PARAMS.num_rollout_states = 1
G1_29DOF_PARAMS.model_urdf_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "urdf/g1_29dof.urdf"
)
G1_29DOF_PARAMS.model_foot_sole_link_name = "left_foot_point_contact"
G1_29DOF_PARAMS.contact_ee_names = ["lf", "rf"]
G1_29DOF_PARAMS.ee_phase_sequence = {"lf": list((10,)), "rf": list((10,))}
G1_29DOF_PARAMS.is_init_contact = {"lf": True, "rf": True}
G1_29DOF_PARAMS.num_contact_points_per_ee = {
    "lf": 4,
    "rf": 4,
}
G1_29DOF_PARAMS.dim_f_ext = 3
G1_29DOF_PARAMS.q0 = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    -0.6,  # left_hip_pitch_joint
    0.0,  # left_hip_roll_joint
    0.0,  # left_hip_yaw_joint
    1.2,  # left_knee_joint
    -0.6,  # left_ankle_pitch_joint
    0.0,  # left_ankle_roll_joint
    -0.6,  # right_hip_pitch_joint
    0.0,  # right_hip_roll_joint
    0.0,  # right_hip_yaw_joint
    1.2,  # right_knee_joint
    -0.6,  # right_ankle_pitch_joint
    0.0,  # right_ankle_roll_joint
    0.0,  # waist_yaw_joint
    0.0,  # waist_roll_joint
    0.0,  # waist_pitch_joint
    0.0,  # left_shoulder_pitch_joint
    0.0,  # left_shoulder_roll_joint
    0.0,  # left_shoulder_yaw_joint
    0.0,  # left_elbow_joint
    0.0,  # left_wrist_roll_joint
    0.0,  # left_wrist_pitch_joint
    0.0,  # left_wrist_yaw_joint
    0.0,  # right_shoulder_pitch_joint
    0.0,  # right_shoulder_roll_joint
    0.0,  # right_shoulder_yaw_joint
    0.0,  # right_elbow_joint
    0.0,  # right_wrist_roll_joint
    0.0,  # right_wrist_pitch_joint
    0.0,  # right_wrist_yaw_joint
]


TALOS_PARAMS = Parameters()
TALOS_PARAMS.total_duration = 0.5
TALOS_PARAMS.num_shooting_states = 10
TALOS_PARAMS.num_rollout_states = 1
TALOS_PARAMS.opt_dt = False
TALOS_PARAMS.model_urdf_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "urdf/talos_fixed_gripper.urdf"
)
TALOS_PARAMS.model_foot_sole_link_name = "left_sole_link"
TALOS_PARAMS.contact_ee_names = ["lf", "rf"]
TALOS_PARAMS.ee_phase_sequence = {
    "lf": list((10,)),
    "rf": list((10,)),
    # "lf": list((9, 18, 33)),
    # "rf": list((33, 18, 9)),
}
TALOS_PARAMS.is_init_contact = {"lf": True, "rf": True}
TALOS_PARAMS.num_contact_points_per_ee = {
    "lf": 4,
    "rf": 4,
}
TALOS_PARAMS.dim_f_ext = 3
TALOS_PARAMS.q0 = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,  # leg_left_1_joint
    0.05,  # leg_left_2_joint
    -0.68983,  # leg_left_3_joint
    1.33113,  # leg_left_4_joint
    -0.64130,  # leg_left_5_joint
    -0.05,  # leg_left_6_joint
    0.0,  # leg_right_1_joint
    -0.05,  # leg_right_2_joint ,
    -0.68983,  #  leg_right_3_joint
    1.33113,  #  leg_right_4_joint
    -0.64130,  #  leg_right_5_joint
    0.05,  #  leg_right_6_joint
    0.0,  #  torso_1_joint
    0.0,  #  torso_2_joint
    -0.03397,  #  arm_left_1_joint ,
    0.308661,  #  arm_left_2_joint
    0.03272,  #  arm_left_3_joint
    -1.0322,  #  arm_left_4_joint
    0.004009,  #  arm_left_5_joint
    0.003586,  #  arm_left_6_joint
    0.0256151,  #  arm_left_7_joint
    -0.033978,  #  arm_right_1_joint
    -0.30866,  #  arm_right_2_joint
    0.032729,  #  arm_right_3_joint
    -1.03224,  #  arm_right_4_joint
    0.004009,  #  arm_right_5_joint
    0.0035862,  #  arm_right_6_joint
    0.0256151,  #  arm_right_7_joint
    0.18,  #  head_1_joint
    0.0,  #  head_2_joint
]
