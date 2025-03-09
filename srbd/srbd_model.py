import numpy as np
import casadi as cs
from functions import quat_to_R_T


class SRBDModel:
    def __init__(self, mass, inertia, contact_ees):
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = cs.inv_minor(self.inertia)

        self.nq = 7
        self.nv = 7

        self.num_contact_ees = contact_ees

        # Feet offset
        self.dx = 0.15
        self.dy = 0.1
        self.dz = 0.1

    def nq(self):
        return self.nq

    def nv(self):
        return self.nv

    def dynamics(self, q, f_total, tau, w_b):
        quat = q[3:7]
        pddot = f_total / mass
        wdot_b = self.inertia_inv @ (
            quat_to_R_T(quat).T @ tau - cs.cross(w_b, self.inertia @ w_b)
        )

        return [pddot - wdot_b]
