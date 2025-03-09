import os
import casadi as cs
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn


def main():

    # Find and load the urdf
    path_to_curr_folder = os.path.dirname(os.path.realpath(__file__))
    urdffile = os.path.join(path_to_curr_folder, "../urdf", "pendulum.urdf")
    urdf = open(urdffile, "r").read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # q = cs.MX.sym("q")
    # qdot = cs.MX.sym("qdot")
    # qddot = kindyn.aba(q=0, v=0, tau=0.0)["a"]
    # print(qddot)

    u_lim = 0.1

    T = 4.0  # Horizon
    N = 100  # Control intervals

    dt = T / (N - 1)  # dt per spline

    # Create casadi variables
    th = cs.MX.sym("th")
    thd = cs.MX.sym("thd")
    thdd = cs.MX.sym("thdd")
    u = cs.MX.sym("u")

    # Objective
    L = u**2  # + (th - np.pi) ** 2 # Uncomment for minimum time
    # L = (th - cs.pi) ** 2

    Cost = cs.Function("cost", [u], [L],)

    fd = kindyn.aba()
    PendulumDynamics = cs.Function(
        "pendulum_dynamics",
        [th, thd, u],
        [fd(q=th, v=thd, tau=u)["a"]],
    )

    x0 = cs.MX.sym("x0", 2)
    x1 = cs.MX.sym("th1", 2)
    u0 = cs.MX.sym("u0")
    u1 = cs.MX.sym("u1")

    # Collocation point position from cubic hermite spline interpolation
    CollocationPos = cs.Function(
        "collocation_pos",
        [x0, x1],
        [
            (dt**3 / 6.0)
            * (
                (2.0 / dt**3) * x0[0]
                + (1.0 / dt**2) * x0[1]
                - (2.0 / dt**3) * x1[0]
                + (1 / dt**2) * x1[1]
            )
            + (dt**2 / 4)
            * (
                -(3.0 / dt**2) * x0[0]
                - (2.0 / dt) * x0[1]
                + (3.0 / dt**2) * x1[0]
                - (1.0 / dt) * x1[1]
            )
            + (dt / 2) * x0[1]
            + x0[0]
        ],
    )

    # Collocation point velocity from cubic hermite spline interpolation
    CollocationVel = cs.Function(
        "collocation_vel",
        [x0, x1],
        [
            (3 * dt**2 / 4.0)
            * (
                (2.0 / dt**3) * x0[0]
                + (1.0 / dt**2) * x0[1]
                - (2.0 / dt**3) * x1[0]
                + (1 / dt**2) * x1[1]
            )
            + dt
            * (
                -(3.0 / dt**2) * x0[0]
                - (2.0 / dt) * x0[1]
                + (3.0 / dt**2) * x1[0]
                - (1.0 / dt) * x1[1]
            )
            + x0[1]
        ],
    )

    CollocationAcc = cs.Function(
        "collocation_acc",
        [x0, x1],
        [
            3
            * dt
            * (
                (2.0 / dt**3) * x0[0]
                + (1.0 / dt**2) * x0[1]
                - (2.0 / dt**3) * x1[0]
                + (1 / dt**2) * x1[1]
            )
            - (3.0 / dt**2) * x0[0]
            - (2.0 / dt) * x0[1]
            + (3.0 / dt**2) * x1[0]
            - (1.0 / dt) * x1[1]
        ],
    )

    FirstOrderInterpolation = cs.Function("midpoint", [u0, u1], [(u0 + u1) / dt])

    InitialAcc = cs.Function(
        "initial_acc",
        [x0, x1],
        [
            -(6.0 / dt**2) * x0[0]
            - (4.0 / dt) * x0[1]
            + (6.0 / dt**2) * x1[0]
            - (2.0 / dt) * x1[1]
        ],
    )

    FinalAcc = cs.Function(
        "final_acc",
        [x0, x1],
        [
            (12.0 / dt**2) * x0[0]
            + (6.0 / dt) * x0[1]
            - (12 / dt**2) * x1[0]
            + (6.0 / dt) * x1[1]
        ],
    )

    # Cost = cs.Function("cost", [u], [u**2])

    # Evaluate at a test point
    # thdd = InitialAcc(0.0, 0.0, 0.0, 0.0)

    # Start with an empty NLP
    w = []  # variables
    w0 = []  # variable initial guess
    lbw = []  # variable lower bound
    ubw = []  # varialble upper bound
    J = 0  # cost
    g = []  # constraints
    lbg = []  # lower bound for contstraints
    ubg = []  # upper bound for contraints

    ## NOTE: We add a final control node, to be able to compute the interpolation at the final collocation point.
    for k in range(N):
        # New state variable
        Xk = cs.MX.sym("X_" + str(k), 2)
        w += [Xk]
        lbw += [-cs.inf, -10.0]
        ubw += [cs.inf, 10.0]
        w0 += [0, 0]

        # New control variable
        Uk = cs.MX.sym("U_" + str(k))
        w += [Uk]
        lbw += [-0.5]
        ubw += [0.5]
        w0 += [0]

    # Bound initial and final state
    lbw[0:2] = [0.0, 0.0]
    ubw[0:2] = [0.0, 0.0]

    lbw[-3:] = [cs.pi, 0.0, 0.0]
    ubw[-3:] = [cs.pi, 0.0, 0.0]

    X = w[0::2]
    U = w[1::2]

    # Fill the dynamics constraint vector and the cost function
    for k in range(N - 1):
        X0 = X[k]
        X1 = X[k + 1]
        U0 = U[k]
        U1 = U[k + 1]

        # Calculate th, thd and thdd at the collocation point
        th_c = CollocationPos(X0, X1)
        thd_c = CollocationVel(X0, X1)
        thdd_c = CollocationAcc(X0, X1)

        # Approximate control with first order intepolation
        u_c = FirstOrderInterpolation(U0, U1)

        thdd = PendulumDynamics(th_c, thd_c, u_c)

        g += [thdd - thdd_c]
        lbg += [0]
        ubg += [0]

        c = Cost(U0)
        J = J + c

    # Add the acceleration constraints
    for k in range(N - 2):
        X0 = X[k]
        X1 = X[k + 1]
        X2 = X[k + 2]

        thdd0 = FinalAcc(X0, X1)
        thdd1 = InitialAcc(X1, X2)

        g += [thdd0 - thdd1]
        lbg += [0]
        ubg += [0]

    # Create an NLP solver
    prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g)}
    solver = cs.nlpsol("solver", "ipopt", prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    # Plot the solution
    th_opt = w_opt[0::3]
    thd_opt = w_opt[1::3]
    u_opt = w_opt[2::3]

    # print(w_opt)
    # print(thd_opt)
    # print(u_opt)

    tgrid = [T / N * k for k in range(N)]
    print(len(tgrid))

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, th_opt, "--")
    plt.plot(tgrid, thd_opt, "-")
    plt.step(tgrid, u_opt, "-.")
    plt.xlabel("t")
    plt.legend(["θ", "θd", "u"])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
