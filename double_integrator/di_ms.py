import casadi as cs
import numpy as np

u_lim = 10.0
x_des = [1.0, 0.0]

T = 2.0  # Horizon
M = 100  # Shooting states
N = 1  # Control intervals per shooting state
DT = T / M / N

# Create casadi variables
x = cs.MX.sym("x", 2)
u = cs.MX.sym("u")

# Objective
L = cs.MX(u**2)
# L = cs.MX((x[0] - x_des[0]) ** 2)
Cost = cs.Function("f", [x, u], [L])

# Calculate the dynamics and the cost function
xdd = cs.MX(u)
DoubleIntegratorDynamics = cs.Function("f", [x, u], [xdd])

# Start with an empty NLP
w = []  # variables
w0 = []  # variable initial guess
lbw = []  # variable lower bound
ubw = []  # varialble upper bound
J = 0  # cost
g = []  # constraints
lbg = []  # lower bound for contstraints
ubg = []  # upper bound for contraints
equality = []

# Initial state
Xk = cs.MX.sym("X0", 2)
w += [Xk]
lbw += [0, 0]
ubw += [0, 0]
w0 += [0, 0]

for i in range(M):
    ## Add control variables
    X = Xk
    # Integrate for N variables
    for j in range(N):
        Uk = cs.MX.sym("U" + str(i) + "_" + str(j))
        w += [Uk]
        lbw += [-u_lim]
        ubw += [u_lim]
        w0 += [0]

        a = DoubleIntegratorDynamics(X, Uk)
        v = X[1] + a * DT
        p = X[0] + X[1] * DT
        X = cs.vertcat(p, v)
        c = Cost(X, Uk)
        J += c

    # Add intermediate state as variable
    Xk = cs.MX.sym("X" + str(i + 1), 2)
    w += [Xk]
    lbw += [-cs.inf, -cs.inf]
    ubw += [cs.inf, cs.inf]
    w0 += [0, 0]

    # Add defect constraint
    g += [X - Xk]
    lbg += [0, 0]
    ubg += [0, 0]
    equality += [True] * 2

lbw[-2:] = x_des
ubw[-2:] = x_des

# Create an NLP solver
opts = {
    "ipopt.linear_solver": "ma57",
    "ipopt.tol": 0.0001,
    "ipopt.constr_viol_tol": 0.0001,
    "ipopt.max_iter": 2000,
}

prob = {"f": J, "x": cs.vertcat(*w), "g": cs.vertcat(*g)}
solver = cs.nlpsol("solver", "ipopt", prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol["x"].full().flatten()

# Plot the solution
x_opt = w_opt[0 :: N + 2]
xd_opt = w_opt[1 :: N + 2]
u_opt = []
for k in range(M):
    start_idx = k * (2 + N) + 2
    end_idx = (k + 1) * (2 + N)
    u_opt.extend(w_opt[start_idx:end_idx])

tgrid = [T / M * k for k in range(M + 1)]
ugrid = [T / M / N * k for k in range(M * N + 1)]

import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt, "--")
plt.plot(tgrid, xd_opt, "-")
plt.step(ugrid, cs.vertcat(cs.DM.nan(1), u_opt), "-.")
plt.xlabel("t")
plt.legend(["p", "v", "u"])
plt.grid()
plt.show()
