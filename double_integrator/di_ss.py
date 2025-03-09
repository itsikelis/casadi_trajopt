import casadi as cs
import numpy as np


u_lim = 10.0
x_des = [1.0, 0.0]

T = 2.0  # Horizon
N = 200  # Control intervals
DT = T / N

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

## Add control variables
X = Xk
for k in range(N):
    Uk = cs.MX.sym("U" + str(k))
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

# Add final state as variable
Xf = cs.MX.sym("X1", 2)
w += [Xf]
lbw += x_des
ubw += x_des
equality += [True] * 2
w0 += [0, 0]

# Add defect constraint
g += [X - Xf]
lbg += [0, 0]
ubg += [0, 0]

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
x0 = w_opt[0:2]
xf = w_opt[-2:]
u_opt = w_opt[2:-2]

# Integrate states to plot
x_opt = []
xd_opt = []
p = w_opt[0]
v = w_opt[1]

x_opt += [x0[0]]
xd_opt += [x0[1]]

# Integrate the dynamics to get each state
for u in u_opt:
    a = u
    v = v + a * DT
    p = p + v * DT
    x_opt += [p]
    xd_opt += [v]

tgrid = [T / N * k for k in range(N + 1)]

import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt, "--")
plt.plot(tgrid, xd_opt, "-")
plt.step(tgrid, cs.vertcat(cs.DM.nan(1), u_opt), "-.")
plt.xlabel("t")
plt.legend(["x", "xd", "u"])
plt.grid()
plt.show()
