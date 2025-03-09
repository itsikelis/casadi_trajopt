import casadi as cs
import numpy as np

N = 5  # Control intervals
T = 2.0  # Horizon
DT = T / N

nx = 2
nu = 1

x_des = [1.0, 0.0]

opti = cs.Opti()
var_xs = [opti.variable(nx) for _ in range(1)]
var_us = [opti.variable(nu) for _ in range(N)]
totalcost = 0
opti.subject_to(var_xs[0] == [0.0, 0.0])
# opti.subject_to(var_xs[1] == [1.0, 0.0])

X = var_xs[0]
for k in range(N):
    totalcost += var_us[k] ** 2
    a = var_us[k]
    v = X[1] + a * DT
    p = X[0] + X[1] * DT
    X = cs.vertcat(p, v)

opti.subject_to(X == x_des)

#
# ### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt")  # set numerical backend
# opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][:nq])))

sol = opti.solve_limited()
print([opti.value(var_x) for var_x in var_us])
