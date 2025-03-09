import casadi as ca
from numpy import sin, cos, tan, pi

N = 20
T = 1
dt = T / N

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)

ode = ca.vertcat(x[1] * dt, u * dt)

# Discretize system
# dt = ca.MX.sym("dt")
sys = {}
sys["x"] = x
sys["u"] = u
sys["ode"] = ode * dt  # Time scaling

intg = ca.integrator(
    "intg", "rk", sys, 0, 1, {"simplify": True, "number_of_finite_elements": 4}
)

F = ca.Function("F", [x, u], [intg(x0=x, u=u)["xf"]], ["x", "u"], ["xnext"])

nx = x.numel()
nu = u.numel()

f = 0  # Objective
x = []  # List of decision variable symbols
lbx = []
ubx = []  # Simple bounds
x0 = []  # Initial value
g = []  # Constraints list
lbg = []
ubg = []  # Constraint bounds
equality = []  # Boolean indicator helping structure detection
p = []  # Parameters
p_val = []  # Parameter values


X = []
U = []
for k in range(N + 1):
    sym = ca.MX.sym("X", nx)
    x.append(sym)
    X.append(sym)
    x0.append(ca.vertcat(k * dt, 0.0))
    lbx.append(-ca.DM.inf(nx, 1))
    ubx.append(ca.DM.inf(nx, 1))

    sym = ca.MX.sym("T")
    x.append(sym)

    if k < N:
        sym = ca.MX.sym("U", nu)
        x.append(sym)
        U.append(sym)
        x0.append(0)
        lbx.append(0)
        ubx.append(0)

# Round obstacle
pos0 = ca.vertcat(0.2, 5)
r0 = 1

X0 = ca.MX.sym("X0", nx)
p.append(X0)
p_val.append(ca.vertcat(0, 0))

# f = sum(T)  # Time Optimal objective
for k in range(N):
    # Multiple shooting gap-closing constraint
    g.append(X[k + 1] - F(X[k], U[k]))
    lbg.append(ca.DM.zeros(nx, 1))
    ubg.append(ca.DM.zeros(nx, 1))
    equality += [True] * nx

    if k == 0:
        # Initial constraints
        g.append(X[0] - X0)
        lbg.append(ca.DM.zeros(nx, 1))
        ubg.append(ca.DM.zeros(nx, 1))
        equality += [True] * nx

    if k == N - 1:
        # Final constraints
        g.append(X[-1])
        lbg.append(ca.vertcat(1, 0))
        ubg.append(ca.vertcat(1, 0))
        equality += [True]

print(X)
exit()

# Add some regularization
for k in range(N + 1):
    f += X[k][0] ** 2

# Solve the problem

nlp = {}
nlp["f"] = f
nlp["g"] = ca.vcat(g)
nlp["x"] = ca.vcat(x)
nlp["p"] = ca.vcat(p)

options = {}
options["expand"] = True
options["fatrop"] = {"mu_init": 0.1}
options["structure_detection"] = "auto"
options["debug"] = True
options["equality"] = equality

# (codegen of helper functions)
# options["jit"] = True
# options["jit_temp_suffix"] = False
# options["jit_options"] = {"flags": ["-O3"],"compiler": "ccache gcc"}

solver = ca.nlpsol("solver", "fatrop", nlp, options)

res = solver(
    x0=ca.vcat(x0),
    lbg=ca.vcat(lbg),
    ubg=ca.vcat(ubg),
    lbx=ca.vcat(lbx),
    ubx=ca.vcat(ubx),
    p=ca.vcat(p_val),
)
