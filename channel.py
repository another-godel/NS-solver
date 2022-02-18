from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
from ufl.tensors import unit_list

T = 10.0           # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density

mesh = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_ = Function(V) # newest computed value of u
p_n = Function(Q)
p_ = Function(Q) # newest computed value of p

f = Constant((0, 0))

# To avoid code regeneration
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

n = FacetNormal(mesh)

def epsilon(u):
    return sym(nabla_grad(u)) 

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define boundary
# x here is an array indicating position x = (x, y)
# near function comes from FEniCS and performs a test 
# with tolerance: abs(x[0] - 0) < 3E-16
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls = 'near(x[0], 0) || near(x[0], 2) || near(x[1], 0)'

# Boundary conditions
bc_u_noslip = DirichletBC(V, Constant((0, 0)), walls)
bc_p_inflow = DirichletBC(Q, Constant(8), inflow)
bc_p_outflow = DirichletBC(Q, Constant(0), outflow)
bc_u = [bc_u_noslip]
bc_p = [bc_p_inflow, bc_p_outflow]

# P1: Find tentative velocity
U = (u + u_n)/2
F1 = rho*dot((u - u_n) / k, v)*dx + dot(dot(u_n, nabla_grad(u_n)), v)*dx\
    + inner(sigma(U, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds\
     - dot(mu*nabla_grad(U)*n, v)*ds - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# P2: Find new pressure
F2 = dot(nabla_grad(p), nabla_grad(q))*dx - dot(nabla_grad(p_n), nabla_grad(q))*dx\
    + rho/k*div(u_)*q*dx  
a2 = lhs(F2)
L2 = rhs(F2)

# P3: Find new velocity
F3 = dot(u, v)*dx - dot(u_, v)*dx + k*rho*dot(nabla_grad(p_ - p_n), v)*dx
a3 = lhs(F3)
L3 = rhs(F3)

# Since the lhs of the variational problem is time independent, we can define
# it before the time-stepping 

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bc_u]
[bc.apply(A2) for bc in bc_p]

# Time-stepping
t = 0
for n in range(num_steps):
    t += dt
    # P1
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bc_u]
    solve(A1, u_.vector(), b1)
    # P2
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bc_p]
    solve(A2, p_.vector(), b2)
    # P3
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    plot(u_)

    u_n.assign(u_)
    p_n.assign(p_)

plt.show()