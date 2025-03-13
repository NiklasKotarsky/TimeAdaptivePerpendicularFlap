# Import required libs
import precice
import numpy as np
import os
from scipy.interpolate import NearestNDInterpolator
from scipy.sparse.linalg import spsolve as solver
from scipy.sparse.linalg import spilu, cg, LinearOperator
# spsolve
import ufl
from dune.fem.space import lagrange as solutionSpace, finiteVolume
from dune.ufl import DirichletBC, Constant
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.operator import galerkin, linear
from dune.fem.utility import Sampler
from dune.grid import cartesianDomain
from dune.alugrid import aluSimplexGrid
from dune.fem.function import uflFunction, gridFunction
from dune.ufl import expression2GF
from dirk import SDIRK2 as SDIRK
from operators import Operator
import json

# Geometry and material properties
dim = 2  # number of dimensions
H = 1

# rho = 3000
# E = 4000000
W = 0.1

nu = 0.3

rho = 3000
E = 4000000


mu = Constant(E / (2.0 * (1.0 + nu)))

lambda_ = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

# create Mesh
n_x_Direction = 10
n_y_Direction = 100
domain = cartesianDomain([-W / 2, 0], [W/2, H], [n_x_Direction, n_y_Direction])
mesh = aluSimplexGrid(domain)
h = Constant(H / n_y_Direction)

# create Function Space
#V = solutionSpace(mesh, dimRange=4, order=2, storage='numpy')
#This has to be order 1!!
V = solutionSpace(mesh, dimRange=4, order=1, storage='numpy')
displacement = V.interpolate([0, 0, 0, 0], name="displacement")

# Trial and Test Functions
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(ufl.triangle)

# function known from previous timestep
u_0 = uflFunction(mesh, name="u_0", order=V.order,
                  ufl=Constant([0, 0, 0, 0]))

u_n = V.interpolate(u_0, name='u_n')
v_n = V.interpolate(u_0, name='v_n')
a_n = V.interpolate(u_0, name='a_n')
u_cp = V.interpolate(u_0, name='u_cp')

u_np1 = V.interpolate(u_0, name='u_np1')
saved_u_old = V.interpolate(u_0, name='saved_u_old')


f_N_function = V.interpolate([1, 0, 0, 0], name="f_N_function")
u_function = V.interpolate([0, 0, 0, 0], name="u_function")

sample_boundary = Sampler(u_np1)
v1, bd_1 = sample_boundary.boundarySample(boundaryId=1)
v4, bd_4 = sample_boundary.boundarySample(boundaryId=4)
v2, bd_2 = sample_boundary.boundarySample(boundaryId=2)
vertices = np.concatenate([v1, v4, v2])

# define coupling mesh
mesh_name = "Solid-Mesh"
participant = precice.Participant("Solid", "../precice-config.xml", 0, 1)

# define coupling mesh
mesh_name = "Solid-Mesh"
vertex_ids = participant.set_mesh_vertices(mesh_name, vertices)

if participant.requires_initial_data():
    boundary_data = np.concatenate([bd_1, bd_4, bd_2])
    participant.write_data(mesh_name, "Displacement", vertex_ids, boundary_data[:, 0:2])


# creating the boundary force function
force = NearestNDInterpolator(vertices, np.zeros([len(vertices), 2]))


class Force():
    def __init__(self, force):
        self.force = force


force_interp = Force(force)


@gridFunction(mesh, name="u_gamma", order=2)
def u_gamma(xg):
    return force_interp.force(np.array(xg))[0]


# Initialize the coupling interface
precice_dt = participant.initialize()

#dune_dt = precice_dt/1000  # if dune_dt == precice_dt, no subcycling is applied
# dune_dt = 0.01  # if dune_dt < precice_dt, subcycling is applied
dt = Constant(1e3)

#timestep controls and stuff
def rms(v):
    r = np.sqrt(np.dot(v, v) / len(v))
    return r

def timestep_controller(dt, err, TOL, k=2):
    return dt*(TOL/rms(err))**(1/k)


# Define strain
def epsilon(u):
    return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


# Define Stress tensor
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(dim) + 2 * mu * epsilon(u)


# Define Mass form
def m(u, v):
    return rho * ufl.inner(u, v) * ufl.dx


# Elastic stiffness form
def k(u, v):
    return ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx

# Update acceleration


eps = 1e-8
bc_bottom = DirichletBC(V, Constant([0, 0, 0, 0]), x[1] < 0.0 + eps)

# parameters=params_solver1
params_solver1 = {"newton.tolerance": 1e-6,  # tolerance for newton solver
                  "newton.verbose": True,  # toggle iteration output
                  "newton.linear.tolerance": 1e-6,  # tolerance for linear solver
                  # absolute or "relative" or "residualreduction"
                  "newton.linear.errormeasure": "absolute",
                  # (see table below)
                  "newton.linear.preconditioning.method": "none",
                  # "boomeramg" "pilu-t" "parasails"
                  "newton.linear.preconditioning.hypre.method": "boomeramg",
                  "newton.linear.preconditioning.iteration": 5,  # iterations for preconditioner
                  "newton.linear.preconditioning.relaxation": 1.0,  # omega for SOR and ILU
                  "newton.linear.maxiterations": 1000,  # max number of linear iterations
                  "newton.linear.verbose": True,     # toggle linear iteration output
                  "newton.linear.preconditioning.level": 0}

un1 = ufl.as_vector([du[0], du[1]])
un2 = ufl.as_vector([du[2], du[3]])

unold1 = ufl.as_vector([u_n[0], u_n[1]])
unold2 = ufl.as_vector([u_n[2], u_n[3]])

phi1 = ufl.as_vector([v[0], v[1]])
phi2 = ufl.as_vector([v[2], v[3]])

L1 = (ufl.inner(un1, phi1) - ufl.inner(unold1, phi1)
      - dt*ufl.inner(un2, phi1))*ufl.dx

#This should not change anything?? both are linear!
# L1 = (k(un1, phi1) - k(unold1, phi1)
#       - dt*k(un2, phi1))

# L1 = (m(un1, phi1) - m(unold1, phi1)
#       - dt*m(un2, phi1))

# Factor 2 missing here with a high probability!
#or more if second order elements are used!

L2 = (m(un2, phi2)-m(unold2, phi2)) + dt*(k(un1, phi2) -
                                          1/ufl.FacetArea(V)*ufl.inner(u_gamma, phi2)*ufl.ds)

# Weak form in dot u form not currently used!
# L1 = (ufl.inner(un1, phi1) - ufl.inner(un2, phi1))*ufl.dx

# L2 = m(un2, phi2) + k(un1, phi2) - 1 / \
#     ufl.FacetArea(V)*ufl.inner(u_gamma, phi2)*ufl.ds


res = L1 + L2
# res = m(a_np1, v) + k(du, v)
# m(a_np1, v) + - ufl.inner(u_gamma, v)*ufl.ds

a_form = ufl.lhs(res)
L_form = ufl.rhs(res)


# Domain [-W / 2, 0], [W/2, H

Scheme = solutionScheme([a_form == L_form, bc_bottom],
                        solver="gmres", parameters=params_solver1)


Matrix = linear(Scheme)
res = u_np1.copy(name="residual")
zeros = V.interpolate(Constant([0, 0, 0, 0]), "zeros")
Scheme(zeros, res)
Scheme.jacobian(u_np1, Matrix)


def solve(u_bar, t_bar, alpha, rtol):
    u_n.as_numpy[:] = u_bar
    dt.value = alpha

    # read data from preCICE
    read_data = participant.read_data(mesh_name, "Force", vertex_ids, t_bar)
    force_interp.force = NearestNDInterpolator(
        vertices, read_data)
    # Scheme.solve(target=zeros)

    Scheme(zeros, res)
    Scheme.jacobian(zeros, Matrix)
    # ilu = spilu(Matrix.as_numpy, drop_tol=1e-10, fill_factor=20, drop_rule=None, permc_spec=None,
    #             diag_pivot_thresh=None, relax=None, panel_size=None, options=None)
    # precond = LinearOperator(ilu.shape, matvec=ilu.solve)

    # vec, conv = cg(Matrix.as_numpy, - res.as_numpy, x0=u_bar, tol=rtol,
    #                maxiter=100, M=precond, callback=None, atol=rtol)

    # x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None
    vec = solver(Matrix.as_numpy, - res.as_numpy)
    return vec, True


# parameters for Time-Stepping
t = 0.0
n = 0
E_ext = 0

# For writing the output in vtk files
if not os.path.exists("output"):
    os.makedirs("output")
vtk = mesh.sequencedVTK(
    "output/displacement", pointdata=[u_np1, saved_u_old, v_n, a_n])

TOL = 1e-5
#TOL=5e-5
precice_dt = participant.get_max_time_step_size()

time_stepper = SDIRK(solver=solve,
                        atol=1e-5,
                        rtol=1e-4,
                        solver_tol=None,
                        initial_time_step=1e-6,
                        max_time_step=float(precice_dt))

delta_t = min(1e-5,precice_dt)

timeSteps = []
positions = []
timeStepsOld = []
positionsOld = []


while participant.is_coupling_ongoing():


    # write checkpoint
    if participant.requires_writing_checkpoint():
        delta_t_old = delta_t
        u_cp = u_n.copy()
        t_cp = t
        n_cp = n
        timeStepsOld = [i for i in timeSteps]
        positionsOld = [i for i in positions]
    # delta_t = time_stepper.step_to_time(precice_dt, u_np1)
    _, unew, error_estimate = time_stepper(u_np1, t=0, dt=delta_t)
    u_np1.as_numpy[:] = unew
    print(error_estimate)
    print(delta_t)

    _, bd_1 = sample_boundary.boundarySample(boundaryId=1)
    _, bd_4 = sample_boundary.boundarySample(boundaryId=4)
    _, bd_2 = sample_boundary.boundarySample(boundaryId=2)
    boundary_data = np.concatenate([bd_1, bd_4, bd_2])
    
    positions.append(bd_4[0][0])
    timeSteps.append(t)
    t += delta_t

    # Write new displacements to preCICE
    participant.write_data(
        mesh_name,"Displacement", vertex_ids, boundary_data[:, 0:2])

    # Call to advance coupling, also returns the optimum time step value
    participant.advance(delta_t)
    delta_t = timestep_controller(delta_t, error_estimate, TOL)
    precice_dt = participant.get_max_time_step_size()
    delta_t = min(delta_t, precice_dt)


    # Either revert to old step if timestep has not converged or move to next timestep
    if participant.requires_reading_checkpoint():
        delta_t = delta_t_old
        u_n.assign(u_cp)
        u_np1.assign(u_cp)
        t = t_cp
        n = n_cp
        timeSteps = [i for i in timeStepsOld]
        positions = [i for i in positionsOld]


    else:
        u_n.assign(u_np1)
        n += 1

results = dict()
results["timesteps"] = timeSteps
results["positions"] = positions
with open("resultsBeamPos", 'w') as myfile:
    myfile.write(json.dumps(results, indent=2, sort_keys=True))

# Plot tip displacement evolution
vtk()

participant.finalize()
