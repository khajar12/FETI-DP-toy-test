import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import dolfinx.io
import ufl

from utils import create_2D_mesh, write_solution, get_faces_dofs

# Checking version 0.8.0 of dolfinx. Likely, versions 0.8.x are also valid.
assert [int(v) for v in dolfinx.__version__.split(".")] == [0, 8, 0]


def monolithic_solver(n: list[int], N: list[int, int], degree: int):
    """Solves the Poisson problem with N subdomains, n elements per direction
    in every subdomain, and the given degree for discretizing the solution, but
    using a standard monolithic solver instead of FETI-DP.

    The generated solutions is written to the folder "results" as a single VTX
    folder named "single_domain.pb". This can be imported in ParaView
    for visualization.

    Args:
        n (list[int]): Number of elements per direction in every single
            subdomain.
        N (list[int]): Number of subdomains per direction.
        degree (int): Discretization space degree.
    """

    assert N[0] > 0 and N[1] > 0, "Invalid number of subdomains."
    assert degree > 0, "Invalid degree."

    Lx = 1.0
    Ly = 1.0
    n = [n[0] * N[0], n[1] * N[1]]
    mesh = create_2D_mesh(n, p1=[Lx, Ly])

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = dolfinx.fem.Constant(mesh, 1.0) * v * ufl.dx

    facets_dofs = get_faces_dofs(V)
    zero = dolfinx.fem.Function(V)
    bcs = []
    for face_id in range(len(facets_dofs)):
        bcs.append(dolfinx.fem.dirichletbc(zero, facets_dofs[face_id]))

    # Setting LU solver.
    solver_options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options=solver_options
    )
    uh = problem.solve()

    write_solution(uh, "results", "single_domain")
