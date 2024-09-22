from fetidp import fetidp_solver
from monolithic import monolithic_solver

if __name__ == "__main__":
    # Number of elements per direction in mesh
    n = [32, 24]

    # Number of subdomains
    N = [3, 4]

    # Discretization degree
    degree = 2

    fetidp_solver(n, N, degree)
    monolithic_solver(n, N, degree)
