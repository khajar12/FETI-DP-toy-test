import numpy as np
import numpy.typing as npt
import scipy.sparse

import dolfinx.fem

from global_dofs_manager import GlobalDofsManager
from utils import write_solution

type SparseMatrix = scipy.sparse._csr.csr_matrix


def create_primal_Schur(
    gbl_dofs_mngr: GlobalDofsManager,
    Krr: SparseMatrix,
    Krp: SparseMatrix,
    Kpp: SparseMatrix,
) -> SparseMatrix:
    """Creates the global primal Schur complement.

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        Krr (SparseMatrix): Local stiffnes matrix for the remainder dofs.
        Krp (SparseMatrix): Local stiffnes matrix for the remainder-primal
            dofs.
        Kpp (SparseMatrix): Local stiffnes matrix for the primal dofs.

    Returns:
        SparseMatrix: Global primal Schur complement.
    """

    # Global Schur.
    P = gbl_dofs_mngr.get_num_primals()
    SPP = scipy.sparse.csr_matrix((P, P), dtype=Krr.dtype)

    #raise ValueError("To be implemented!")

    # Loop over each subdomain
    N = gbl_dofs_mngr.get_num_subdomains()
    for s_id in range(N):
        
        Ap = gbl_dofs_mngr.create_Ap(s_id)
        # Compute the local Schur complement
        Urp = scipy.sparse.linalg.spsolve(Krr, Krp)
        Spp_local = Kpp - Krp.T @ Urp

        # Add local Schur complement to the global Schur complement
        SPP += Ap @ Spp_local @ Ap.T

    return SPP




def create_F_and_d_bar(
    gbl_dofs_mngr: GlobalDofsManager,
) -> tuple[SparseMatrix, npt.NDArray[np.float64]]:
    """Assembles the stiffness matrix and right-hand-side vector of the global
    dual problem.

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.

    Returns:
        tuple[SparseMatrix, npt.NDArray[np.float64]]: Stiffness matrix and
            right-hand-side vector of the dual problem.
    """

    subdomain = gbl_dofs_mngr.subdomain


    K, f = subdomain.K, subdomain.f

    rem_dofs = subdomain.get_remainder_dofs()
    primal_dofs = subdomain.get_primal_dofs()

    #raise ValueError("To be implemented")
    K_rr = K[rem_dofs,:][:,rem_dofs]
    K_rp = K[rem_dofs,:][:,primal_dofs]
    K_pp = K[primal_dofs,:][:,primal_dofs]

    f_r = f[rem_dofs]
    f_p = f[primal_dofs]

    T_dr_local = subdomain.create_Tdr()

    # Initialize the global stiffness matrix for primal DOFs
    num_global_primal_dofs = gbl_dofs_mngr.get_num_primals()
    K_PP = scipy.sparse.csr_matrix((num_global_primal_dofs, num_global_primal_dofs))
    f_P = np.zeros(num_global_primal_dofs)

    # number of subdomains
    N = gbl_dofs_mngr.get_num_subdomains()

    # Initialize the block-diagonal remainder stiffness matrix and the force vector
    K_rr_blocks = [K_rr.copy() for _ in range(N)]
    K_RR = scipy.sparse.block_diag(K_rr_blocks)
    f_R = np.concatenate([f_r.copy() for _ in range(N)])

    # Lists to store K_rp blocks and B_r blocks for each subdomain
    K_rp_list = []
    B_r_list = []

    # Loop over each subdomain 
    for subdomain_idx in range(N):
        A_p = gbl_dofs_mngr.create_Ap(subdomain_idx)

        K_PP += A_p @ K_pp @ A_p.T
        f_P += A_p @ f_p

        # Construct the K_rp block for the current subdomain
        K_rP = K_rp @ A_p.T
        K_rp_list.append(K_rP)

        # calculate the B_d and B_r
        B_d = gbl_dofs_mngr.create_Bd(subdomain_idx)
        B_r = B_d @ T_dr_local
        B_r_list.append(B_r)

    # Stack the K_rp blocks vertically to form the global KRP matrix
    K_RP = scipy.sparse.vstack(K_rp_list)

    # KPR
    K_PR = K_RP.T

    # Horizontally stack the B_r blocks to form the global BR matrix
    B_R = scipy.sparse.hstack(B_r_list)

    # the primal Schur complement
    S_PP = create_primal_Schur(gbl_dofs_mngr, K_rr, K_rp, K_pp)

    # Assemble the global dual stiffness matrix F and the right-hand side dbar
    F = (B_R @ np.linalg.inv(K_RR.toarray()) @ B_R.T +
         B_R @ np.linalg.inv(K_RR.toarray()) @ K_RP @ np.linalg.inv(S_PP.toarray()) @ K_PR @ np.linalg.inv(K_RR.toarray()) @ B_R.T)

    d_bar = (B_R @ np.linalg.inv(K_RR.toarray()) @ f_R -
             B_R @ np.linalg.inv(K_RR.toarray()) @ K_RP @ np.linalg.inv(S_PP.toarray()) @ (f_P - K_PR @ np.linalg.inv(K_RR.toarray()) @ f_R))

    return F, d_bar


def reconstruct_uP(
    gbl_dofs_mngr: GlobalDofsManager,
    lambda_: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reconstructs the global primal solution vector from the multipliers
    lambda_ at the interfaces (the solution of the global dual problem).

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        lambda_ (npt.NDArray[np.float64]): Dual problem solution vector.

    Returns:
        npt.NDArray[np.float64]: Global primal solution vector.
    """

    subdomain = gbl_dofs_mngr.subdomain

    #raise ValueError("To be implemented!")
    K, f = subdomain.K, subdomain.f

    rem_dofs = subdomain.get_remainder_dofs()
    primal_dofs = subdomain.get_primal_dofs()

    # number of subdomains
    N = gbl_dofs_mngr.get_num_subdomains()

    num_global_primal_dofs = gbl_dofs_mngr.get_num_primals()
    K_PP = scipy.sparse.csr_matrix((num_global_primal_dofs, num_global_primal_dofs))
    f_P = np.zeros(num_global_primal_dofs)

    #raise ValueError("To be implemented")
    K_rr = K[rem_dofs,:][:,rem_dofs]
    K_rp = K[rem_dofs,:][:,primal_dofs]
    K_pp = K[primal_dofs,:][:,primal_dofs]

    f_r = f[rem_dofs]
    f_p = f[primal_dofs]

    # the primal Schur complement
    S_PP = create_primal_Schur(gbl_dofs_mngr, K_rr, K_rp, K_pp)

    # Initialize the block-diagonal remainder stiffness matrix and the force vector
    K_rr_blocks = [K_rr.copy() for _ in range(N)]
    K_RR = scipy.sparse.block_diag(K_rr_blocks)
    f_R = np.concatenate([f_r.copy() for _ in range(N)])

    # Lists to store K_rp blocks and B_r blocks for each subdomain
    K_rp_list = []
    B_r_list = []

    T_dr_local = subdomain.create_Tdr()

    # Loop over each subdomain
    for subdomain_idx in range(N):
        A_p = gbl_dofs_mngr.create_Ap(subdomain_idx)

        # Accumulate the global primal stiffness matrix and primal force vector
        K_PP += A_p @ K_pp @ A_p.T
        f_P += A_p @ f_p

        # K_rp
        K_rP = K_rp @ A_p.T
        K_rp_list.append(K_rP)

        # calculate the B_d and B_r
        B_d = gbl_dofs_mngr.create_Bd(subdomain_idx)
        B_r = B_d @ T_dr_local
        B_r_list.append(B_r)

    # Stack the K_rp blocks vertically to form the global KRP matrix
    K_RP = scipy.sparse.vstack(K_rp_list)

    # KPR 
    K_PR = K_RP.T

    # Horizontally stack the B_r blocks to form the global BR matrix
    B_R = scipy.sparse.hstack(B_r_list)

    uP = np.linalg.inv(S_PP.toarray()) @ ((f_P - K_PR @ np.linalg.inv(K_RR.toarray()) @ f_R) + (K_PR @ np.linalg.inv(K_RR.toarray()) @ B_R.T) @ lambda_)


    return uP


def reconstruct_Us(
    gbl_dofs_mngr: GlobalDofsManager,
    uP: npt.NDArray[np.float64],
    lambda_: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float64]]:
    """Reconstructs the full solution vector of every subdomain, once the
    multipliers lambda_ at the interfaces (the solution of the global dual
    problem), and the global primal solution uP have been computed.

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        uP (npt.NDArray[np.float64]): Global primal solution vector.
        lambda_ (npt.NDArray[np.float64]): Dual problem solution vector.


    Returns:
        list[npt.NDArray[np.float64]]: Vector of solutions for every subdomain.
    """

    subdomain = gbl_dofs_mngr.subdomain

    rem_dofs = subdomain.get_remainder_dofs()
    primal_dofs = subdomain.get_primal_dofs()

    #raise ValueError("To be implemented.")
    K, f = subdomain.K, subdomain.f

    K_rr = K[rem_dofs,:][:,rem_dofs]
    K_rp = K[rem_dofs,:][:,primal_dofs]

    us = []
    N = gbl_dofs_mngr.get_num_subdomains()

    # Loop over each subdomain
    for s_id in range(N):
        Ap = gbl_dofs_mngr.create_Ap(s_id)

        u = np.zeros(K.shape[0], dtype=K.dtype)
        u[primal_dofs] = Ap.T @ uP

        # Ensure the correct size of uP is used for the current subdomain
        uP_local = uP[:len(primal_dofs)] 

        # Solve for remainder DOFs
        u[rem_dofs] = np.linalg.solve(K_rr.toarray(), K_rp @ uP_local)

        us.append(u)
        
    return us


def reconstruct_solutions(
    gbl_dofs_mngr: GlobalDofsManager,
    lambda_: npt.NDArray[np.float64],
) -> list[dolfinx.fem.Function]:
    """Reconstructs the solution function of every subdomain, starting from the
    multipliers lambda_ at the interfaces (the solution of the global dual
    problem).

    Args:
        gbl_dofs_mngr (GlobalDofsManager): Global dofs manager.
        lambda_ (npt.NDArray[np.float64]): Solution of the dual problem.

    Returns:
        list[dolfinx.fem.Function]: List of functions describing the solution
            in every single subdomain. The FEM space of every function has the
            same structure as the one of the reference subdomain, but placed
            at its corresponding position.
    """

    uP = reconstruct_uP(gbl_dofs_mngr, lambda_)
    Urs = reconstruct_Us(gbl_dofs_mngr, uP, lambda_)

    us = []
    N = gbl_dofs_mngr.get_num_subdomains()
    for s_id in range(N):
        subdomain_i = gbl_dofs_mngr.create_subdomain(s_id)
        uh = dolfinx.fem.Function(subdomain_i.V)
        uh.x.array[:] = Urs[s_id]
        us.append(uh)

    return us


def write_output_subdomains(us: list[dolfinx.fem.Function]) -> None:
    """Writes the solution functions as VTX folders named
    "subdomain_i.pb" into the folder "results", with i running from 0
    to N-1 (N being the number of subdomains). One folder per subdomain.

    To visualize them, the ".pb" folders can be directly imported in
    ParaView.

    Args:
        us (list[dolfinx.fem.Function]): List of functions describing
            the solution in every subdomain.
    """

    for s_id, uh in enumerate(us):
        write_solution(uh, "results", f"subdomain_{s_id}")


def fetidp_solver(n: list[int], N: list[int, int], degree: int) -> None:
    """Solves the Poisson problem with N subdomains per direction using
    a non-preconditioned FETI-DP solver.

    Every subdomain is considered to have n elements per direction, and
    the input degree is used for discretizing the solution.

    The generated solutions are written to the folder "results" as VTX
    folders named "subdomain_i.pb", with i running from 0 to N-1.
    One file per subdomain. Thy can be visualized using ParaView.

    Args:
        n (list[int]): Number of elements per direction in every single
            subdomain.
        N (list[int]): Number of subdomains per direction.
        degree (int): Discretization space degree.
    """

    assert N[0] * N[1] > 1, "Invalid number of subdomains."
    assert degree > 0, "Invalid degree."

    gbl_dofs_mngr = GlobalDofsManager.create_unit_square(n, degree, N)

    F, dbar = create_F_and_d_bar(gbl_dofs_mngr)
    lambda_ = scipy.sparse.linalg.spsolve(F, dbar)

    us = reconstruct_solutions(gbl_dofs_mngr, lambda_)
    write_output_subdomains(us)
