import numpy as np
import numpy.typing as npt

from pathlib import Path

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import dolfinx.io
import ufl
import basix

# Checking version 0.8.0 of dolfinx. Likely, versions 0.8.x are also valid.
assert [int(v) for v in dolfinx.__version__.split(".")] == [0, 8, 0]


def get_faces_dofs(V: dolfinx.fem.FunctionSpace) -> list[npt.NDArray[np.int32]]:
    """Gets the degrees-of-freedom for the 4 faces of the single domain.

    The ordering of the four faces follows the basix convention for
    quadrilaterals.
    See https://docs.fenicsproject.org/basix/v0.8.0/index.html

    Args:
        V (dolfinx.fem.FunctionSpace): Function space of the single domain.

    Returns:
        list[npt.NDArray[np.int32]]: Sorted arrays of degrees-of-freedom.
            One for every face.
    """
    mesh = V.mesh
    xmin = np.min(mesh.geometry.x, axis=0)
    xmax = np.max(mesh.geometry.x, axis=0)

    # Faces ordering follows the basix convention for quadrilaterals
    # See https://docs.fenicsproject.org/basix/v0.8.0/index.html
    markers = [
        lambda x: np.isclose(x[1], xmin[1]),
        lambda x: np.isclose(x[0], xmin[0]),
        lambda x: np.isclose(x[0], xmax[0]),
        lambda x: np.isclose(x[1], xmax[1]),
    ]

    faces_dofs = []
    for face_id in range(4):
        face_dofs = dolfinx.fem.locate_dofs_geometrical(V, markers[face_id])
        faces_dofs.append(np.sort(face_dofs))

    return faces_dofs


def write_solution(u: dolfinx.fem.Function, folder: str, filename_str: str) -> None:
    """Writes the given function to a VTX folder with extension ".pb".
    It can visualized by importing it in ParaView.

    Args:
        u (dolfinx.fem.Function): Function to dump into a VTX file.
        folder (str): Folder in which files are placed.
            It is created if it doesn't exist.
        filename_str (str): Name (without extension) of the folder containing
            the function.
    """

    results_folder = Path(folder)
    results_folder.mkdir(exist_ok=True, parents=True)

    filename = results_folder / filename_str
    filename = filename.with_suffix(".bp")

    V = u.function_space
    comm = V.mesh.comm

    with dolfinx.io.VTXWriter(comm, filename, [u]) as vtx:
        vtx.write(0.0)


def create_Cartesian_mesh_nodes(
    pts_1D: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Creates the coordinates matrix of the nodes of a 2D tensor-product
    Cartesian mesh.
    The ordering of the generated nodes follow the lexicographical
    ordering convetion.

    Args:
        pts_1D: Points coordinates along the two parametric directions.

    Returns:
        nodes: Coordinates of the nodes stored in a 2D np.ndarray.
            The rows correspond to the different points and columns
            to the coordinates.
    """

    assert len(pts_1D) == 2

    x = np.meshgrid(pts_1D[0], pts_1D[1], indexing="xy")

    nodes = np.zeros((x[0].size, 2), dtype=x[0].dtype)
    for dir in range(2):
        nodes[:, dir] = x[dir].ravel()

    return nodes


def create_2D_tensor_prod_mesh_conn(n_cells: list[int]) -> list[list[int]]:
    """Creates the cells' connectivity of a 2D Cartesian mesh made of
    linear quadrilaterals.

    Args:
        n_cells: Number of cells per direction in the Cartesian mesh.

    Returns:
        conn: Generated connectivity. It is a list, where
            every entry is a list of nodes ids.
            The connectivity of each cells follows the DOLFINx convention.
            See https://docs.fenicsproject.org/basix/v0.8.0/index.html.
    """

    assert len(n_cells) == 2 and n_cells[0] > 0 and n_cells[1] > 0

    n_cells = np.array(n_cells)
    n_pts = n_cells + 1

    # First cell.
    first_cell = np.array([0, 1, n_pts[0], n_pts[0] + 1])

    # First line of cells.
    conn = first_cell + np.arange(0, n_cells[0]).reshape(-1, 1)

    # Full connecitivity
    conn = conn.ravel() + np.arange(0, n_pts[0] * n_cells[1], n_pts[0]).reshape(-1, 1)

    conn = conn.reshape(np.prod(n_cells), len(first_cell))
    return conn.tolist()


def create_2D_mesh(
    n: list[int],
    p0: list[float] = [0.0, 0.0],
    p1: list[float] = [1.0, 1.0],
) -> dolfinx.mesh.Mesh:
    """Creates a 2D mesh of rectangular domain, using linear quadrilaterals,
    with n[0] and n[1] elements per direction, respectively. The rectangular
    domain is defined its left-bottom and rigth-top corners, p0, and p1,
    respectively.

    Args:
        n (list[int]): Number of elements per direction.
        p0 (list[float], optional): Left-bottom corner of the rectangular
            domain.  Defaults to [0.0, 0.0].
        p1 (list[float], optional): Right-top corner of the rectangular domain.
            Defaults to [1.0, 1.0].

    Returns:
        dolfinx.mesh.Mesh: Generated mesh.
    """

    assert len(n) == 2 and n[0] > 0 and n[1] > 0

    pts_0 = np.linspace(p0[0], p1[0], n[0] + 1)
    pts_1 = np.linspace(p0[1], p1[1], n[1] + 1)
    coords = create_Cartesian_mesh_nodes([pts_0, pts_1])

    conn = create_2D_tensor_prod_mesh_conn(n)

    domain = ufl.Mesh(
        basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,), dtype=np.float64)
    )

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_mesh(comm, conn, coords, domain)

    return mesh
