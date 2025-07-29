# -*- coding: utf-8 -*-
# stokes_solver.py
"""
Created on Fri Apr  4 07:15:06 2025

@author: skje
"""

import warnings
import pickle
import gzip
import os
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import scipy.sparse as sp
from scipy.spatial import cKDTree

from mesher import create_mesh, load_mesh
from datetime import datetime

import pypardiso
from numba import njit

class Simulation:
    """
   Simulation of incompressible Stokes flow in a 2D domain with support for
   periodic boundary conditions and mixed P2-P1 finite elements.

   Attributes
   ----------
   mesh : Mesh
       A structured mesh object containing geometry, connectivity, and boundary classification.

   f : callable
       Body force function, accepting (x, y) and returning a tuple (fx, fy).

   inner_radius : float
       Radius of the central hole in the domain geometry.

   mu : float, optional
       Dynamic viscosity of the fluid [Pa·s]. Default is 0.001.

   rho : float, optional
       Fluid density [kg/m³]. Default is 1000.

   geometry_height : float
       Thickness of the pseudo-3D channel, used to scale forces.

   u1, u2 : ndarray or None
       Computed velocity components after solving.

   p : ndarray or None
       Computed pressure field after solving.

   lhs, rhs : scipy sparse matrix and ndarray
       Assembled linear system (before solve).

   Methods
   -------
   get_description() -> str
       Returns a short formatted description string with geometry and material parameters.

   save(filename)
       Serializes the simulation object to a file using pickle.

   static load(filename) -> Simulation
       Loads a Simulation object from a pickle file.

   assemble()
       Builds the global linear system including periodicity enforcement and boundary conditions.

   solve()
       Solves the Stokes system using Pypardiso and stores velocity and pressure fields.

   run()
       Executes the full simulation pipeline: assemble → debug → solve → plot.

   plot()
       Generates and saves plots of velocity magnitude, vectors, and pressure field.

   apply_periodic_bc()
       Applies no-slip conditions on walls and anchors the pressure DOF.

   debug_system(tol=1e-12)
       Checks symmetry, sparsity, and approximate rank of the assembled system.

   calculate_permeability(direction='y') -> float
       Estimates average permeability using average velocity and known body force.
   """
   
    def __init__(self, mesh, f, geometry_length, inner_radius, geometry_height, mu=0.001, rho=1000.0, periodic_bc=False, nodim=True):
        self.mesh = mesh
        self.f = f
        self.mu = mu
        self.rho = rho
        self.geometry_length= geometry_length
        self.geometry_height = geometry_height
        self.inner_radius = inner_radius
        self.periodic_bc = periodic_bc
        self.nodim = nodim
        
        self.u1 = None
        self.u2 = None
        self.p = None
        self.lhs = None
        self.rhs = None
        
    def get_description(self):
        """
        Returns a short string summarizing the simulation parameters 
        (viscosity, density, mesh size, height, radius).
        """
        
        desc = ""
        if self.mu is not None:
            desc += f"mu={self.mu:.2e}"
        if self.rho is not None:
            desc += f", rho={self.rho:.2e}"
        if self.mesh.triangles is not None:
            desc += f", N={self.mesh.triangles.shape[0]}"
        if self.geometry_height is not None:
            desc += f", h={self.geometry_height:.2e}"
        if self.inner_radius is not None:
            desc += f", r={self.inner_radius:.2e}"
            
        return desc
        
    def save(self, filename):
        """
        Saves the simulation object to a compressed file using pickle and gzip.
        """
        
        self.lhs = None
        self.rhs = None
        with gzip.open(filename, "wb") as f:
            pickle.dump(self, f)
            

    def assemble(self):
        """
        Assembles the global linear system (LHS matrix and RHS vector) for the Stokes problem.
        
        This includes building stiffness, mass, divergence, and load matrices, applying boundary
        conditions, enforcing periodicity if enabled, and constructing the full block system.
        """

        nodes = self.mesh.nodes
        triangles = self.mesh.triangles # P2 elements (6 nodes per triangle)
        pressure_nodes = self.mesh.pressure_nodes # P1 nodes (vertices)
        
        if self.periodic_bc:
            periodic_map = self.mesh.periodic_map # Dictionary mapping slave_node_idx -> master_node_idx
        else:
            periodic_map = {} 
        
        print("\rAssembling matrices...", end="", flush=True)
        start_matrices = datetime.now()
        
        # Coefficient for the drag term (from - (8*mu/H^2)*v )
        alpha = (8*self.mu ) / (self.geometry_height **2)
        
        # Assemble matrices
        M = MassAssembler2D(nodes, triangles, periodic_map)
        A = StiffnessAssembler2D(nodes, triangles, periodic_map) 
        B1, B2 = DivergenceAssembler2D(nodes, triangles, pressure_nodes, self.mesh.pressure_index_map, periodic_map)
        b1, b2 = LoadAssembler2D(nodes, triangles, self.f, periodic_map)
        
        # Combine mass ad stiffness matrix,
        A11 = (A * self.mu ) + (alpha * M)

        end_matrices = datetime.now()
        print(f"\rAssembled matrices in {(end_matrices - start_matrices).total_seconds():.3f} seconds.")
        
        print("Assembling full system...", end="", flush=True)

        B1T = B1.T.tocsr()
        B2T = B2.T.tocsr()

        n = A11.shape[0]
        m = len(self.mesh.pressure_nodes)

        zero_A = sp.csr_matrix((n, n))
        zero_B = sp.csr_matrix((m, m))
        zero_b = np.zeros(m)

        self.lhs = sp.bmat([
            [A11,    zero_A, B1T],
            [zero_A, A11,    B2T],
            [B1,     B2,     zero_B]
        ], format='lil')

        self.rhs = np.concatenate([b1, b2, zero_b])

        # Explicitly enforce LIL before modifications
        self.lhs = self.lhs.tolil()
        
        end_system = datetime.now()
        print(f"\rAssembled final system {(end_system - end_matrices).total_seconds():.3f} seconds.")
                
        print("Handling BC's and cleaning up...", end="", flush=True)

        # Apply other boundary conditions AFTER enforcing periodicity constraints
        if self.periodic_bc:
            self.apply_periodic_bc()
        else:
            self.apply_bc()     
            
        # Convert to scr for faster solve.
        self.lhs = self.lhs.tocsr()
        
        # Optional: Check for empty rows *after* all modifications
        empty_rows = np.where(np.diff(self.lhs.indptr) == 0)[0]
        if len(empty_rows) > 0:
            warnings.warn(f"Empty rows in system matrix after boundary condition application: {empty_rows}")
            
        end_bc = datetime.now()
        print(f"\rHandled bc in {(end_bc - end_system).total_seconds():.3f} seconds.")
        
        # Clean up intermediate matrices to save memory
        del A, M, B1, B2, B1T, B2T, A11

        
    def solve(self):
        """
        Solves the assembled Stokes system using the PARDISO sparse solver.
        
        Extracts and stores the velocity components (u1, u2) and pressure field (p),
        and checks for residual and numerical stability.
        """

        sol = pypardiso.spsolve(self.lhs, self.rhs)
        
        residual = self.lhs @ sol - self.rhs
        self.res_norm = np.linalg.norm(residual)
        rhs_norm = np.linalg.norm(self.rhs)
        self.rel_res = self.res_norm / (rhs_norm + 1e-15)  # Avoid divide-by-zero
        
        if rhs_norm < 1e-12 and self.res_norm > 1e-9 : # If RHS is virtually zero, residual should also be
             warnings.warn(f"RHS vector norm is very small ({rhs_norm:.2e}), but residual norm is {self.res_norm:.2e}. Solution might be problematic.")
        elif self.rel_res > 1e-6: # Arbitrary threshold for concerning relative residual
             warnings.warn(f"Relative residual norm is high: {self.rel_res:.2e}. Solution accuracy may be poor.")

        n = self.mesh.nodes.shape[0]
    
        self.u1 = sol[:n]
        self.u2 = sol[n:2*n]
        self.p = sol[2*n:]


    def run(self):
        """
        Runs the complete simulation pipeline: assemble → debug → solve → plot.
        
        Measures and reports timing and residuals for each stage, and visualizes the results.
        """

        total_start = datetime.now()
    
        print("Assembling system...", end="", flush=True)
        self.assemble()
        end_assemble = datetime.now()
        print(f"\rAssembled system in {(end_assemble - total_start).total_seconds():.3f} seconds.")
        
        print("Debugging system...", end="", flush=True)
        self.debug_system()
        end_debug = datetime.now()
        print(f"\rDebugged system in {(end_debug - end_assemble).total_seconds():.3f} seconds.")

        print("Solving system...", end="", flush=True)
        self.solve()
        end_solver = datetime.now()
        print(f"\rSolved system in {(end_solver - end_debug).total_seconds():.3f} seconds.")
        print(f"Solver residual norm: {self.res_norm:.3e} (relative: {self.rel_res:.3e})")
        
        if np.any(np.isnan(self.u1)) or np.any(np.isnan(self.u2)) or np.any(np.isnan(self.p)):
            warnings.warn("Solution contains NaN values! Check boundary conditions and matrix assembly.")

        print("Plotting results...", end="", flush=True)
        self.plot()
        end_plot = datetime.now()
        print(f"\rPlotted results in {(end_plot - end_solver).total_seconds():.3f} seconds.")
    
        total_end = datetime.now()
        print(f"Total solver runtime: {(total_end - total_start).total_seconds():.3f} seconds.")

        
    def plot(self):
        """
        Generates and saves plots for velocity magnitude, pressure field,
        and velocity vectors using the current solution.
        """

        desc = self.get_description()
        plot_velocity_magnitude(self.mesh, self.u1, self.u2, title_suffix=desc, nodim = nodim)
        plot_pressure(self.mesh, self.p, title_suffix=desc, nodim = nodim)
        plot_velocity_vectors(self.mesh, self.u1, self.u2, title_suffix=desc, nodim = nodim) 

        
    def apply_periodic_bc(self):
        """
        - No-slip condition on interior wall nodes
        - Pressure anchoring at one pressure node
        """
        n = self.mesh.nodes.shape[0]
        lhs = self.lhs.tolil()  # Faster for row operations
        rhs = self.rhs
        
        # Enforce periodicity (slave-master)
        self.slave_nodes = np.array(list(self.mesh.periodic_map.keys()))
        self.master_nodes = np.array([self.mesh.periodic_map[node] for node in self.slave_nodes])
        n = self.mesh.nodes.shape[0]
        
        # --- u1 (x-velocity) DOFs ---
        slave_dof_u1 = self.slave_nodes         
        master_dof_u1 = self.master_nodes
        for s, m in zip(slave_dof_u1, master_dof_u1):
            self.lhs.rows[s] = [s, m]
            self.lhs.data[s] = [1.0, -1.0]
            self.rhs[s] = 0.0
        
        # --- u2 (y-velocity) DOFs ---
        slave_dof_u2 = self.slave_nodes + n
        master_dof_u2 = self.master_nodes + n
        for s, m in zip(slave_dof_u2, master_dof_u2):
            self.lhs.rows[s] = [s, m]
            self.lhs.data[s] = [1.0, -1.0]
            self.rhs[s] = 0.0
            
        # --- Pressure DOFs (P1 nodes) ---
        offset = 2 * n
        for slave_node, master_node in self.mesh.periodic_map.items():
            # Only apply periodic constraint if both nodes are pressure nodes
            if slave_node in self.mesh.pressure_index_map and master_node in self.mesh.pressure_index_map:
                slave_dof = offset + self.mesh.pressure_index_map[slave_node]
                master_dof = offset + self.mesh.pressure_index_map[master_node]
                self.lhs.rows[slave_dof] = [slave_dof, master_dof]
                self.lhs.data[slave_dof] = [1.0, -1.0]
                self.rhs[slave_dof] = 0.0
    
        # --- Set no-slip on interior boundary wall ---
        boundary_nodes = np.unique(np.array(self.mesh.interior_boundary_edges).flatten())
        boundary_dofs_u1 = boundary_nodes
        boundary_dofs_u2 = boundary_nodes + n
        all_boundary_dofs = np.concatenate([boundary_dofs_u1, boundary_dofs_u2])
    
        for dof in all_boundary_dofs:
            apply_dirichlet_bc(lhs, rhs, dof, 0.0)
            
        # --- Fix pressure at one arbitrary node ---
        target_y = 0 * np.max(self.mesh.nodes[:, 1])
        target_x = 0.5 * np.max(self.mesh.nodes[:, 0])

        slave_pressure_nodes = set(self.mesh.periodic_map.keys()) & set(self.mesh.pressure_index_map.keys())
        candidates = [idx for idx in self.mesh.pressure_nodes if idx not in slave_pressure_nodes]
        
        if not candidates:
            raise RuntimeError("No valid (non-slave) pressure nodes available for anchoring.")
        
        anchor_node = min(candidates, key=lambda idx: (self.mesh.nodes[idx][1] - target_y)**2 + (self.mesh.nodes[idx][0] - target_x)**2)

        print("\rAnchoring using node: ", anchor_node, ".    ")
    
        pressure_dof_local = self.mesh.pressure_index_map[anchor_node]
        anchor_dof = 2 * n + pressure_dof_local
        apply_dirichlet_bc(lhs, rhs, anchor_dof, 0.0)
    
    
    def apply_bc(self):
        n = len(self.mesh.nodes)
        lhs, rhs = self.lhs.tolil(), self.rhs
    
        # no‐slip on the side walls:
        wall_nodes = np.unique(np.array(self.mesh.wall_edges).flatten())
        for i in wall_nodes:
            apply_dirichlet_bc(lhs, rhs,  i    , 0.0)  # u_x = 0
            apply_dirichlet_bc(lhs, rhs,  i + n, 0.0)  # u_y = 0
    
        # flatten inlet/outlet edges → unique nodes
        inlet_nodes  = np.unique(np.array(self.mesh.inlet_edges).flatten())
        outlet_nodes = np.unique(np.array(self.mesh.outlet_edges).flatten())
    
        # pressure at inlet = 1
        for i in inlet_nodes:
            p_dof = 2*n + self.mesh.pressure_index_map[i]
            apply_dirichlet_bc(lhs, rhs, p_dof, 1.0)
    
        # pressure at outlet = 0
        for i in outlet_nodes:
            p_dof = 2*n + self.mesh.pressure_index_map[i]
            apply_dirichlet_bc(lhs, rhs, p_dof, 0.0)
    
        # convert back
        self.lhs, self.rhs = lhs.tocsr(), rhs

    
    def debug_system(self, tol=1e-12):
        """
        Debugs the assembled linear system by checking:
          - matrix shape
          - symmetry
          - sparsity
          - numerical rank
        
        Parameters
        ----------
        tol : float
            Tolerance for rank deficiency detection.
        """
        print("\r--- System Debug Info ---")
        print(f"LHS shape: {self.lhs.shape}")
        print(f"RHS shape: {self.rhs.shape}")
        
        # Check symmetry
        asymmetry = (self.lhs - self.lhs.T).nnz
        if asymmetry == 0: 
            print("Matrix is exactly symmetric.")
        else:
            print(f"Matrix is NOT symmetric! Nonzero entries in (A - A^T): {asymmetry}")

        # Check numerical rank
        print("Checking numerical rank (this can be slow for large systems)...")
        try:
            # Compute A*A^T and use sparse QR
            AtA = self.lhs @ self.lhs.T
            diag = AtA.diagonal()
            numerical_rank = np.sum(np.abs(diag) > tol)
            print(f"\rNumerical rank estimate: {numerical_rank} / {self.lhs.shape[0]}")
            if numerical_rank < self.lhs.shape[0]:
                warnings.warn("System matrix is numerically rank deficient. This could cause instability or ill-conditioning.")
        except Exception as e:
            print(f"\rRank check failed: {e}")
        
        # Print matrix density
        nnz = self.lhs.nnz
        total = np.prod(self.lhs.shape)
        print(f"Matrix sparsity: {100*nnz/total:.2e}% nonzero entries.")
        print("--- End Debug Info ---")

    def calculate_permeability(self, direction='y'):
        nodes = self.mesh.nodes[:, :2]
        triangles = self.mesh.triangles
        u2 = self.u2
    
        area_weighted_velocity = 0.0
    
        # Quadrature: Barycentric coordinates and weights for 7-point integration
        bary_coords = np.array([
            [1/3, 1/3, 1/3],
            [0.0597158717, 0.4701420641, 0.4701420641],
            [0.4701420641, 0.0597158717, 0.4701420641],
            [0.4701420641, 0.4701420641, 0.0597158717],
            [0.7974269853, 0.1012865073, 0.1012865073],
            [0.1012865073, 0.7974269853, 0.1012865073],
            [0.1012865073, 0.1012865073, 0.7974269853],
        ])
        weights = np.array([
            0.225,
            0.1323941527,
            0.1323941527,
            0.1323941527,
            0.1259391805,
            0.1259391805,
            0.1259391805,
        ])
    
        for tri in triangles:
            coords = nodes[tri]
            v0, v1, v2 = coords[:3]
            J = np.column_stack((v1 - v0, v2 - v0))
            triangle_area = abs(np.linalg.det(J)) / 2
            u2_local = u2[tri]
            
            triangle_velocity = 0.0
            
            # Use quadrature for evaluating average velocity of triangle.
            for bary, weight in zip(bary_coords, weights):
                xi, eta = bary[1], bary[2]
                N = np.array([P2_basis(i, xi, eta) for i in range(6)])
                u2_at_point = N @ u2_local
    
                triangle_velocity += u2_at_point * weight * triangle_area
    
            area_weighted_velocity += triangle_velocity
    
        self.avg_velocity = area_weighted_velocity / self.geometry_length**2
        b_mag = abs(self.f(0,0)[1])
        
        if self.nodim:
            self.k = (2 / 3) * (0.001**2) * abs(self.avg_velocity) / b_mag
            
        else:
            self.k = (2 / 3) * self.mu * abs(self.avg_velocity) / b_mag

        print(f"Avg y-velocity: {self.avg_velocity:.4e} m/s")
        print(f"Permeability: {self.k:.4e} m²")
        
        return self.k

def apply_dirichlet_bc(lhs, rhs, dof, value):
    lhs[dof, :] = 0.0
    lhs[:, dof] = 0.0
    lhs[dof, dof] = 1.0
    rhs[dof] = value

def canonical_dof(node, periodic_map):
    """Redirects node to its periodic master if needed."""
    return periodic_map.get(node, node)

def LoadAssembler2D(nodes, triangles, f, periodic_map):
    """
    Assembles the global load vector for a 2D finite element problem using P2 elements.

    This function loops over all elements in the mesh and Calculates the local load vector
    using a given source function, then assembles it into the global vector.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Coordinates of the mesh nodes.

    triangles : ndarray of shape (n_elements, 6)
        Connectivity array, where each row contains 6 node indices of a P2 triangle.

    f : callable
        A function f(x, y) defining the source term of the PDE.

    Returns
    -------
    b : ndarray of shape (n_nodes,)
        Global load vector.
    """

    n_nodes = nodes.shape[0]
    b1 = np.zeros(n_nodes)
    b2 = np.zeros(n_nodes)

    for tri in triangles:
        b1_local, b2_local = localLoadVector2D(nodes, tri, f)
        for i in range(6):
            node = canonical_dof(tri[i], periodic_map)
            b1[node] += b1_local[i]
            b2[node] += b2_local[i]

    return b1, b2

def localLoadVector2D(nodes, triangle, f):
    """
    Calculates the local load vector for a single P2 triangle using 3-point 
    barycentric quadrature.

    The function evaluates the integral of f(x, y) * φ_i(x, y) over the triangle using 
    quadratic (P2) basis functions and returns the result as a local load vector.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Coordinates of the mesh nodes.

    triangle : array-like of length 6
        Indices of the 6 nodes forming the P2 triangle (3 vertices + 3 edge midpoints).

    function : callable
        A function f(x, y) that returns a scalar value at point (x, y).

    Returns
    -------
    b_local : ndarray of shape (6,)
        Local load vector corresponding to the 6 basis functions of the triangle.
    """

    coords = nodes[triangle][:, :2]
    v0, v1, v2 = coords[:3]

    # Affine transform to real triangle
    J = np.column_stack((v1 - v0, v2 - v0))
    area = abs(np.linalg.det(J))/2

    # Barycentric quadrature points and weights
    bary_coords = np.array([
        [1/2, 1/2, 0],
        [0,   1/2, 1/2],
        [1/2, 0,   1/2]
    ])
    weights = np.array([1/3, 1/3, 1/3])

    b1_local = np.zeros(6)
    b2_local = np.zeros(6)

    for (L1, L2, L3), w in zip(bary_coords, weights):
        xi, eta = L2, L3
        x, y = L1 * v0 + L2 * v1 + L3 * v2  # barycentric to Cartesian
        fx, fy = f(x, y)

        for i in range(6):
            N_i = P2_basis(i, xi, eta)
            b1_local[i] += fx * N_i * w * area
            b2_local[i] += fy * N_i * w * area

    return b1_local, b2_local

def StiffnessAssembler2D(nodes, triangles, periodic_map):
    """
    Assemble the global stiffness matrix A for a 2D FEM mesh 
    using quadratic basis functions (P2 elements).

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Node coordinates.

    triangles : ndarray of shape (n_triangles, 6)
        Each row contains 6 node indices of a P2 triangle (vertices + midpoints).

    Returns
    -------
    A : scipy.sparse.lil_matrix
        Global stiffness matrix.
    """
    n_nodes = nodes.shape[0]
    A = sp.lil_matrix((n_nodes, n_nodes))

    for tri in triangles:
        A_local = localStiffnessMatrix2D(nodes, tri)
        for i in range(6):
            i_global = canonical_dof(tri[i], periodic_map)
            for j in range(6):
                j_global = canonical_dof(tri[j], periodic_map)
                A[i_global, j_global] += A_local[i, j]
    return A


@njit
def localStiffnessMatrix2D(nodes, triangle):
    """
    Compute local stiffness matrix for a P2 (quadratic) triangle using 3-point 
    Gaussian quadrature.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Node coordinates.

    triangle : array-like of shape (6,)
        Indices of the 6 nodes forming the P2 triangle.

    Returns
    -------
    A_local : ndarray of shape (6, 6)
        Local stiffness matrix.
    """
    coords = nodes[triangle][:, :2]  # Removing z-coordinates
    v0, v1, v2 = coords[:3]
    
    # Quadrature points and weights in barycentric coordinates
    bary_coords = np.array([
        [1/3, 1/3, 1/3],
        [0.0597158717, 0.4701420641, 0.4701420641],
        [0.4701420641, 0.0597158717, 0.4701420641],
        [0.4701420641, 0.4701420641, 0.0597158717],
        [0.7974269853, 0.1012865073, 0.1012865073],
        [0.1012865073, 0.7974269853, 0.1012865073],
        [0.1012865073, 0.1012865073, 0.7974269853],
    ])
    weights = np.array([
        0.225,
        0.1323941527,
        0.1323941527,
        0.1323941527,
        0.1259391805,
        0.1259391805,
        0.1259391805,
    ])

    # Build affine map from reference triangle to real triangle (vertices only)
    J = np.column_stack((v1 - v0, v2 - v0))  # Jacobian
    area = abs(np.linalg.det(J)) / 2    
    invJT = np.linalg.inv(J).T

    A_local = np.zeros((6,6))
    for (L1, L2, L3), w in zip(bary_coords, weights):
        xi, eta = L2, L3
        for i in range(6):
            gradN_i = invJT @ P2_grad(i, xi, eta)
            for j in range(6):
                gradN_j = invJT @ P2_grad(j, xi, eta)
                A_local[i,j] += w * (gradN_i @ gradN_j) * area
                
    return A_local

def MassAssembler2D(nodes, triangles, periodic_map):
    """
    Assemble the global mass matrix M for a 2D FEM mesh using P2 elements.
    
    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Node coordinates.
    triangles : ndarray of shape (n_triangles, 6)
        Each row contains 6 node indices of a P2 triangle.

    Returns
    -------
    M : scipy.sparse matrix of shape (n_nodes, n_nodes)
        Global mass matrix.
    """
    n_nodes = nodes.shape[0]
    M = sp.lil_matrix((n_nodes, n_nodes))

    for tri in triangles:
        M_local = localMassMatrix2D(nodes, tri)
        for i in range(6):
            i_global = canonical_dof(tri[i], periodic_map)
            for j in range(6):
                j_global = canonical_dof(tri[j], periodic_map)
                M[i_global, j_global] += M_local[i, j]
    return M


@njit
def localMassMatrix2D(nodes, triangle):
    """
    Compute the local mass matrix for a P2 triangle using barycentric quadrature.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Coordinates of mesh nodes.
    triangle : array-like of shape (6,)
        Indices of the 6 nodes forming the P2 triangle.

    Returns
    -------
    M_local : ndarray of shape (6, 6)
        Local mass matrix.
    """
    coords = nodes[triangle][:, :2]

    # Build affine map from reference triangle to real triangle (vertices only)
    v0, v1, v2 = coords[:3]
    J = np.column_stack((v1 - v0, v2 - v0))  # Jacobian
    area = abs(np.linalg.det(J)) / 2

    bary_coords = np.array([
        [1/3, 1/3, 1/3],
        [0.0597158717, 0.4701420641, 0.4701420641],
        [0.4701420641, 0.0597158717, 0.4701420641],
        [0.4701420641, 0.4701420641, 0.0597158717],
        [0.7974269853, 0.1012865073, 0.1012865073],
        [0.1012865073, 0.7974269853, 0.1012865073],
        [0.1012865073, 0.1012865073, 0.7974269853],
    ])
    weights = np.array([
        0.225,
        0.1323941527,
        0.1323941527,
        0.1323941527,
        0.1259391805,
        0.1259391805,
        0.1259391805,
    ])

    M_local = np.zeros((6, 6))

    for (L1, L2, L3), w in zip(bary_coords, weights):
        for i in range(6):
            xi, eta = L2, L3
            N_i = P2_basis(i, xi, eta)
            for j in range(6):
                N_j = P2_basis(j, xi, eta)
                M_local[i, j] += w * (N_i * N_j ) * area

    return M_local


def DivergenceAssembler2D(nodes, triangles, pressure_nodes, pressure_index_map, periodic_map):
    """
    Assembles the global divergence matrices B1 and B2, from P2 velocity degrees 
    of freedom and P1 pressure degrees of freedom. 
    It returns the discrete divergence matrices:
        - B1: partial derivative w.r.t x
        - B2: partial derivative w.r.t y
    
    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Node coordinates.

    triangles : ndarray of shape (n_elements, 6)
        Triangle connectivity for P2 velocity elements.
    
    pressure_nodes : ndarray of shape (n_p_nodes, 3)
           Pressure node coordinates.     

    Returns
    -------
    -B1, -B2 :  scipy.sparse.lil_matrix
                Global divergence matrices.
    """
    
    n_p = len(pressure_nodes)
    n_v = nodes.shape[0]  # velocity nodes (P2)
    B1 = sp.lil_matrix((n_p, n_v))
    B2 = sp.lil_matrix((n_p, n_v))

    for tri in triangles:
        B1_local, B2_local = localDivergenceMatrix2D(nodes, tri)
        for i in range(3):  # Only first 3 nodes of each triangle are vertex nodes (P1)
            canonical_p_node = canonical_dof(tri[i], periodic_map)
            p_idx = pressure_index_map[canonical_p_node]            
            
            for j in range(6):
                v_indx = canonical_dof(tri[j], periodic_map)
                B1[p_idx, v_indx] += B1_local[i, j]
                B2[p_idx, v_indx] += B2_local[i, j]

    return -B1, -B2 # Minus sign from defintion of our divergence matrix

@njit
def localDivergenceMatrix2D(nodes, triangle):
    """
      Calculates the local divergence matrices for a single P2 triangle element.
    
      Returns the contribution of this element to the global divergence operators 
      B1 and B2 using 3-point barycentric quadrature.
    
      Parameters
      ----------
      nodes : ndarray of shape (n_nodes, 3)
          Coordinates of mesh nodes.
    
      triangle : array-like of length 6
          Indices of the 6 nodes defining a P2 triangle.
    
      Returns
      -------
      B1_local : ndarray of shape (3, 6)
          Local matrix of ∂φ_j/∂x tested against pressure basis functions.
    
      B2_local : ndarray of shape (3, 6)
          Local matrix of ∂φ_j/∂y tested against pressure basis functions.
      """
    coords = nodes[triangle][:, :2]  # shape: (6, 2)
    v0, v1, v2 = coords[:3]

    # Build Jacobian for affine map
    J = np.column_stack((v1 - v0, v2 - v0))
    area = abs(np.linalg.det(J)) / 2
    invJT = np.linalg.inv(J).T

    # Quadrature points (same as in stiffness matrix)
    bary_coords = np.array([
        [1/3, 1/3, 1/3],
        [0.0597158717, 0.4701420641, 0.4701420641],
        [0.4701420641, 0.0597158717, 0.4701420641],
        [0.4701420641, 0.4701420641, 0.0597158717],
        [0.7974269853, 0.1012865073, 0.1012865073],
        [0.1012865073, 0.7974269853, 0.1012865073],
        [0.1012865073, 0.1012865073, 0.7974269853],
    ])
    weights = np.array([
        0.225,
        0.1323941527,
        0.1323941527,
        0.1323941527,
        0.1259391805,
        0.1259391805,
        0.1259391805,
    ])
    
    B1_local = np.zeros((3, 6))
    B2_local = np.zeros((3, 6))

    for (L1, L2, L3), w in zip(bary_coords, weights):
        xi, eta = L2, L3
        for i in range(3):
            M_i = P1_basis(i, xi, eta)
            for j in range(6):
                gradN_j = invJT @ P2_grad(j, xi, eta)
                B1_local[i, j] += M_i * gradN_j[0] * w * area
                B2_local[i, j] += M_i * gradN_j[1] * w * area

    return B1_local, B2_local 

def plot_velocity_magnitude(mesh, u1, u2, title_suffix="", nodim=False):
    magnitude = np.sqrt(u1**2 + u2**2)
    plt.figure(figsize=(6, 5), dpi=300)
    triangles_P1 = mesh.triangles[:, :3]
    
    
    if mesh.triangles.shape[0] > 10000:
        edgecolor = 'none'
    else:
        edgecolor = 'k'
        
    if nodim:
        nodesx = mesh.nodes[:, 0]
        nodesy = mesh.nodes[:, 1]
    else:
        nodesx = mesh.nodes[:, 0] *1000
        nodesy = mesh.nodes[:, 1]*1000
        
    t = mtri.Triangulation(nodesx, nodesy, triangles_P1) # Scaling from m to mm
    plt.tripcolor(t, magnitude, shading='flat', cmap='viridis', edgecolors=edgecolor, linewidth=0.1)
    
    if nodim:
        plt.xlabel("$y_1$")
        plt.ylabel("$y_2$")
        plt.colorbar(label=r"$|\vec w|$ ")
        plt.title(f"Velocity response \n{title_suffix}\n", fontsize=10)

    else:
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.colorbar(label=r"$|u|$ [m/s]")
        plt.title(f"Velocity Magnitude [m/s]\n{title_suffix}\n ", fontsize=10)
    
    plt.tight_layout()

    # Save figure
    safe_desc = title_suffix.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    plt.savefig(f"plots/velocity_magnitude_{safe_desc}.png", dpi=300)
    plt.show()
    plt.close()
    
    # --- NEW: Print some useful statistics ---
    print(f"\rMax node velocity magnitude: {np.max(magnitude):.3e} m/s")
    print(f"Min node velocity magnitude: {np.min(magnitude):.3e} m/s")

    return magnitude
    
def plot_pressure(mesh, pressure, title_suffix="", nodim=False):
    plt.figure(figsize=(6, 5), dpi=300)
    vertex_coords = mesh.nodes[mesh.pressure_nodes] 
    old_to_new = {old: new for new, old in enumerate(mesh.pressure_nodes)}
    triangles_p1 = np.array([
        [old_to_new[i] for i in tri[:3]]
        for tri in mesh.triangles
        if all(i in old_to_new for i in tri[:3])
    ])
    
    if mesh.triangles.shape[0] > 10000:
        edgecolor = 'none'
    else:
        edgecolor = 'k'
        
    if nodim:
        coordsx = vertex_coords[:, 0]
        coordsy = vertex_coords[:, 1]
    else:
        coordsx = vertex_coords[:, 0]*1000
        coordsy = vertex_coords[:, 1]*1000
        
    t = mtri.Triangulation(coordsx, coordsy, triangles_p1) # Scaling from m to mm
    plt.tripcolor(t, pressure, shading='gouraud', cmap='coolwarm', edgecolors=edgecolor)
        
    if nodim:
        plt.xlabel("$y_1$")
        plt.ylabel("$y_2$")
        plt.colorbar(label=r"$p$")
        plt.title(f"Pressure response\n{title_suffix}\n", fontsize=10)  

    else:
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.colorbar(label=r"$p$ [Pa]")
        plt.title(f"Pressure Field [Pa]\n{title_suffix}\n ", fontsize=10)
    
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    safe_desc = title_suffix.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    plt.savefig(f"plots/pressure_field_{safe_desc}.png", dpi=300)
    plt.show()
    plt.close()

    print(f"Max pressure: {np.max(pressure):.4e} Pa")
    print(f"Min pressure: {np.min(pressure):.4e} Pa")
    return

def plot_velocity_vectors(mesh, u1, u2, title_suffix="", nodim=False):
    """
    Plot velocity vectors at uniformly spaced points.

    """
    
    if nodim:
        x = mesh.nodes[:, 0]
        y = mesh.nodes[:, 1]
    else:
        x = mesh.nodes[:, 0] * 1000  # mm
        y = mesh.nodes[:, 1] * 1000
        
    u = u1
    v = u2
    magnitude = np.sqrt(u**2 + v**2)

    # Build spatial index
    kdtree = cKDTree(np.column_stack([x, y]))

    # Create uniform grid
    nx, ny = 30, 30
    x_grid = np.linspace(np.min(x), np.max(x), nx)
    y_grid = np.linspace(np.min(y), np.max(y), ny)
    xv, yv = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])

    # Find closest mesh node and distance
    distances, idx = kdtree.query(grid_points)

    # Filter valid indices
    idx = np.unique(idx)

    # Normalize velocity direction
    u_dir = np.zeros_like(u)
    v_dir = np.zeros_like(v)
    nonzero = magnitude > 1e-14
    u_dir[nonzero] = u[nonzero] / magnitude[nonzero]
    v_dir[nonzero] = v[nonzero] / magnitude[nonzero]

    # Plot
    plt.figure(figsize=(6, 5), dpi=300)
    quiv = plt.quiver(
        x[idx], y[idx],
        u_dir[idx], v_dir[idx],
        magnitude[idx],
        angles='xy',
        scale_units='xy',
        cmap='viridis'
    )

    if nodim:
        plt.colorbar(quiv, label='Magnitude')
        plt.xlabel("$y_1$")
        plt.ylabel("$y_2$")
        plt.title(f"Velocity Direction\n{title_suffix}\n", fontsize=10)
    else:
        plt.colorbar(quiv, label='Velocity Magnitude [m/s]')
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title(f"Velocity Direction (colored by magnitude)\n{title_suffix}\n", fontsize=10)
    
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    safe_desc = title_suffix.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    plt.savefig(f"plots/velocity_vectors_colored_{safe_desc}.png", dpi=300)
    plt.show()
    plt.close()
    
def load_simulation(filepath):
    """
    Load a Simulation object from a pickle file.

    Parameters
    ----------
    filename : str
        Path to the pickle file (.pkl) containing the saved Simulation.

    Returns
    -------
    sim : Simulation
        Loaded Simulation object.
    """
    #filepath = f'simulations/{filename}'
    
    with gzip.open(filepath, "rb") as f:
        sim = pickle.load(f)
    print(f"Loaded simulation from '{filepath}'")
    return sim
    
@njit
def P1_basis(i, xi, eta):
    if i == 0:
        return (1 - xi - eta)
    elif i == 1:
        return xi
    elif i == 2:
        return eta
    else:
        return 0
    
@njit
def P2_basis(i, xi, eta):
    if i == 0:
        return (1 - xi - eta) * (1 - 2*xi - 2*eta)
    elif i == 1:
        return xi * (2 * xi - 1)
    elif i == 2:
        return eta * (2 * eta - 1)
    elif i == 3:
        return 4 * xi * (1 - xi - eta)
    elif i == 4:
        return 4 * xi * eta
    elif i == 5:
        return 4 * eta * (1 - xi - eta)
    else:
        return 0

@njit
def P2_grad(i, xi, eta):
    if i == 0:
        return np.array([4 * xi + 4 * eta - 3, 4 * xi + 4 * eta - 3])
    elif i == 1:
        return np.array([4 * xi - 1, 0.0])
    elif i == 2:
        return np.array([0.0, 4 * eta - 1])
    elif i == 3:
        return np.array([4 - 8 * xi - 4 * eta, -4 * xi])
    elif i == 4:
        return np.array([4 * eta, 4 * xi])
    elif i == 5:
        return np.array([-4 * eta, 4 - 4 * xi - 8 * eta])
    else:
        return np.array([0.0, 0.0])    
    
if __name__ == "__main__":
    # Create folders if it does not exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("simulations", exist_ok=True)
    
    periodic = False
    nodim = True
    
    # Geometry parameters
    if nodim:
        geometry_length = 1 # meters 0.001 is 1mm
        geometry_height = 0.091 # 91 micrometer thickness
        mu = 1#0.00089 # Viscosity Pa*s
        rho = 1#1000.0 # Density kg/m^3
        g = 1 # m/s^2
        
    else:
        geometry_length = 0.001 # meters 0.001 is 1mm
        geometry_height = 0.000091 # 91 micrometer thickness
        mu = 0.00089 # Viscosity Pa*s
        rho = 1000.0 # Density kg/m^3
        g = 9.81 # m/s^2
        
    # Mesh parameters
    mesh_size = geometry_length * 0.02 # meters
    inner_radius = geometry_length * 0.40
    
    # Body forces
    def f(x, y):
        return (0, -rho*g) # Body force wtih units N/m^3

    # Create mesh
    mesh_start = datetime.now()
    #create_mesh(3,4,geometry_length, mesh_size, inner_radius, "square_with_hole.msh", periodic=periodic)
    create_mesh(geometry_length, mesh_size, inner_radius, "square_with_hole.msh", periodic=periodic)
    mesh_time = datetime.now()
    print(f"Created mesh in in {(mesh_time - mesh_start).total_seconds():.3f} seconds.")

    raw_mesh = meshio.read("square_with_hole.msh")
    mesh = load_mesh(raw_mesh, periodic)
    mesh.mesh_size = mesh_size  # manually attach it
    load_time = datetime.now()
    print(f"Mesh loaded: {mesh.triangles.shape[0]} P2 elements, {mesh.nodes.shape[0]} P2 nodes.")
    print(f"Mesh loading and processing time: {(load_time - mesh_time).total_seconds():.3f} seconds.")

    mesh.check_mesh_quality()
    check_time = datetime.now()
    print(f"Checked mesh in {(check_time - load_time).total_seconds():.3f} seconds.")

    # Setup and run simulation
    sim = Simulation(mesh, f, geometry_length, inner_radius, geometry_height, mu=mu, rho=rho, periodic_bc=periodic, nodim=nodim) #"periodic"
    print(f"Simulation parameters: {sim.get_description()}")
    sim.run()
    
    start_calc = datetime.now()
    _ = sim.calculate_permeability(direction='y')
    print(f"Calculated permeability in {(datetime.now()-start_calc).total_seconds():.3f} seconds.")

    # Save the entire Simulation object
    desc = sim.get_description()
    safe_desc = desc.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    sim.save(f"simulations/sim_{safe_desc}.pkl")
