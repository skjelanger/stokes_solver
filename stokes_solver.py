# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 07:15:06 2025

@author: skje
"""

import pickle
import os
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mesher import create_mesh, load_mesh
from datetime import datetime


class Simulation:
    def __init__(self, mesh, f, velocity_inlet_profile, mu=0.001, rho=1000.0, outlet_pressure=100000):
        self.mesh = mesh
        self.f = f
        self.mu = mu
        self.rho = rho
        
        self.velocity_inlet_profile = velocity_inlet_profile
        self.outlet_pressure = outlet_pressure

        self.u1 = None
        self.u2 = None
        self.p = None
        self.lhs = None
        self.rhs = None
        
        
    def get_description(self):
        desc = f"mu={self.mu:.2e}, rho={self.rho:.1f}"
        if self.mesh.mesh_size is not None:
            desc += f", h={self.mesh.mesh_size:.3f}"
        return desc
        
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


    def assemble(self):
        nodes = self.mesh.nodes
        triangles = self.mesh.triangles
        pressure_nodes = self.mesh.pressure_nodes
        periodic_map = self.mesh.periodic_map

        A11 = (self.mu / self.rho) * StiffnessAssembler2D(nodes, triangles, periodic_map)
        B1, B2 = DivergenceAssembler2D(nodes, triangles, pressure_nodes, periodic_map)
        b1, b2 = LoadAssembler2D(nodes, triangles, self.f, periodic_map)


        B1T = (1 / self.rho) * B1.T
        B2T = (1 / self.rho) * B2.T

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

        self.apply_boundary_conditions()
        
        slave_nodes = set(periodic_map.keys())
        n = self.mesh.nodes.shape[0]
        
        
        # Remove DOFs for the slave ndoes.
        for node in slave_nodes:
            for offset in [0, n]:  # Handle both u1 and u2 velocities
                dof = node + offset
                self.lhs[dof, :] = 0
                self.lhs[:, dof] = 0 
                self.lhs[dof, dof] = 1 
                self.rhs[dof] = 0
                
        self.lhs_scr = self.lhs.tocsr()
        
        
    def solve(self):
        sol = spla.spsolve(self.lhs_scr, self.rhs) 
        n = self.mesh.nodes.shape[0]
    
        self.u1 = sol[:n]
        self.u2 = sol[n:2*n]
        self.p = sol[2*n:]
        
        
        # Copies the correct velocity from the master to the slave
        for slave, master in self.mesh.periodic_map.items():
            self.u1[slave] = self.u1[master]
            self.u2[slave] = self.u2[master]

    def run(self):
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
        print(f"\rSolved system in {(end_solver - total_start).total_seconds():.3f} seconds.")
    
        print("Plotting results...", end="", flush=True)
        self.plot()
        end_plot = datetime.now()
        print(f"\rPlotted results in {(end_plot - end_solver).total_seconds():.3f} seconds.")
    
        total_end = datetime.now()
        print(f"Total runtime: {(total_end - total_start).total_seconds():.3f} seconds.")

        
    def plot(self):
        desc = self.get_description()
        plot_velocity_magnitude(self.mesh, self.u1, self.u2, title_suffix=desc)
        plot_pressure(self.mesh, self.p, title_suffix=desc)
        
        
    def apply_boundary_conditions(self):
        """
        Modifies the system matrix (lhs) and RHS vector (rhs) to apply:
        - No-slip condition on walls (v = 0)
        - Pressure at inlet
        - Pressure at outlet
        """
        n = self.mesh.nodes.shape[0]
        m = len(self.mesh.pressure_nodes)
        lhs = self.lhs
        rhs = self.rhs
        periodic_slaves = set(self.mesh.periodic_map.keys())

        # --- No-slip walls on interior boundary (velocity = 0)
        for edge in mesh.interior_boundary_edges:
            for node in edge:
                if node in periodic_slaves:
                    continue  # skip: handled by master node
                # x-velocity
                lhs[node, :] = 0
                lhs[:, node] = 0
                lhs[node, node] = 1
                rhs[node] = 0
    
                # y-velocity
                node_y = node + n
                lhs[node_y, :] = 0
                lhs[:, node_y] = 0
                lhs[node_y, node_y] = 1
                rhs[node_y] = 0
    

        # --- Fix pressure at one arbitrary node (to remove null space)
        if len(mesh.pressure_nodes) > 0:
            for node in mesh.pressure_nodes:
                if node not in periodic_slaves:
                    anchor_node = node
                    break
            anchor_dof = 2 * n + mesh.pressure_index_map[anchor_node]
            lhs[anchor_dof, :] = 0
            lhs[:, anchor_dof] = 0
            lhs[anchor_dof, anchor_dof] = 1
            rhs[anchor_dof] = 0

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
        print("\n--- System Debug Info ---")
        print(f"LHS shape: {self.lhs_scr.shape}")
        print(f"RHS shape: {self.rhs.shape}")
        
        # Check symmetry
        asymmetry = (self.lhs_scr - self.lhs_scr.T).nnz
        if asymmetry == 0: 
            print("Matrix is exactly symmetric.")
        else:
            print(f"Matrix is NOT symmetric! Nonzero entries in (A - A^T): {asymmetry}")

        # Check numerical rank
        print("Checking numerical rank (this can be slow for large systems)...")
        try:
            # Compute A*A^T and use sparse QR
            AtA = self.lhs_scr @ self.lhs_scr.T
            diag = AtA.diagonal()
            numerical_rank = np.sum(np.abs(diag) > tol)
            print(f"Numerical rank estimate: {numerical_rank} / {self.lhs_scr.shape[0]}")
        except Exception as e:
            print(f"Rank check failed: {e}")
        
        # Print matrix density
        nnz = self.lhs_scr.nnz
        total = np.prod(self.lhs_scr.shape)
        print(f"Matrix sparsity: {100*nnz/total:.2f}% nonzero entries.")
        print("--- End Debug Info ---\n")


def canonical_dof(node, periodic_map):
    """Redirects node to its periodic master if needed."""
    return periodic_map.get(node, node)


def LoadAssembler2D(nodes, triangles, f, periodic_map):
    """
    Assembles the global load vector for a 2D finite element problem using P2 elements.

    This function loops over all elements in the mesh and computes the local load vector
    using a given source function, then assembles it into the global vector.

    Parameters
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Coordinates of the mesh nodes.

    triangles : ndarray of shape (n_elements, 6)
        Connectivity array, where each row contains 6 node indices of a P2 triangle.

    function : callable
        A function f(x, y) defining the source term of the PDE.

    Returns
    -------
    b : ndarray of shape (n_nodes,)
        Global load vector.
    """

    n_nodes = nodes.shape[0]
    b1 = np.zeros(n_nodes)
    b2 = np.zeros(n_nodes)

    for triangle in triangles:
        b1_local, b2_local = localLoadVector2D(nodes, triangle, f)
        for i in range(6):
            node = canonical_dof(triangle[i], periodic_map)
            b1[node] += b1_local[i]
            b2[node] += b2_local[i]

    return b1, b2

def localLoadVector2D(nodes, triangle, f):
    """
    Computes the local load vector for a single P2 triangle using 3-point 
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
    detJ = abs(np.linalg.det(J))

    # Barycentric quadrature points and weights
    bary_coords = np.array([
        [1/6, 1/6, 2/3],
        [1/6, 2/3, 1/6],
        [2/3, 1/6, 1/6]
    ])
    weights = np.array([1/3, 1/3, 1/3])

    b1_local = np.zeros(6)
    b2_local = np.zeros(6)

    for q, w in zip(bary_coords, weights):
        xi, eta = q[0], q[1]
        x, y = v0 + xi * (v1 - v0) + eta * (v2 - v0)
        fx, fy = f(x, y)
        for i in range(6):
            phi = P2Basis.basis(i, xi, eta)
            b1_local[i] += fx * phi * w * detJ / 2
            b2_local[i] += fy * phi * w * detJ / 2

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
    A : ndarray of shape (n_nodes, n_nodes)
        Global stiffness matrix.
    """
    n_nodes = nodes.shape[0]
    A = sp.lil_matrix((n_nodes, n_nodes))

    for triangle in triangles:
        A_local = localStiffnessMatrix2D(nodes, triangle)
        for i in range(6):
            for j in range(6):
                i_global = canonical_dof(triangle[i], periodic_map)
                j_global = canonical_dof(triangle[j], periodic_map)
                A[i_global, j_global] += A_local[i, j]
    return A

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
    coords = nodes[triangle][:, :2]  # Removing z-coordinates - shape (6, 2)

    # Quadrature points and weights in barycentric coordinates
    bary_coords = np.array([
        [1/6, 1/6, 2/3],
        [1/6, 2/3, 1/6],
        [2/3, 1/6, 1/6]
    ])
    weights = np.array([1/3, 1/3, 1/3])

    # Build affine map from reference triangle to real triangle (vertices only)
    v0, v1, v2 = coords[:3]
    J = np.column_stack((v1 - v0, v2 - v0))  # Jacobian
    detJ = abs(np.linalg.det(J))
    invJT = np.linalg.inv(J).T
    G = detJ * (invJT @ invJT.T)

    A_local = np.zeros((6, 6))

    for q, w in zip(bary_coords, weights):
        xi, eta = q[0], q[1]
        for i in range(6):
            for j in range(6):
                A_local[i, j] += w * (P2Basis.grad(i, xi, eta) @ G @ P2Basis.grad(j, xi, eta)) / 2

    return A_local

def DivergenceAssembler2D(nodes, triangles, pressure_nodes, periodic_map):
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

    triangle : array-like of shape (6,)
        Indices of the 6 nodes forming the P2 triangle.
    
    pressure_nodes : ndarray of shape (n_p_nodes, 3)
           Pressure node coordinates.     

    Returns
    -------
    B1, B2
    """
    
    pressure_index_map = {node: i for i, node in enumerate(pressure_nodes)}

    n_p = len(pressure_nodes)
    n_v = nodes.shape[0]  # velocity nodes (P2)
    B1 = sp.lil_matrix((n_p, n_v))
    B2 = sp.lil_matrix((n_p, n_v))

    for tri in triangles:
        B1_local, B2_local = localDivergenceMatrix2D(nodes, tri)
        for i in range(3):  # Only first 3 nodes of each triangle are vertex nodes (P1)
            p_node = tri[i]
            if p_node in pressure_index_map:
                p_idx = pressure_index_map[p_node]
                for j in range(6):
                    v_node = canonical_dof(tri[j], periodic_map)
                    B1[p_idx, v_node] += B1_local[i, j]
                    B2[p_idx, v_node] += B2_local[i, j]

    return B1, B2

def localDivergenceMatrix2D(nodes, triangle):
    """
      Computes the local divergence matrices for a single P2 triangle element.
    
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
    detJ = abs(np.linalg.det(J))
    invJT = np.linalg.inv(J).T

    # Quadrature points (same as in stiffness matrix)
    bary_coords = np.array([
        [1/6, 1/6, 2/3],
        [1/6, 2/3, 1/6],
        [2/3, 1/6, 1/6]
    ])
    weights = np.array([1/3, 1/3, 1/3])

    B1_local = np.zeros((3, 6))
    B2_local = np.zeros((3, 6))

    for q, w in zip(bary_coords, weights):
        xi, eta = q[0], q[1]
        for i in range(3):
            for j in range(6):
                grad_ref = P2Basis.grad(j, xi, eta)
                grad_xy = invJT @ grad_ref
                B1_local[i, j] += w * grad_xy[0] * detJ / 2
                B2_local[i, j] += w * grad_xy[1] * detJ / 2

    return -B1_local, -B2_local

def plot_velocity_magnitude(mesh, u1, u2, title_suffix=""):
    magnitude = np.sqrt(u1**2 + u2**2)
    plt.figure(figsize=(6, 5))
    triangles_P1 = mesh.triangles[:, :3]
    t = mtri.Triangulation(mesh.nodes[:, 0] *1000, mesh.nodes[:, 1]*1000, triangles_P1) # Scaling from m to mm
    plt.tripcolor(t, magnitude, shading='flat', cmap='viridis', edgecolors='k', linewidth=0.1)
    
    plt.title(f"Velocity Magnitude [m/s]\n{title_suffix}")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.colorbar(label=r"$|u|$ [m/s]")
    plt.tight_layout()

    # Save figure
    safe_desc = title_suffix.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    plt.savefig(f"plots/velocity_magnitude_{safe_desc}.png", dpi=300)
    plt.show()
    plt.close()
    
    # --- NEW: Print some useful statistics ---
    print(f"Max velocity magnitude: {np.max(magnitude):.4f} m/s")
    print(f"Min velocity magnitude: {np.min(magnitude):.4f} m/s")

    return magnitude
    
def plot_pressure(mesh, pressure, title_suffix=""):
    plt.figure(figsize=(6, 5))
    vertex_coords = mesh.nodes[mesh.pressure_nodes] 
    old_to_new = {old: new for new, old in enumerate(mesh.pressure_nodes)}
    triangles_p1 = np.array([
        [old_to_new[i] for i in tri[:3]]
        for tri in mesh.triangles
        if all(i in old_to_new for i in tri[:3])
    ])
    t = mtri.Triangulation(vertex_coords[:, 0]*1000, vertex_coords[:, 1]*1000, triangles_p1) # Scaling from m to mm
    plt.tripcolor(t, pressure, shading='gouraud', cmap='coolwarm', edgecolors='k')
    
    plt.title(f"Pressure Field (P1) [Pa]\n{title_suffix}")
    plt.colorbar(label=r"$p$ [Pa]")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    safe_desc = title_suffix.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    plt.savefig(f"plots/pressure_field_{safe_desc}.png", dpi=300)
    plt.show()
    plt.close()

    # --- NEW: Print some useful statistics ---
    print(f"Max pressure: {np.max(pressure):.4f} Pa")
    print(f"Min pressure: {np.min(pressure):.4f} Pa")
   
    
def load_simulation(filename):
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
    filepath = f'simulations/{filename}'
    
    with open(filepath, "rb") as f:
        sim = pickle.load(f)
    print(f"Loaded simulation from '{filename}'")
    return sim
    
    
class P2Basis:
    @staticmethod
    def basis(i, xi, eta):
        basis = [
            (1 - xi - eta) * (1 - 2*xi - 2*eta),
            xi * (2*xi - 1),
            eta * (2*eta - 1),
            4 * xi * (1 - xi - eta),
            4 * xi * eta,
            4 * eta * (1 - xi - eta)
        ]
        return basis[i]
    
    @staticmethod
    def grad(i, xi, eta):
        grads = [
            [(4*xi + 4*eta - 3), (4*xi + 4*eta - 3)],
            [4*xi - 1, 0],
            [0, 4*eta - 1],
            [(4 - 8*xi - 4*eta), (-4*xi)],
            [4*eta, 4*xi],
            [(-4*eta), (4 - 4*xi - 8*eta)]
        ]
        return np.array(grads[i])

    
if __name__ == "__main__":
    # Create folders if it does nto exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("simulations", exist_ok=True)
    
    # Define constants
    geometry_length = 0.01 # meters
    mesh_size = geometry_length * 0.1 # meters
    outlet_pressure = 0 #Pa
    mu = 0.001 # Viscosity Pa*s
    rho = 1000.0 # Density kg/m^3
    
    # Body forces
    def f(x, y): return (1, -1)
    
    def velocity_inlet_profile(x, y):
        height = geometry_length  # match geometry height (in meters)
        u = 0.2 * 4 * (y / height) * (1 - y / height)  # max velocity at center
        v = 0.0
        return u, v

    # Create mesh
    create_mesh(geometry_length, mesh_size, "square_with_hole.msh")
    raw_mesh = meshio.read("square_with_hole.msh")
    mesh = load_mesh(raw_mesh)
    mesh.mesh_size = mesh_size  # manually attach it

    # Setup and run simulation
    sim = Simulation(mesh, f, velocity_inlet_profile,
                     mu=mu, rho=rho, outlet_pressure=outlet_pressure)
    sim.run()
    
    # Save the entire Simulation object
    desc = sim.get_description()
    safe_desc = desc.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
    sim.save(f"simulations/sim_{safe_desc}.pkl")
