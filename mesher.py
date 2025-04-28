# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:23:45 2025

@author: skje
"""

import gmsh 
import meshio
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime

class Mesh:
    """
    Container for a 2D triangular finite element mesh with boundary classification.

    Attributes
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Coordinates of the mesh nodes.

    triangles : ndarray of shape (n_elements, 6)
        P2 element connectivity (6 node indices per triangle).

    interior_edges : list of tuple[int, int]
        Edges shared by two elements (internal to the mesh).

    boundary_edges : list of tuple[int, int]
        Boundary edges excluding inlet and outlet.

    inlet_edges : list of tuple[int, int]
        Boundary edges on the left boundary (x ≈ 0.0).

    outlet_edges : list of tuple[int, int]
        Boundary edges on the right boundary (x ≈ 1.0).

    pressure_nodes : ndarray of int
        Indices of pressure DOFs (P1 vertex nodes).

    pressure_index_map : dict
        Maps global node index to pressure DOF index.
    """

    def __init__(self, nodes, triangles, interior_edges, boundary_edges, inlet_edges, 
                 outlet_edges, pressure_nodes, pressure_index_map):
        self.nodes = nodes
        self.triangles = triangles
        self.interior_edges = interior_edges
        self.boundary_edges = boundary_edges
        self.inlet_edges = inlet_edges
        self.outlet_edges = outlet_edges
        self.pressure_nodes = pressure_nodes
        self.pressure_index_map = pressure_index_map

        
    def plot(self, show_node_ids=True, filename="mesh_plot.png"):
        """
        Plot the mesh with color-coded edge types and node labels.

        Parameters
        ----------
        show_node_ids : bool, optional
            Whether to annotate nodes with their indices. Default is True.

        filename : str, optional
            Filename to save the plot. Default is "mesh_plot.png".

        Returns
        -------
        None
        """
        print("Plotting mesh...", end="", flush=True)

        plt.figure(figsize=(6, 6))

        if show_node_ids:
            for idx, (xi, yi) in enumerate(self.nodes[:, :2]):
                plt.scatter(xi, yi, color='C4', s=5, zorder=5)
                plt.text(xi+0.01, yi+0.01, str(idx), fontsize=6,
                         color='black', ha='center', va='center')

        def draw_edges(edges, color, label):
            for edge in edges:
                p1, p2 = self.nodes[edge[0]], self.nodes[edge[1]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color, linewidth=1.2, label=label)

        draw_edges(self.interior_edges, 'C0', 'Interior')
        draw_edges(self.boundary_edges, 'C1', 'Boundary')
        draw_edges(self.inlet_edges, 'C2', 'Inlet')
        draw_edges(self.outlet_edges, 'C3', 'Outlet')

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center')

        plt.gca().set_aspect("equal")
        plt.title("M*E*S*H*")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        
        print(f"Mesh plot saved as {filename}.")
        return

def create_mesh(geometry_length=0.01, mesh_size=0.0001, output_file="square_with_hole.msh"):
    """
    Generates a 2D unstructured P2 triangular mesh of a square domain with a circular hole.

    The domain is a unit square [0,l] x [0,l] with a circular hole of radius l/3 centered at (l/2, l/2).
    The mesh is generated using Gmsh and saved to a `.msh` file.

    Parameters
    ----------
    mesh_size : float
        Target element size for the mesh (controls mesh resolution).

    output_file: str
        Name of file to save mesh to.
        
    Returns
    -------
    None
    """

    gmsh.initialize()
    gmsh.model.add(output_file.rstrip(".msh"))
    
    l = geometry_length

    # Outer square
    s_p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, mesh_size)
    s_p2 = gmsh.model.occ.addPoint(l, 0.0, 0.0, mesh_size)
    s_p3 = gmsh.model.occ.addPoint(l, l, 0.0, mesh_size)
    s_p4 = gmsh.model.occ.addPoint(0.0, l, 0.0, mesh_size)

    l1 = gmsh.model.occ.addLine(s_p1, s_p2)  # bottom
    l2 = gmsh.model.occ.addLine(s_p2, s_p3)  # right
    l3 = gmsh.model.occ.addLine(s_p3, s_p4)  # top
    l4 = gmsh.model.occ.addLine(s_p4, s_p1)  # left

    square_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    square_surface = gmsh.model.occ.addPlaneSurface([square_loop])

    # Circle hole
    r = l/3
    center = gmsh.model.occ.addPoint(l/2, l/2, 0.0, mesh_size)
    c_p1 = gmsh.model.occ.addPoint(l/2 + r, l/2, 0.0, mesh_size)
    c_p2 = gmsh.model.occ.addPoint(l/2, l/2 + r, 0.0, mesh_size)
    c_p3 = gmsh.model.occ.addPoint(l/2 - r, l/2, 0.0, mesh_size)
    c_p4 = gmsh.model.occ.addPoint(l/2, l/2 - r, 0.0, mesh_size)

    arc1 = gmsh.model.occ.addCircleArc(c_p1, center, c_p2)
    arc2 = gmsh.model.occ.addCircleArc(c_p2, center, c_p3)
    arc3 = gmsh.model.occ.addCircleArc(c_p3, center, c_p4)
    arc4 = gmsh.model.occ.addCircleArc(c_p4, center, c_p1)

    circle_loop = gmsh.model.occ.addCurveLoop([arc1, arc2, arc3, arc4])
    circle_surface = gmsh.model.occ.addPlaneSurface([circle_loop])

    # Subtract circle from square
    [cut_surface], _ = gmsh.model.occ.cut([(2, square_surface)], [(2, circle_surface)])

    gmsh.model.occ.synchronize()
    
    # Generate P2 mesh!
    gmsh.model.mesh.setOrder(2)  # Make sure this is BEFORE generate()
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # optional, reinforces the intent
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)  # use complete P2 elements
    gmsh.model.mesh.generate(2)

    gmsh.write(output_file)
    gmsh.finalize()

    return


def load_mesh(mesh=None):
    """
    Load a mesh from file or memory and classify its edges.

    Identifies inlet/outlet/boundary edges and builds pressure DOF information.
    If no mesh is passed, reads from 'square_with_hole.msh'.

    Parameters
    ----------
    mesh : meshio.Mesh, optional
        A meshio mesh object. If None, a file is read from disk.

    Returns
    -------
    Mesh
        A structured mesh object with node/triangle data and classified edges.
    """
    
    if mesh is None:
        mesh = meshio.read("square_with_hole.msh")

    if "triangle6" in mesh.cells_dict:
        triangles = mesh.cells_dict["triangle6"]
    elif "triangle" in mesh.cells_dict:
        triangles = mesh.cells_dict["triangle"]
        print("Warning: using linear triangles (P1), not P2")
    else:
        raise ValueError("No triangle or triangle6 elements found in mesh.")

    nodes = mesh.points

    if triangles is None:
        print("No triangle cells found in mesh.")
        return
    
    # --- Filter out unused nodes ---
    used_node_indices = np.unique(triangles.flatten())
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_node_indices)}
    
    nodes = nodes[used_node_indices]
    triangles = np.array([[index_map[i] for i in tri] for tri in triangles])


    # Classify edges
    edge_list = defaultdict(int)
    for tri in triangles:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for edge in edges:
            edge_list[tuple(sorted(edge))] += 1

    interior_edges = []
    boundary_edges = []
    inlet_edges = []
    outlet_edges = []

    for edge, count in edge_list.items():
        if count == 1:  # boundary edge
            p1, p2 = nodes[edge[0]], nodes[edge[1]]
            x1, x2 = p1[0], p2[0]

            if np.isclose(x1, 0.0) and np.isclose(x2, 0.0):
                inlet_edges.append(edge)
            elif np.isclose(x1, 1.0) and np.isclose(x2, 1.0):
                outlet_edges.append(edge)
            else:
                boundary_edges.append(edge)
        elif count == 2:
            interior_edges.append(edge)  
        else:
            print(f"ERROR! Edge appears f{count} times.")
            
    # Pressure node info (P1 nodes only)
    pressure_nodes = triangles[:, :3].flatten()
    pressure_nodes = np.unique(pressure_nodes)
    pressure_index_map = {node: i for i, node in enumerate(pressure_nodes)}

    # Initialize mesh class
    mesh = Mesh(
        nodes=nodes,
        triangles=triangles,
        interior_edges=interior_edges,
        boundary_edges=boundary_edges,
        inlet_edges=inlet_edges,
        outlet_edges=outlet_edges,
        pressure_nodes=pressure_nodes,
        pressure_index_map=pressure_index_map
    )

    return mesh

if __name__ == "__main__":
    """
    When run directly, this script will:
    1. Generate a mesh using Gmsh.
    2. Load and classify mesh edges and pressure nodes.
    3. Plot the mesh with labels and edge classifications.
    
    """
    mesh_file = "square_with_hole.msh"
    mesh_size = 0.1
    
    total_start = datetime.now()

    print("Creating mesh...", end="", flush=True)
    create_mesh(mesh_size, mesh_file)

    print("Loading mesh...", end="", flush=True)
    raw_mesh = meshio.read(mesh_file)
    mesh = load_mesh(raw_mesh)

    print("Plotting mesh...", end="", flush=True)
    mesh.plot(show_node_ids=False, filename="mesh.png")

    end_mesh = datetime.now()
    print(f"\rMeshed in {(end_mesh - total_start).total_seconds():.3f} seconds")