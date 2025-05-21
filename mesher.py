# -*- coding: utf-8 -*-
# mesher.py
"""
Created on Fri Apr 25 11:23:45 2025

@author: skje
"""

import warnings
import gmsh 
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

from collections import defaultdict
from datetime import datetime
from scipy.spatial import cKDTree


class Mesh:
    """
    Container for a 2D triangular finite element mesh using P2 elements,
    with classified boundaries and support for periodic mappings.

    Attributes
    ----------
    nodes : ndarray of shape (n_nodes, 3)
        Coordinates of the mesh nodes (x, y, z), where z is typically 0.

    triangles : ndarray of shape (n_elements, 6)
        Connectivity for each P2 triangle using 6 node indices.

    interior_edges : list of tuple[int, int]
        Edges shared by two elements — interior to the domain.

    interior_boundary_edges : list of tuple[int, int]
        Edges on inner walls/boundaries (e.g., circle hole).

    exterior_boundary_edges : list of tuple[int, int]
        Edges on the outer rectangular boundary.

    inlet_edges : list of tuple[int, int]
        Edges on the left boundary of the domain (inlet).

    outlet_edges : list of tuple[int, int]
        Edges on the right boundary of the domain (outlet).

    pressure_nodes : ndarray of int
        Global node indices corresponding to pressure degrees of freedom (P1 vertex nodes).

    pressure_index_map : dict
        Mapping from global node index to index in the pressure DOF vector.

    periodic_map : dict[int, int]
        Mapping of slave nodes to master nodes for enforcing periodicity.

    Methods
    -------
    plot(show_node_ids=False, filename="mesh_plot.png")
        Plots the mesh geometry, including edge types and optionally node IDs.
        Saves the plot as a PNG image.

    plot_periodic_pairs()
        Visualizes periodic slave-master node pairs using color-coded markers.
        Saves the plot as a PNG image.

    check_triangle_orientation()
        Checks if all triangles are oriented positively (non-inverted).
        Reports number of inverted or zero-area triangles.
    """

    def __init__(self, nodes, triangles, interior_edges,pressure_nodes, pressure_index_map, 
                 interior_boundary_edges=None, exterior_boundary_edges=None, 
                 inlet_edges=None, 
                 outlet_edges=None,
                 periodic_map=None,
                 wall_edges=None):
        self.nodes = nodes
        self.triangles = triangles
        self.interior_edges = interior_edges
        self.wall_edges = wall_edges
        self.interior_boundary_edges = interior_boundary_edges or []
        self.exterior_boundary_edges = exterior_boundary_edges or []        
        self.inlet_edges = inlet_edges or []
        self.outlet_edges = outlet_edges or []
        self.pressure_nodes = pressure_nodes
        self.pressure_index_map = pressure_index_map
        self.periodic_map = periodic_map or {}


    def plot(self, show_node_ids=False, filename="mesh_plot.png"):
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
        start_plot = datetime.now()
    
        print("Plotting mesh...", end="", flush=True)
    
        plt.figure(figsize=(6, 6))
    
        def scale_coords(coord):
            return coord * 1000  # convert meters to millimeters
    
        if show_node_ids:
            for idx, (xi, yi) in enumerate(self.nodes[:, :2]):
                xi_mm, yi_mm = scale_coords(xi), scale_coords(yi)
                plt.scatter(xi_mm, yi_mm, color='C4', s=5, zorder=5)
                plt.text(xi_mm, yi_mm, str(idx), fontsize=6, zorder=6,
                         color='black', ha='center', va='center')
    
        def draw_edges(edges, color, label):
            for edge in edges:
                p1, p2 = self.nodes[edge[0]], self.nodes[edge[1]]
                x_vals = [scale_coords(p1[0]), scale_coords(p2[0])]
                y_vals = [scale_coords(p1[1]), scale_coords(p2[1])]
                plt.plot(x_vals, y_vals, color, linewidth=1.2, label=label)
    
        draw_edges(self.interior_edges, 'C0', 'Interior')
        #draw_edges(self.interior_boundary_edges, 'C1', 'Interior wall')
        #draw_edges(self.exterior_boundary_edges, 'C2', 'Periodic BC')
        draw_edges(self.inlet_edges, 'C3', 'Inlet')
        draw_edges(self.outlet_edges, 'C4', 'Outlet')
        draw_edges(self.wall_edges, 'C6', 'Wall')

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center')
    
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect("equal")
        plt.title("M*E*S*H*")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
    
        end_plot = datetime.now()
        print(f"\rPlotted mesh in {(end_plot - start_plot).total_seconds():.3f} seconds.")
        print(f"Mesh plot saved as {filename}.")

    def plot_periodic_pairs(self):
        """
        Plots each periodic slave/master node pair with a unique color and marker.
        """
    
        nodes = np.array(self.nodes)
        periodic_map = self.periodic_map
        filename = "periodic_pairs.png"
    
        plt.figure(figsize=(6, 6))
    
        # Set up color and marker cycles
        colors = list(cm.get_cmap('tab20').colors)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']
        color_marker_cycle = itertools.cycle([(c, m) for c in colors for m in markers])
    
        # Loop through periodic pairs
        for i, (slave, master) in enumerate(periodic_map.items()):
            x_slave, y_slave = nodes[slave, 0] * 1000, nodes[slave, 1] * 1000
            x_master, y_master = nodes[master, 0] * 1000, nodes[master, 1] * 1000
    
            color, marker = next(color_marker_cycle)
    
            plt.scatter(x_slave, y_slave, c=[color], marker=marker, s=30, label=f"Pair {i}" if i < 10 else "")
            plt.scatter(x_master, y_master, c=[color], marker=marker, s=30)
    
        plt.gca().set_aspect("equal")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title("Periodic Node Pairs")
        if len(periodic_map) < 10:
            plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        
        
    def check_triangle_orientation(self):
        bad = 0
        for tri in self.triangles:
            coords = self.nodes[tri[:3], :2]  # Only P1 nodes
            v0, v1, v2 = coords
            area = 0.5 * ((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
            if area <= 0:
                bad += 1
        if bad > 0:
            print(f"{bad} triangle(s) have non-positive orientation (possibly inverted).")
        else:
            print("All triangles have consistent positive orientation.")
            
    def check_mesh_quality(self, show_histograms=True):
        """
        Compute and display quality metrics for mesh triangles:
        - aspect ratio
        - minimum angle
        - area
    
        Parameters
        ----------
        show_histograms : bool
            Whether to plot histograms of the computed metrics.
    
        Returns
        -------
        dict
            Summary statistics for quality metrics.
        """
            
        def triangle_angles(a, b, c):
            """Returns angles (in degrees) of triangle with sides a, b, c using the Law of Cosines."""
            cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
            cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
            angles = np.degrees(np.arccos(np.clip([cos_A, cos_B, cos_C], -1.0, 1.0)))
            return angles
    
        min_angles = []
        aspect_ratios = []
        areas = []
    
        for tri in self.triangles:
            pts = self.nodes[tri[:3], :2]
            a = np.linalg.norm(pts[1] - pts[0])
            b = np.linalg.norm(pts[2] - pts[1])
            c = np.linalg.norm(pts[0] - pts[2])
    
            s = 0.5 * (a + b + c)
            area = max(s * (s - a) * (s - b) * (s - c), 0.0) ** 0.5
    
            angles = triangle_angles(a, b, c)
            min_angles.append(np.min(angles))
            aspect_ratios.append(max(a, b, c) / min(a, b, c))
            areas.append(area)
    
        min_angles = np.array(min_angles)
        aspect_ratios = np.array(aspect_ratios)
        areas = np.array(areas)
    
        print("\r--- Mesh Debug Info ---")
        self.check_triangle_orientation()

        print(f"Minimum angle: {min_angles.min():.2f}°")
        print(f"Maximum aspect ratio: {aspect_ratios.max():.2f}")
        print(f"Area range: [{areas.min():.2e}, {areas.max():.2e}]")
    
        # --- Warnings ---
        if min_angles.min() < 20.0:
            warnings.warn(f"Minimum angle is very small ({min_angles.min():.2f}°). This may lead to numerical instability.")
        if aspect_ratios.max() > 5.0:
            warnings.warn(f"Maximum aspect ratio is high ({aspect_ratios.max():.2f}). Consider refining mesh.")
        if areas.min() / areas.max() < 0.01:
            warnings.warn("Element size variation is large. Check if size field is behaving as intended.")
        
        # Check for duplicate triangles
        tri_tuples = [tuple(sorted(tri)) for tri in self.triangles]
        unique_tris = set(tri_tuples)
        if len(unique_tris) != len(self.triangles):
            dup_count = len(self.triangles) - len(unique_tris)
            warnings.warn(f"Detected {dup_count} duplicate triangles in mesh. This may indicate an export or periodicity issue.")
        else:
            print("No duplicate triangles found after meshing.")
            
        # check for duplicate nodes
        node_tuples = [tuple(np.round(node[:2], decimals=12)) for node in self.nodes]
        unique_nodes = set(node_tuples)
        if len(unique_nodes) != len(self.nodes):
            dup_count = len(self.nodes) - len(unique_nodes)
            warnings.warn(f"Detected {dup_count} duplicate nodes in mesh. This may indicate an export or periodicity issue.")
        else:
            print("No duplicate nodes found after meshing.")
            
        print("--- End mesh Info ---")
    
        if show_histograms:
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 3, 1)
            plt.hist(min_angles, bins=30, color='C0')
            plt.title("Min Angle [°]")
            plt.xlabel("Angle")
            plt.grid(True)
    
            plt.subplot(1, 3, 2)
            plt.hist(aspect_ratios, bins=30, color='C1')
            plt.title("Aspect Ratio")
            plt.xlabel("Ratio")
            plt.grid(True)
    
            plt.subplot(1, 3, 3)
            plt.hist(areas, bins=30, color='C2')
            plt.title("Triangle Area")
            plt.xlabel("Area")
            plt.grid(True)
    
            plt.tight_layout()
            plt.savefig("mesh_quality_histograms.png", dpi=300)
            plt.show()
    
        return {
            "min_angle_deg": min_angles.min(),
            "max_aspect_ratio": aspect_ratios.max(),
            "area_range": (areas.min(), areas.max())
        }


    def plot_pressure_nodes(self, filename="pressure_nodes.png", annotate=False):
        """
        Plots only the P1 (pressure) nodes over the mesh.
    
        Parameters
        ----------
        filename : str
            File to save the image to.
    
        annotate : bool
            Whether to annotate node indices.
        """
        plt.figure(figsize=(6, 6))
        coords = self.nodes[self.pressure_nodes]
        x = coords[:, 0] * 1000  # mm
        y = coords[:, 1] * 1000
    
        plt.scatter(x, y, c='C3', s=10, label='Pressure Nodes (P1)')
    
        if annotate:
            for idx, (xi, yi) in zip(self.pressure_nodes, coords):
                plt.text(xi * 1000, yi * 1000, str(idx), fontsize=6, color='black',
                         ha='center', va='center')
    
        # Use full self.nodes here, because triangle indices are global
        for tri in self.triangles:
            pts = self.nodes[tri[:3], :2] * 1000  # scale to mm
            plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
                     color='lightgray', linewidth=0.5)
    
        plt.title("Pressure Nodes (P1)")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved pressure node plot to {filename}.")


    def plot_velocity_nodes(self, filename="velocity_nodes.png", annotate=False):
        """
        Plots all P2 (velocity) nodes in the mesh.
    
        Parameters
        ----------
        filename : str
            File to save the image to.
    
        annotate : bool
            Whether to annotate node indices.
        """
        plt.figure(figsize=(6, 6))
        coords = self.nodes[:, :2]
    
        x = coords[:, 0] * 1000  # mm
        y = coords[:, 1] * 1000
    
        plt.scatter(x, y, c='C0', s=10, label='Velocity Nodes (P2)')
    
        if annotate:
            for idx, (xi, yi) in enumerate(coords):
                plt.text(xi * 1000, yi * 1000, str(idx), fontsize=6, color='black',
                         ha='center', va='center')
                
        # Use full self.nodes here, because triangle indices are global
        for tri in self.triangles:
            pts = self.nodes[tri[:3], :2] * 1000  # scale to mm
            plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
                     color='lightgray', linewidth=0.5)
    
        plt.title("Velocity Nodes (P2)")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved velocity node plot to {filename}.")
    

def create_mesh(geometry_length=0.01, mesh_size=0.0001, inner_radius=0.004, output_file="square_with_hole.msh", periodic=True):
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
    
    gmsh.model.occ.synchronize()

    if periodic:
        # --- Add periodicity: bottom (l1) is master, top (l3) is slave ---
        gmsh.model.mesh.setPeriodic(
            1,      # 1D entities (curves)
            [l3],   # slave: top boundary curve
            [l1],   # master: bottom boundary curve
            [1, 0, 0, 0,     # x' = x
             0, 1, 0, -l,    # y' = y - l → shift top to align with bottom
             0, 0, 1, 0,     # z unchanged
             0, 0, 0, 1]     # homogeneous coordinate
        )
        
        # --- Add periodicity: left (l4) is master, right (l2) is slave ---
        gmsh.model.mesh.setPeriodic(
            1,      # 1D entities (curves)
            [l2],   # slave: right
            [l4],   # master: left
            [1, 0, 0, -l,    # x' = x - l
             0, 1, 0, 0,     # y' = y
             0, 0, 1, 0,     # z unchanged
             0, 0, 0, 1]     # homogeneous coordinate
        )

    square_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    square_surface = gmsh.model.occ.addPlaneSurface([square_loop])

    # Circle hole
    r = inner_radius
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
    gmsh.model.occ.synchronize()  # <--- Synchronize BEFORE mesh field setup
    
    # Define points near which we want finer mesh (e.g., near circle center)
    p_left_circle = gmsh.model.occ.addPoint((geometry_length/2)-inner_radius, geometry_length/2, 0)
    p_right_circle = gmsh.model.occ.addPoint((geometry_length/2)+inner_radius, geometry_length/2, 0)
    p_right_wall = gmsh.model.occ.addPoint(geometry_length, geometry_length/2, 0)
    p_left_wall = gmsh.model.occ.addPoint(0, geometry_length/2, 0)
    
    gmsh.model.occ.synchronize()
    
    # --- Field 1: Distance from all fine-mesh areas (circle + walls) ---
    gmsh.model.mesh.field.add("Distance", 1)
    all_refine_pts = [p_left_circle, p_right_circle, p_left_wall, p_right_wall]
    gmsh.model.mesh.field.setNumbers(1, "NodesList", all_refine_pts)
    
    # --- Field 2: Threshold size control based on distance ---
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.2 * mesh_size)   # very fine near features
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 2.0 * mesh_size)   # coarse elsewhere
    gmsh.model.mesh.field.setNumber(2, "DistMin", (geometry_length*0.5 - inner_radius)/4)
    gmsh.model.mesh.field.setNumber(2, "DistMax", geometry_length)  # ~3% of geometry size
    
    # Set background mesh size
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # Set P2 with linear edges
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)  # Make the triangles straight-lined
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)

    gmsh.model.mesh.generate(2)

    gmsh.write(output_file)
    gmsh.clear()
    gmsh.finalize()

    return

def load_mesh(mesh=None, periodic=False):
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
        warnings.warn("Using linear triangles (P1), not P2. Higher-order accuracy may be reduced.")
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


    # Classify edges by counting how many times they appear
    edge_list = defaultdict(int)
    for tri in triangles:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for edge in edges:
            edge_list[tuple(sorted(edge))] += 1

    interior_edges = []
    interior_boundary_edges = []
    exterior_boundary_edges = []
    inlet_edges = []
    outlet_edges = []
    wall_edges = []

    for edge, count in edge_list.items():
        if count == 1:  # boundary edge
            # First  check if inlet or outlet
            p1, p2 = nodes[edge[0]], nodes[edge[1]]
            x1, x2 = p1[0], p2[0]
            y1, y2 = p1[1], p2[1]

            xmax = np.max(nodes)
            xmin = np.min(nodes)
            if np.isclose(y1, xmax) and np.isclose(y2, xmax):
                inlet_edges.append(edge)  # Top → inflow
            elif np.isclose(y1, xmin) and np.isclose(y2, xmin):
                outlet_edges.append(edge)  # Bottom → outflow
            else:
                wall_edges.append(edge)
                
            # Then check if interior or exterior
            margin = 1e-7  # numerical tolerance
            xc = (p1[0] + p2[0]) / 2
            yc = (p1[1] + p2[1]) / 2
            xmin, xmax = np.min(nodes[:, 0]), np.max(nodes[:, 0])
            ymin, ymax = np.min(nodes[:, 1]), np.max(nodes[:, 1])
            
            if (np.isclose(xc, xmin, atol=margin) or
                np.isclose(xc, xmax, atol=margin) or
                np.isclose(yc, ymin, atol=margin) or
                np.isclose(yc, ymax, atol=margin)):
                exterior_boundary_edges.append(edge)
            else:
                interior_boundary_edges.append(edge)
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
        pressure_nodes=pressure_nodes,
        pressure_index_map=pressure_index_map,
        interior_edges=interior_edges,
        wall_edges=wall_edges,
        interior_boundary_edges=interior_boundary_edges,
        exterior_boundary_edges=exterior_boundary_edges,
        inlet_edges=inlet_edges,
        outlet_edges=outlet_edges
    )
    
    if periodic:
        # Create periodic map, mapping masters and slaves
        x_map = find_periodic_pairs(nodes, axis=0)
        y_map = find_periodic_pairs(nodes, axis=1)
        mesh.periodic_map = {**x_map, **y_map}

    return mesh


def find_periodic_pairs(nodes, axis, tol=1e-10):
    """
    Finds periodic node pairs along a given axis (0 for x, 1 for y).
    Matches all nodes on one boundary (slave) to their master counterparts
    on the opposite side, based on orthogonal coordinate proximity.
    """
    coord_min = np.min(nodes[:, axis])
    coord_max = np.max(nodes[:, axis])

    master_nodes = np.where(np.isclose(nodes[:, axis], coord_min, atol=tol))[0]
    slave_nodes = np.where(np.isclose(nodes[:, axis], coord_max, atol=tol))[0]

    other_axis = 1 - axis
    master_coords = nodes[master_nodes][:, other_axis][:, None]
    slave_coords = nodes[slave_nodes][:, other_axis][:, None]

    tree = cKDTree(master_coords)
    dists, idxs = tree.query(slave_coords, distance_upper_bound=tol)

    periodic_map = {}
    for slave_idx, (master_idx, dist) in enumerate(zip(idxs, dists)):
        if dist < tol:
            slave = slave_nodes[slave_idx]
            master = master_nodes[master_idx]
            periodic_map[slave] = master

    return periodic_map

if __name__ == "__main__":
    """
    When run directly, this script will:
    1. Generate a mesh using Gmsh.
    2. Load and classify mesh edges and pressure nodes.
    3. Plot the mesh with labels and edge classifications.
    
    """
    mesh_file = "square_hole.msh"
    geometry_length=0.1
    inner_radius = geometry_length * 0.2
    mesh_size = 0.5 * geometry_length
    
    total_start = datetime.now()

    print("Creating mesh...", end="", flush=True)
    create_mesh(geometry_length, mesh_size, inner_radius, output_file=mesh_file)
    
    print("Loading mesh...", end="", flush=True)
    raw_mesh = meshio.read(mesh_file)
    mesh = load_mesh(raw_mesh)
    
    print("Checking mesh quality...", end="", flush=True)
    mesh.check_mesh_quality()

    print("Plotting mesh...", end="", flush=True)
    mesh.plot(show_node_ids=False, filename="mesh.png")
    
    end_mesh = datetime.now()
    print(f"\rMeshed in {(end_mesh - total_start).total_seconds():.3f} seconds")