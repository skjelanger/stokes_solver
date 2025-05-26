# -*- coding: utf-8 -*-
"""
Created on Tue May 13 20:14:03 2025

@author: skje
"""


import os
from datetime import datetime
import sys
import meshio
import numpy as np
from stokes_solver import Simulation
from mesher import create_mesh, load_mesh
import matplotlib.pyplot as plt

# Create folders
os.makedirs("plots", exist_ok=True)
os.makedirs("simulations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Constants
geometry_length = 0.001
<<<<<<< HEAD






inner_radii = geometry_length * np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35])
mesh_size = geometry_length * np.array([0.04, 0.008, 0.003, 0.002, 0.0014, 0.0012])
geometry_height = 0.000091 # 91 micrometer thickness
mu = 0.00089
rho = 1000.0

# Tracking results
mesh_sizes_tracked = []
results_by_radius = {}  # Dict: radius → dict of lists

def f(x, y):
    return (0, -1)

# Setup log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/simulation_batch_{timestamp}.txt"

class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # Needed for compatibility with some environments
    
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_filename)

# Run simulations
for idx, inner_radius in enumerate(inner_radii, 1):
    print(f"\n\n=== Simulation {idx} of {len(inner_radii)}: inner_radius = {inner_radius:.5f} ===")
    try:
        mesh_start = datetime.now()
        create_mesh(geometry_length, mesh_size[idx - 1], inner_radius, "square_with_hole.msh")
        mesh_time = datetime.now()
        print(f"Created mesh in {(mesh_time - mesh_start).total_seconds():.3f} seconds.")
        
        raw_mesh = meshio.read("square_with_hole.msh")
        mesh = load_mesh(raw_mesh)
        mesh.mesh_size = mesh_size[idx - 1]
        load_time = datetime.now()
        print(f"Loaded mesh with {mesh.triangles.shape[0]} elements in {(load_time - mesh_time).total_seconds():.3f} seconds.")

        mesh.check_mesh_quality()
        check_time = datetime.now()
        print(f"Checked mesh in {(check_time - load_time).total_seconds():.3f} seconds.")
        
        sim = Simulation(mesh, f, geometry_length, inner_radius, geometry_height, mu=mu, rho=rho)
        sim.run()
        
        start_calc = datetime.now()
        sim.calculate_permeability(direction='y')
        print(f"Calculated permeability in {(datetime.now() - start_calc).total_seconds():.3f} seconds.")
        
        desc = sim.get_description()
        safe_desc = desc.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
        sim.save(f"simulations/sim_{safe_desc}.pkl")
        
        # Store convergence data by radius
        if hasattr(sim, "permeability") and hasattr(sim, "res_norm"):
            r_key = f"{inner_radius:.5e}"
            if r_key not in results_by_radius:
                results_by_radius[r_key] = {
                    "mesh_sizes": [],
                    "residuals": [],
                    "permeabilities": []
                }
            results_by_radius[r_key]["mesh_sizes"].append(mesh.mesh_size)
            results_by_radius[r_key]["residuals"].append(sim.res_norm)
            results_by_radius[r_key]["permeabilities"].append(sim.permeability)

    except Exception as e:
        print(f"Simulation with radius {inner_radius:.2f} FAILED: {e}")
        
print(f"\nMulti-sim completed: {results_by_radius}.")        

if results_by_radius:
    # Plot residuals vs mesh size (log-log) for all radii
    plt.figure(figsize=(6, 5), dpi=300)
    for r, data in results_by_radius.items():
        sorted_data = sorted(zip(data["mesh_sizes"], data["residuals"], data["permeabilities"]))
        mesh_sizes, residuals, permeabilities = zip(*sorted_data)
        r_float = float(r)
        mesh_sizes_mm = [s * 1000 for s in data["mesh_sizes"]]  # Convert to mm
        plt.loglog(mesh_sizes_mm, data["residuals"], marker='o', label=f"r = {r_float*1000:.2f} mm")
        
    plt.xlabel("Mesh size [mm]")
    plt.ylabel("Solver Residual Norm")
    plt.title("Convergence: Residual vs. Mesh Size (Log-Log)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/residual_vs_mesh_size_loglog_by_radius.png", dpi=300)
    plt.show()
    
    # Plot permeability vs mesh size (log-log) for all radii
    plt.figure(figsize=(6, 5), dpi=300)
    for r, data in results_by_radius.items():
        sorted_data = sorted(zip(data["mesh_sizes"], data["residuals"], data["permeabilities"]))
        mesh_sizes, residuals, permeabilities = zip(*sorted_data)
        r_float = float(r)
        mesh_sizes_mm = [s * 1000 for s in data["mesh_sizes"]]  # Convert to mm
        plt.loglog(mesh_sizes_mm, data["permeabilities"], marker='o', label=f"r = {r_float*1000:.2f} mm")
        
    plt.xlabel("Mesh size [mm]")
    plt.ylabel("Permeability [m²]")
    plt.title("Permeability vs. Mesh Size (Log-Log)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/permeability_vs_mesh_size_loglog_by_radius.png", dpi=300)
    plt.show()
else:
    print("No valid simulation data to plot.")
    
print("\n=== All simulations completed ===")

