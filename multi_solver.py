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
geometry_length = 1
inner_radii = geometry_length * np.array([0.35, 0.40, 0.45, 0.47, 0.49])
mesh_size = geometry_length * np.array([0.0035, 0.0030, 0.0023, 0.00195, 0.0016])
geometry_height = 0.091 # 91 micrometer thickness
mu = 1 #0.00089
rho = 1 # 1000.0

periodic = True

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
        create_mesh(geometry_length, mesh_size[idx - 1], inner_radius, "square_with_hole.msh", periodic = periodic)
        mesh_time = datetime.now()
        print(f"Created mesh in {(mesh_time - mesh_start).total_seconds():.3f} seconds.")
        
        raw_mesh = meshio.read("square_with_hole.msh")
        mesh = load_mesh(raw_mesh, periodic)
        mesh.mesh_size = mesh_size[idx - 1]
        load_time = datetime.now()
        print(f"Loaded mesh with {mesh.triangles.shape[0]} elements in {(load_time - mesh_time).total_seconds():.3f} seconds.")

        mesh.check_mesh_quality()
        check_time = datetime.now()
        print(f"Checked mesh in {(check_time - load_time).total_seconds():.3f} seconds.")
        
        sim = Simulation(mesh, f, geometry_length, inner_radius, geometry_height, mu=mu, rho=rho, periodic_bc=periodic)
        sim.run()
        
        start_calc = datetime.now()
        sim.calculate_permeability(direction='y')
        print(f"Calculated permeability in {(datetime.now() - start_calc).total_seconds():.3f} seconds.")
        
        desc = sim.get_description()
        safe_desc = desc.replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p")
        sim.save(f"simulations/sim_{safe_desc}.pkl")
        
        # Store convergence data by radius
        if hasattr(sim, "permeability") and hasattr(sim, "res_norm"):
            r_key = f"{inner_radius}"
            if r_key not in results_by_radius:
                results_by_radius[r_key] = {
                    "mesh_sizes": [],
                    "residuals": [],
                    "permeabilities": []
                }
            results_by_radius[r_key] = {
                "mesh_size": mesh.mesh_size,
                "residual": sim.res_norm,
                "permeability": sim.permeability
            }

    except Exception as e:
        print(f"Simulation with radius {inner_radius:.2f} FAILED: {e}")
        
print(f"\nMulti-sim completed: {results_by_radius}.")        

if results_by_radius:
    radii_mm = []
    permeabilities = []
    residuals = []

    for r, data in results_by_radius.items():
        radii_mm.append(float(r) * 1000)
        permeabilities.append(data["permeability"])
        residuals.append(data["residual"])

    # Permeability vs Radius
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(radii_mm, permeabilities, marker='o')
    plt.xlabel("Inner Radius [mm]")
    plt.ylabel("Permeability [m²]")
    plt.title("Permeability vs. Inner Radius")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/permeability_vs_radius.png", dpi=300)
    plt.show()

    # Residuals vs Radius
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(radii_mm, residuals, marker='o')
    plt.xlabel("Inner Radius [mm]")
    plt.ylabel("Residual Norm")
    plt.title("Residual Norm vs. Inner Radius")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/residual_vs_radius.png", dpi=300)
    plt.show()
else:
    print("No valid simulation data to plot.")
    
print("\n=== All simulations completed ===")

