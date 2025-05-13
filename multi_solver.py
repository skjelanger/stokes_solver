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


# Create folders
os.makedirs("plots", exist_ok=True)
os.makedirs("simulations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Values to simulate

# Constants
geometry_length = 0.001
inner_radii = geometry_length * np.array([0.35, 0.40, 0.45, 0.47, 0.49])
mesh_size = geometry_length * np.array([0.05, 0.05, 0.05, 0.05, 0.05])
geometry_height = 0.000091 # 91 micrometer thickness
mu = 0.00089
rho = 1000.0

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
        create_mesh(geometry_length, mesh_size, inner_radius, "square_with_hole.msh")
        mesh_time = datetime.now()
        print(f"Created mesh in {(mesh_time - mesh_start).total_seconds():.3f} seconds.")
        
        raw_mesh = meshio.read("square_with_hole.msh")
        mesh = load_mesh(raw_mesh)
        mesh.mesh_size = mesh_size
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

    except Exception as e:
        print(f"Simulation with radius {inner_radius:.2f} FAILED: {e}")

print("\n=== All simulations completed ===")

