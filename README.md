# FEM Stokes Flow Solver with Periodic Boundaries and Adaptive Meshing

This project simulates 2D incompressible Stokes flow through a square domain with a circular obstacle. The solver is based on **P2-P1 mixed finite elements**, periodic boundary conditions, and is designed to assess **permeability** under varying geometries.

Mesh generation and classification are handled via **Gmsh** and **Meshio**, while linear systems are solved with **Pypardiso** for efficiency.

---

## Features

- Quadratic P2 elements for velocity and linear P1 elements for pressure
- Support for periodic boundary conditions (top-bottom and left-right)
- Adaptive meshing around obstacles and domain walls
- Batch simulation support with logging and result saving
- Permeability calculation via Darcy's law
- Mesh and result visualization (pressure, velocity fields, vector plots)

---

## Dependencies

Install the following Python packages (e.g., via pip or conda):

- `gmsh`
- `meshio`
- `numpy`
- `scipy`
- `matplotlib`
- `pypardiso` *(or switch to another solver manually if needed)*

## Running the Stokes Solver
###To execute a simulation:

1. Download stokes_solver.py and mesher.py.
2. Install the required dependencies listed above.
3. Run the main solver script:

bash
python stokes_solver.py
bash

###This will:

1. Generate a mesh using Gmsh.
2. Assemble and solve the Stokes system using P2-P1 mixed finite elements.
3. Apply periodic or inflow/outflow boundary conditions (as configured).
4. Plot and save the velocity magnitude, velocity vectors, and pressure fields.
5. Compute the permeability.
6. Save the entire simulation state (solution + mesh + metadata) as a compressed file under simulations/.

###Output
- Figures are saved to the plots/ folder as .png files.
- Full simulation results are saved as .pkl.gz files under simulations/.

You can modify mesh parameters and boundary settings directly in the __main__ section at the bottom of stokes_solver.py.

## Notes
Periodic boundary conditions are applied at the matrix level via slave-master DOF enforcement.

Meshing can be adapted by adjusting create_mesh() in mesher.py.

Pressure is anchored to one node to ensure solvability.

## Acknowledgements
Created during the PCS911 course at Western Norway University of Applied Sciences
