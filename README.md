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

## Notes
Periodic boundary conditions are applied at the matrix level via slave-master DOF enforcement.

Meshing can be adapted by adjusting create_mesh() in mesher.py.

Pressure is anchored to one node to ensure solvability.

## Acknowledgements
Created during the PCS911 course at Western Norway University of Applied Sciences
