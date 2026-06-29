#pip install ase mace-torch
#pip install weas-widget
# !curl -L -J -O https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model

import copy
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)

import ase.build
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.optimize import MDMin

# Set up MACE calculator
from mace.calculators import MACECalculator
model_path = "../mace-mh-1.model"
calc = MACECalculator(
    model_paths=model_path, device=device, head="oc20_usemppbe"
)


# Buat surface
surf = ase.build.fcc111("Ni", (4, 3, 5), vacuum=10.0 )
# Get the lowest layer z-coordinate
zmin = surf.get_positions()[:, 2].min()
# Fix the lowest layer
idx_atoms_lowest_layer = surf.positions[:, 2] < zmin + 0.01
constraint = FixAtoms(mask=idx_atoms_lowest_layer)
surf.set_constraint(constraint)

# Perturb positions of other atoms
import numpy as np
idx_atoms_others = ~idx_atoms_lowest_layer
num_moved = np.sum(idx_atoms_others)
surf.positions[idx_atoms_others] += 0.1*np.random.randn(num_moved,3)

surf.calc = calc
# Set up MDMin dynamics object
ftraj = "Ni111_surf_geoopt_mdmin.traj"
opt = MDMin(surf, trajectory=ftraj)
opt.run(0.05, 1000)
E_surf = surf.get_total_energy()

"""
# XXX: No need to set the unit cell here?
mol_isolated = ase.build.molecule("N2H4")
mol_isolated.calc = calc
ftraj = "N2H4_geoopt_mdmin.traj"
opt = MDMin(mol_isolated, trajectory=ftraj)
opt.run(0.05, 1000)
E_mol_isolated = mol_isolated.get_total_energy()

surf_with_ads = surf.copy()
mol_isolated.rotate(90, "z", center="COM")
ase.build.add_adsorbate(
    surf_with_ads, mol_isolated, 2.5, position=(4,2.5), mol_index=0)
surf_with_ads.calc = calc
ftraj = "Ni111_N2H4_geoopt_mdmin.traj"
opt = MDMin(surf_with_ads, trajectory=ftraj)
opt.run(0.05, 1000)
E_surf_with_ads = surf_with_ads.get_total_energy()

E_ads = E_surf_with_ads - ( E_surf + E_mol_isolated )
print("E_ads = ", E_ads)
"""