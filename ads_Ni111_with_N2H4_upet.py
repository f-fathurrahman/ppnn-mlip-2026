import copy
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)

import ase.build
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.optimize import LBFGS

from upet.calculator import UPETCalculator
checkpoint_path = "./pet-mad-s-v1.5.0.ckpt"
calc = UPETCalculator(checkpoint_path=checkpoint_path, device="cpu")


# Buat surface
surf = ase.build.fcc111("Ni", (4, 4, 5), vacuum=10.0 )
# Get the lowest layer z-coordinate
zmin = surf.get_positions()[:, 2].min()
# Fix the lowest layer
idx_atoms_lowest_layer = surf.positions[:, 2] < zmin + 0.01
constraint = FixAtoms(mask=idx_atoms_lowest_layer)
surf.set_constraint(constraint)

surf.calc = calc
# Set up LBFGS dynamics object
ftraj = "Ni111_surf_geoopt.traj"
opt = LBFGS(surf, trajectory=ftraj)
opt.run(0.05, 1000)
E_surf = surf.get_total_energy()

# XXX: No need to set the unit cell here?
mol_isolated = ase.build.molecule("N2H4")
mol_isolated.calc = calc
ftraj = "N2H4_geoopt.traj"
opt = LBFGS(mol_isolated, trajectory=ftraj)
opt.run(0.05, 1000)
E_mol_isolated = mol_isolated.get_total_energy()

surf_with_ads = surf.copy() #XXX cannot use deepcopy?
ase.build.add_adsorbate(
    surf_with_ads, mol_isolated, 2.5, position=(3,3), mol_index=0)
surf_with_ads.calc = calc
ftraj = "Ni111_N2H4_geoopt.traj"
opt = LBFGS(surf_with_ads, trajectory=ftraj)
opt.run(0.05, 1000)
E_surf_with_ads = surf_with_ads.get_total_energy()

E_ads = E_surf_with_ads - ( E_surf + E_mol_isolated )
print("E_ads = ", E_ads)

