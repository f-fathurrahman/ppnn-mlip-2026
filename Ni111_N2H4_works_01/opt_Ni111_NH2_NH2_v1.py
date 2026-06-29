import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)

# Set up MACE calculator
from mace.calculators import MACECalculator
model_path = "./mace-mh-1.model"
calc = MACECalculator(
    model_paths=model_path, device=device, head="oc20_usemppbe"
)

import ase.io
from ase.optimize import LBFGS

surf_with_ads = ase.io.read("initial_Ni111_NH2_NH2.xyz")
surf_with_ads.calc = calc
ftraj = "Ni111_NH2_NH2_geoopt.traj"
opt = LBFGS(surf_with_ads, trajectory=ftraj)
opt.run(0.05, 1000)
E_surf_with_ads = surf_with_ads.get_total_energy()

