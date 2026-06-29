# Set up MACE calculator
from mace.calculators import MACECalculator
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)
model_path = "../mace-mh-1.model"
calc = MACECalculator(
    model_paths=model_path, device=device, head="oc20_usemppbe"
)

import ase.io
from ase.mep import NEB, SingleCalculatorNEB
from ase.optimize import MDMin, LBFGS

# Read initial and final states:
initial = ase.io.read("initial_struct.xyz")
final = ase.io.read("final_struct.xyz")

# Make a band consisting:
Nimages_intermediate = 6
images = [initial]
images += [initial.copy() for i in range(Nimages_intermediate)]
images += [final]

# Set calculators (for all images)
for image in images:
    image.calc = calc

neb = NEB(images, allow_shared_calculator=True)
# Interpolate linearly the potisions of the three middle images:
neb.interpolate()
ftraj = "NEB_Ni111_N2H4_to_NH2_NH2.traj"
optimizer = LBFGS(neb, trajectory=ftraj)
optimizer.run(fmax=0.04)

