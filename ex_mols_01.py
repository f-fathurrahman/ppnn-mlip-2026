from ase import Atoms
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

atoms = Atoms('OHH', positions=[
    (0, 0, 0),
    (0, 0, 0.9),
    (0, 1.1, 0.0)
])

fig, ax = plt.subplots(figsize=(4, 4))
plot_atoms(atoms, ax=ax, rotation='45x,10y,0z', scale=1.0)
plt.axis("off")
plt.savefig("IMG_atoms.png", dpi=300, bbox_inches='tight')
plt.show()
