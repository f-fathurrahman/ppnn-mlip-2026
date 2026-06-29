import ase.build
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.visualize.plot import plot_atoms

surf = ase.build.fcc111("Ni", (3, 3, 5), vacuum=10.0 )
mol_isolated = ase.build.molecule("N2H4")
mol_isolated.rotate(90, 'z', center="COM")

surf_with_ads = surf.copy()
ase.build.add_adsorbate(
    surf_with_ads, mol_isolated, 2.5, position=(4,2.5), mol_index=0
)

view(surf_with_ads)

#fig, ax = plt.subplots(figsize=(4, 4))
#plot_atoms(surf_with_ads, ax=ax, rotation='0x,0y,0z', scale=1.0)
#plt.axis("off")
#plt.savefig("IMG_surf-with_ads.png", dpi=300, bbox_inches='tight')
#plt.show()
