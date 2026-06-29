from ase.io import read, write

# Read all frames from the trajectory file
# Use '::skip' if you want to skip frames to save space
traj = read('simulation.traj', index=':')

# Iterate through each frame and save to a separate file
for i, atoms in enumerate(traj):
    filename = f'frame_{i+1}.xyz' # or .vasp, .cif, etc.
    write(filename, atoms)

print(f"Successfully exported {len(traj)} frames!")
