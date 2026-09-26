import os
import glob
import subprocess
import yaml
import h5py
import numpy as np
import pandas as pd
import random

# Absolute paths are required so the script can run seamlessly from any directory
build_dir = "/home/sajid/gyselalibxx/build/simulations/geometryXVx"
executable = "/home/sajid/gyselalibxx/build/simulations/geometryXVx/vlasovpoisson_xvx_fem_uniform_xperiod_vx"
base_yaml = "/home/sajid/gyselalibxx/simulations/geometryXVx/landau_twospecies.yaml"

# 1. Define the continuous parameter ranges for random sampling
range_energy = (0.0, 2.0)       # Heating power
range_extent = (0.05, 0.45)     # Deposition width (shape)
range_stiffness = (1.0, 10.0)   # Profile sharpness (gradient)

num_random_samples = 20         # Change this to generate a larger dataset

with open(base_yaml, 'r') as file:
    config = yaml.safe_load(file)

dataset = []

for i in range(num_random_samples):
    # Randomly pick values from a uniform distribution
    rand_energy = random.uniform(*range_energy)
    rand_extent = random.uniform(*range_extent)
    rand_stiffness = random.uniform(*range_stiffness)
    
    print(f"\n--- Run {i+1}/{num_random_samples} ---")
    print(f"Energy: {rand_energy:.3f} | Extent: {rand_extent:.3f} | Stiffness: {rand_stiffness:.3f}")
    
    # Update config dynamically
    config['KineticSource']['energy'] = float(rand_energy)
    config['KineticSource']['extent'] = float(rand_extent)
    config['KineticSource']['stiffness'] = float(rand_stiffness)
    
    # Use absolute path so the C++ app can always find it
    temp_yaml = os.path.abspath("temp_random_run.yaml")
    with open(temp_yaml, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    # Clean old HDF5 files from the build directory to avoid reading stale data
    for f in glob.glob(os.path.join(build_dir, "GYSELALIBXX_*.h5")):
        os.remove(f)
        
    # Execute the C++ simulation (Removed stdout=DEVNULL so you can see actual C++ physics errors)
    try:
        subprocess.run([executable, temp_yaml], check=True, cwd=build_dir)
    except subprocess.CalledProcessError:
        print("Simulation crashed/aborted. Skipping data extraction for this run...")
        continue
        
    # 2. Data Extraction - Look for files inside the build directory
    h5_files = sorted(glob.glob(os.path.join(build_dir, "GYSELALIBXX_*.h5")))
    if not h5_files:
        print("No HDF5 files found. Skipping extraction.")
        continue
    
    last_file = h5_files[-1]
    
    try:
        with h5py.File(last_file, 'r') as h5:
            time_val = h5['time_saved'][()]
            
            # Extract mean spatial values
            te_mean = np.mean(h5['temperature'][0, :])
            ti_mean = np.mean(h5['temperature'][1, :])
            ne_mean = np.mean(h5['density'][0, :])
            ni_mean = np.mean(h5['density'][1, :])
            
            # Extract core temperature (center of the grid at index 64)
            te_core = h5['temperature'][0, 64]
            
        # 3. Append to dataset
        dataset.append({
            'shot': i + 1,
            'energy': rand_energy,
            'extent': rand_extent,
            'stiffness': rand_stiffness,
            'final_time': time_val,
            'mean_te': te_mean,
            'mean_ti': ti_mean,
            'core_te': te_core,
            'mean_ne': ne_mean,
            'mean_ni': ni_mean
        })
    except Exception as e:
        print(f"Failed to read HDF5 output: {e}")

# 4. Push everything to a CSV file
df = pd.DataFrame(dataset)
csv_filename = "random_exploration_dataset.csv"
df.to_csv(csv_filename, index=False)

# Cleanup temporary yaml
if os.path.exists(temp_yaml):
    os.remove(temp_yaml)

print(f"\nRandom sampling complete! Data saved to {csv_filename}")
