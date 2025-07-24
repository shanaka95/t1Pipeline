import numpy as np
import pickle
import sys

# Usage: python save_pose3d_to_pickle.py <input_npy> <output_pkl>
if len(sys.argv) != 3:
    print("Usage: python save_pose3d_to_pickle.py <input_npy> <output_pkl>")
    sys.exit(1)

input_npy = sys.argv[1]
output_pkl = sys.argv[2]

arr = np.load(input_npy)
print(f"Loaded {input_npy} with shape {arr.shape}")

# Save as a list with one segment
with open(output_pkl, 'wb') as f:
    pickle.dump([arr], f)
print(f"Saved as {output_pkl} (list with 1 segment)") 