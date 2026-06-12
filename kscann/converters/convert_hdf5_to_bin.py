import h5py
import numpy as np

with h5py.File('../data/gist/gist-960-euclidean.hdf5', 'r') as f:
    neighbors = f['neighbors'][:]  # shape (1000, 100)
    # Save as binary (int32)
    with open('gist1m_groundtruth.bin', 'wb') as out:
        for row in neighbors:
            out.write(row.astype(np.int32).tobytes())