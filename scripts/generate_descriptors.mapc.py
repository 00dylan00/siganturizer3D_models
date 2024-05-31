"""Generating MAPC descriptors

Structure:
    1. Imports, Variables, Functions
    2. Load Data
    3. Compute MAPC Descriptors
    4. Save Data
"""

# 1. Imports, Variables, Functions
# imports
from rdkit import Chem
import sys, os, h5py
from mapchiral.mapchiral import encode, jaccard_similarity
import logging
import numpy as np
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# variables
s0_filtered_path = os.path.join("..", "data", "B4.001", "s0_filtered.h5")
output_mapc_path = os.path.join("..", "data", "features", "mapc.h5")

# functions


# 2. Load Data
if os.path.exists(output_mapc_path):
    logging.info("Data already exists")
    with h5py.File(output_mapc_path, "r") as f:
        mapc_matrix = f["V"][:]
        s0_smiles = f["smiles"][:].astype(str)
        s0_iks = f["inchikeys"][:].astype(str)

    logging.info(
        f"MAPC matrix shape {mapc_matrix.shape}, nº of smiles {len(s0_smiles)}, nº of inchikeys {len(s0_iks)}"
    )

else:

    with h5py.File(s0_filtered_path, "r") as f:
        s0_V = f["V"][:]
        s0_features = f["features"][:].astype(str)
        s0_smiles = f["smiles"][:].astype(str)
        s0_iks = f["inchikeys"][:].astype(str)

    # 3. Compute MAPC Descriptors
    mpac_descriptors = list()
    for smiles in tqdm(s0_smiles):
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = encode(molecule, max_radius=2, n_permutations=2048, mapping=False)
        mpac_descriptors.append(fingerprint)

    mapc_matrix = np.array(mpac_descriptors)

    # 4. Save Data
    s0_smiles = np.array(s0_smiles, dtype="S")
    s0_iks = np.array(s0_iks, dtype="S")

    with h5py.File(output_mapc_path, "w") as f:
        f.create_dataset("V", data=mapc_matrix)
        f.create_dataset("smiles", data=s0_smiles)
        f.create_dataset("inchikeys", data=s0_iks)