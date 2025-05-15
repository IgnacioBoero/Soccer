#!/usr/bin/env python3
"""
Randomly split data/trajectories_00.pkl … 68.pkl into
  data/train  (59 files)
  data/val    (10 files)
"""

import random
from pathlib import Path
import shutil

# --------------------------------------------------------------------
SRC_DIR = Path("data/processed")           # where the .pkl files already live
DEST_TRAIN = SRC_DIR / "train"
DEST_VAL   = SRC_DIR / "val"

N_TRAIN, N_VAL = 59, 10          # counts for each split
SEED = 42                        # change for a different random split
MOVE_INSTEAD_OF_COPY = False     # set True to move rather than copy
# --------------------------------------------------------------------

# make sure destination dirs exist
DEST_TRAIN.mkdir(parents=True, exist_ok=True)
DEST_VAL.mkdir(parents=True, exist_ok=True)

# list of available indices (00 … 68 inclusive → 69 files)
indices = list(range(69))
random.seed(SEED)
random.shuffle(indices)

train_idx = indices[:N_TRAIN]
val_idx   = indices[N_TRAIN:N_TRAIN + N_VAL]

def transfer(split_idx, dest_root):
    for i in split_idx:
        src_file  = SRC_DIR / f"trajectories_0{i:02d}.pkl"
        dest_file = dest_root / src_file.name
        if dest_file.exists():
            dest_file.unlink()      # remove any earlier copy/move
        if MOVE_INSTEAD_OF_COPY:
            shutil.move(src_file, dest_file)
        else:
            shutil.copy2(src_file, dest_file)  # preserves metadata

transfer(train_idx, DEST_TRAIN)
transfer(val_idx,   DEST_VAL)

print(f"Copied {len(train_idx)} files to {DEST_TRAIN}")
print(f"Copied {len(val_idx)} files to {DEST_VAL}")
print("Train indices:", sorted(train_idx))
print("Val   indices:", sorted(val_idx))
