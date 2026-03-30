"""
Extract subformulae JSON entries from HDF5 to a directory of individual .json files.

FFN, GNN, and 3DMolMS data loaders expect a directory of JSON files named {spec_name}.json.
The HDF5 stores keys as {spec_name}_collision {ce}.json.

For MSG, each spectrum has exactly one CE, so we map 1:1.

Usage:
    cd ms-pred
    python scripts/extract_subform_to_dir.py \
        --hdf5 data/spec_datasets/msg/subformulae/no_subform.hdf5 \
        --labels data/spec_datasets/msg/labels.tsv \
        --out-dir data/spec_datasets/msg/subformulae/no_subform
"""
import argparse
import ast
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ms_pred import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels, sep="\t")
    h5 = common.HDF5Dataset(args.hdf5)
    all_keys = set(h5.get_all_names())

    spec_to_keys = {}
    for k in all_keys:
        base = k.rsplit("_collision ", 1)[0] if "_collision " in k else k.rsplit(".json", 1)[0]
        spec_to_keys.setdefault(base, []).append(k)

    print(f"HDF5 keys: {len(all_keys)}, unique specs: {len(spec_to_keys)}, labels rows: {len(labels)}")

    n_ok = 0
    n_miss = 0
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Extracting"):
        spec_name = row["spec"]
        keys = spec_to_keys.get(spec_name)
        if keys:
            json_str = h5.read_str(keys[0])
            out_path = out_dir / f"{spec_name}.json"
            with open(out_path, "w") as f:
                f.write(json_str)
            n_ok += 1
        else:
            n_miss += 1

    h5.h5_obj.close()
    print(f"Extracted: {n_ok}, missing: {n_miss}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
