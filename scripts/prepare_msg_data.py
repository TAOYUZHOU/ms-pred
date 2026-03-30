"""
Prepare MassSpecGym data for ms-pred training.

Reads MassSpecGym.tsv and generates:
  1. spec_files.hdf5  — spectra in ms-pred MGF-like text format
  2. splits/split.tsv — train/val/test split (from MassSpecGym 'fold' column)
  3. labels.tsv is already present, but we regenerate labels_withev.tsv
     with per-spectrum collision energy info

Usage:
    cd ms-pred
    python scripts/prepare_msg_data.py \
        --msg-tsv ../data/MassSpecGym.tsv \
        --out-dir data/spec_datasets/msg
"""
import argparse
import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msg-tsv", required=True, help="Path to MassSpecGym.tsv")
    parser.add_argument("--out-dir", required=True, help="Output directory (e.g. data/spec_datasets/msg)")
    return parser.parse_args()


def build_spec_string(row):
    """Build an ms-pred compatible spectrum text block."""
    spec_id = row["identifier"]
    formula = row["formula"]
    precursor_mz = float(row["precursor_mz"])
    parent_mass = float(row["parent_mass"])
    adduct = row["adduct"]
    ce = row["collision_energy"]
    if pd.isna(ce):
        ce = 0.0

    mzs_str = str(row["mzs"])
    ints_str = str(row["intensities"])
    if mzs_str in ("", "nan") or ints_str in ("", "nan"):
        return None

    mzs = [float(x) for x in mzs_str.split(",")]
    ints = [float(x) for x in ints_str.split(",")]

    max_int = max(ints) if ints else 1.0
    if max_int <= 0:
        max_int = 1.0

    ce_str = str(int(ce)) if ce == int(ce) else str(ce)
    lines = [
        f">compound {spec_id}",
        f">parentmass {parent_mass:.4f}",
        f">formula {formula}",
        f">ionization {adduct}",
        "",
        f">collision {ce_str}",
    ]
    for m, i in zip(mzs, ints):
        norm_i = i / max_int
        lines.append(f"{m:.4f} {norm_i:.6f}")

    return "\n".join(lines)


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.msg_tsv} ...")
    df = pd.read_csv(args.msg_tsv, sep="\t")
    print(f"  Total rows: {len(df)}")

    # ── 1. spec_files.hdf5 ───────────────────────────────────────────────
    print("\n[1/3] Generating spec_files.hdf5 ...")
    h5_path = out_dir / "spec_files.hdf5"
    n_written = 0
    n_skipped = 0
    with h5py.File(h5_path, "w") as h5:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Writing spectra"):
            spec_id = row["identifier"]
            spec_str = build_spec_string(row)
            if spec_str is None:
                n_skipped += 1
                continue
            key = f"{spec_id}.ms"
            h5.create_dataset(key, data=np.array([spec_str.encode("utf-8")], dtype=object),
                              dtype=h5py.special_dtype(vlen=bytes))
            n_written += 1

    print(f"  Wrote {n_written} spectra to {h5_path} (skipped {n_skipped})")

    # ── 2. splits/split.tsv ──────────────────────────────────────────────
    print("\n[2/3] Generating splits/split.tsv ...")
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    fold_map = {"train": "train", "val": "val", "test": "test"}
    split_data = {"spec": [], "Fold_0": []}
    for _, row in df.iterrows():
        fold_raw = str(row["fold"]).strip().lower()
        fold = fold_map.get(fold_raw, "exclude")
        split_data["spec"].append(row["identifier"])
        split_data["Fold_0"].append(fold)

    split_df = pd.DataFrame(split_data)
    split_path = splits_dir / "split.tsv"
    split_df.to_csv(split_path, sep="\t", index=False)

    counts = split_df["Fold_0"].value_counts()
    print(f"  Split distribution:")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print(f"  Wrote to {split_path}")

    # ── 3. labels_withev.tsv ─────────────────────────────────────────────
    print("\n[3/3] Generating labels_withev.tsv ...")
    labels_rows = []
    for _, row in df.iterrows():
        ce_val = row["collision_energy"]
        if pd.isna(ce_val):
            ce_val = 0.0
        ce_str = str(int(ce_val)) if ce_val == int(ce_val) else str(ce_val)
        labels_rows.append({
            "dataset": "MassSpecGym",
            "spec": row["identifier"],
            "ionization": row["adduct"],
            "formula": row["formula"],
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "instrument": row["instrument_type"],
            "collision_energies": str([ce_str]),
            "precursor": row["precursor_mz"],
        })

    labels_df = pd.DataFrame(labels_rows)
    labels_path = out_dir / "labels_withev.tsv"
    labels_df.to_csv(labels_path, sep="\t", index=False)
    print(f"  Wrote {len(labels_df)} entries to {labels_path}")

    print(f"\nDone! Output directory: {out_dir}")
    print(f"  spec_files.hdf5:   {h5_path.stat().st_size / 1e6:.1f} MB")
    print(f"  splits/split.tsv:  {split_path.stat().st_size / 1e6:.1f} MB")
    print(f"  labels_withev.tsv: {labels_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
