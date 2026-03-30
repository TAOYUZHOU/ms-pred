"""
ms-pred (ICEBERG) Smoke Test: data prep + 10-epoch gen training + prediction.
Usage:
    /root/autodl-tmp/taoyuzhou/conda_envs/ms-gen/bin/python smoke_test.py
"""
import os
import sys
import json
import logging
import warnings
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

SMOKE_DIR = Path("data/spec_datasets/smoke")
NUM_TRAIN_EPOCHS = 10

SAMPLE_MOLECULES = [
    ("SMOKE_001", "C16H17NO4", "CC(=O)NC(CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC", "[M+H]+", "Orbitrap"),
    ("SMOKE_002", "C9H8O4", "OC(=O)C1=CC=CC=C1OC(C)=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_003", "C8H9NO2", "CC(=O)NC1=CC=C(O)C=C1", "[M+H]+", "Orbitrap"),
    ("SMOKE_004", "C10H13NO2", "COC1=CC(CCN)=CC(OC)=C1", "[M+H]+", "Orbitrap"),
    ("SMOKE_005", "C6H8O6", "OCC(O)C1OC(=O)C(O)=C1O", "[M+H]+", "Orbitrap"),
    ("SMOKE_006", "C5H9NO4", "NC(CC(O)=O)C(O)=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_007", "C9H11NO3", "OC(=O)C(CC1=CC=C(O)C=C1)N", "[M+H]+", "Orbitrap"),
    ("SMOKE_008", "C10H12N2O3S", "CC1=CC2=C(C=C1C)N(CC(O)=O)C(=S)N2", "[M+H]+", "Orbitrap"),
    ("SMOKE_009", "C7H6O3", "OC(=O)C1=CC=CC=C1O", "[M+H]+", "Orbitrap"),
    ("SMOKE_010", "C12H22O11", "OCC1OC(OC2C(O)C(O)C(OC2CO)O)C(O)C(O)C1O", "[M+H]+", "Orbitrap"),
    ("SMOKE_011", "C14H18N2O5", "CCOC(=O)C(CC1=CC=C(O)C=C1)NC(=O)C(N)CC(O)=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_012", "C6H12O6", "OCC(O)C(O)C(O)C(O)C=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_013", "C4H6O3", "CC(=O)CC(O)=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_014", "C3H7NO3", "OCC(N)C(O)=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_015", "C5H11NO2S", "CSCC(N)C(O)=O", "[M+H]+", "Orbitrap"),
    ("SMOKE_016", "C9H11NO2", "NCC(=O)C1=CC=C(OC)C=C1", "[M+H]+", "Orbitrap"),
    ("SMOKE_017", "C8H8O3", "COC1=CC=C(C=O)C=C1O", "[M+H]+", "Orbitrap"),
    ("SMOKE_018", "C6H6O2", "OC1=CC=C(O)C=C1", "[M+H]+", "Orbitrap"),
    ("SMOKE_019", "C7H6O2", "OC(=O)C1=CC=CC=C1", "[M+H]+", "Orbitrap"),
    ("SMOKE_020", "C10H15N3O", "CNC(=O)C1=CN=CN1CCCC", "[M+H]+", "Orbitrap"),
]


def create_labels_tsv():
    """Create labels.tsv for the smoke dataset."""
    print("[SMOKE] Creating labels.tsv ...")
    rows = []
    for i, (spec_id, formula, smiles, ionization, instrument) in enumerate(SAMPLE_MOLECULES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        precursor_mz = Descriptors.ExactMolWt(mol) + 1.00728  # [M+H]+
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) if Chem.MolToInchi(mol) else "UNKNOWN"
        rows.append({
            "dataset": "smoke",
            "spec": f"MassSpecGymID{i:07d}",
            "ionization": ionization,
            "formula": formula,
            "smiles": Chem.MolToSmiles(mol),
            "inchikey": inchikey,
            "instrument": instrument,
            "collision_energies": "['30']",
            "precursor": f"{precursor_mz:.4f}",
        })
    df = pd.DataFrame(rows)
    labels_path = SMOKE_DIR / "labels.tsv"
    df.to_csv(labels_path, sep="\t", index=True)
    print(f"  Created {labels_path} with {len(df)} entries")
    return df


def create_spec_files_hdf5(df):
    """Create spec_files.hdf5 with spectra using real fragment masses."""
    print("[SMOKE] Creating spec_files.hdf5 (with fragment-based peaks) ...")
    from ms_pred.magma import fragmentation
    import ms_pred.common as common

    h5_path = SMOKE_DIR / "spec_files.hdf5"
    dt = h5py.special_dtype(vlen=bytes)

    with h5py.File(h5_path, "w") as f:
        for _, row in df.iterrows():
            spec_id = row["spec"]
            smiles = row["smiles"]
            formula = row["formula"]
            precursor = float(row["precursor"])
            adduct = row["ionization"]

            mol = Chem.MolFromSmiles(smiles)
            mol_wt = Descriptors.ExactMolWt(mol)

            mzs = []
            intensities = []
            try:
                fe = fragmentation.FragmentEngine(
                    mol_str=smiles, max_broken_bonds=3, max_tree_depth=2,
                )
                fe.generate_fragments()
                frag_hashes, frag_inds, shift_inds, masses, scores = fe.get_frag_masses()
                if len(masses) > 0:
                    adduct_mass = common.ion2mass.get(adduct, 1.00728)
                    frag_mzs = masses + adduct_mass
                    np.random.seed(hash(spec_id) % (2**31))
                    n_pick = min(len(frag_mzs), np.random.randint(3, 8))
                    chosen = np.random.choice(len(frag_mzs), n_pick, replace=False)
                    for idx in chosen:
                        mzs.append(frag_mzs[idx])
                        intensities.append(np.random.uniform(0.1, 1.0))
            except Exception:
                pass

            if len(mzs) < 2:
                np.random.seed(hash(spec_id) % (2**31))
                mzs = sorted(np.random.uniform(50, mol_wt * 0.8, 3).tolist())
                intensities = np.random.uniform(0.1, 1.0, 3).tolist()

            mzs.append(precursor)
            intensities.append(1.0)

            spec_lines = []
            spec_lines.append(f">compound {spec_id}")
            spec_lines.append(f">parentmass {mol_wt:.4f}")
            spec_lines.append(f">formula {formula}")
            spec_lines.append(f">ionization {adduct}")
            spec_lines.append("")
            spec_lines.append(">collision_energy 30")
            for mz, inten in zip(mzs, intensities):
                spec_lines.append(f"{mz:.4f} {inten:.6f}")

            content = "\n".join(spec_lines)
            f.create_dataset(spec_id, data=np.array([content.encode("utf-8")], dtype=object), dtype=dt)
    print(f"  Created {h5_path}")


def create_splits(df):
    """Create train/val/test split file."""
    print("[SMOKE] Creating splits ...")
    splits_dir = SMOKE_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    spec_names = df["spec"].values
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = max(int(n * 0.6), 1)
    n_val = max(int(n * 0.2), 1)

    split_data = []
    for i in indices[:n_train]:
        split_data.append({"name": spec_names[i], "split": "train"})
    for i in indices[n_train:n_train + n_val]:
        split_data.append({"name": spec_names[i], "split": "val"})
    for i in indices[n_train + n_val:]:
        split_data.append({"name": spec_names[i], "split": "test"})

    split_df = pd.DataFrame(split_data)
    split_path = splits_dir / "split_1.tsv"
    split_df.to_csv(split_path, sep="\t", index=False)
    print(f"  Created {split_path}")


def run_magma(df):
    """Run MAGMa fragmentation to create magma_tree.hdf5."""
    print("[SMOKE] Running MAGMa fragmentation ...")
    import ms_pred.common as common
    from ms_pred.magma import fragmentation
    from ms_pred.magma.run_magma import magma_augmentation

    magma_dir = SMOKE_DIR / "magma_outputs"
    magma_dir.mkdir(parents=True, exist_ok=True)

    spec_to_smiles = dict(zip(df["spec"], df["smiles"]))
    spec_to_adduct = dict(zip(df["spec"], df["ionization"]))

    magma_h5_path = magma_dir / "magma_tree.hdf5"
    spec_dir = SMOKE_DIR / "spec_files.hdf5"

    all_trees = {}
    for spec_name in df["spec"].values:
        result = magma_augmentation(
            spec_name=spec_name,
            spec_dir=spec_dir,
            spec_to_smiles=spec_to_smiles,
            spec_to_adduct=spec_to_adduct,
            max_peaks=50,
            ppm_diff=20,
            merge_specs=True,
            debug=False,
        )
        if result is not None:
            tsv_return, tree_return = result
            all_trees.update(tree_return)

    dt = h5py.special_dtype(vlen=bytes)
    with h5py.File(magma_h5_path, "w") as f:
        for tree_name, tree_data in all_trees.items():
            content = tree_data if isinstance(tree_data, str) else json.dumps(tree_data)
            f.create_dataset(tree_name, data=np.array([content.encode("utf-8")], dtype=object), dtype=dt)

    print(f"  Created {magma_h5_path} with {len(all_trees)} trees")
    return len(all_trees)


def run_training():
    """Run ICEBERG gen model training for 10 epochs."""
    print(f"[SMOKE] Training ICEBERG gen model ({NUM_TRAIN_EPOCHS} epochs) ...")
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    import ms_pred.common as common
    from ms_pred.dag_pred import dag_data, gen_model

    pl.seed_everything(42)

    dataset_name = "smoke"
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / "labels.tsv"
    split_file = data_dir / "splits" / "split_1.tsv"

    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    magma_folder = "magma_outputs"
    magma_tree_h5 = common.HDF5Dataset(data_dir / f"{magma_folder}/magma_tree.hdf5")
    name_to_json = {Path(i).stem: i for i in magma_tree_h5.get_all_names()}

    pe_embed_k = 0
    root_encode = "gnn"
    add_hs = True
    embed_elem_group = True

    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k,
        root_encode=root_encode,
        add_hs=add_hs,
        embed_elem_group=embed_elem_group,
    )

    train_dataset = dag_data.GenDataset(
        train_df, magma_h5=data_dir / f"{magma_folder}/magma_tree.hdf5",
        magma_map=name_to_json, num_workers=0, tree_processor=tree_processor,
    )
    val_dataset = dag_data.GenDataset(
        val_df, magma_h5=data_dir / f"{magma_folder}/magma_tree.hdf5",
        magma_map=name_to_json, num_workers=0, tree_processor=tree_processor,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("  WARNING: Empty dataset, skipping training. MAGMa may have failed.")
        return False

    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(train_dataset, num_workers=0, collate_fn=collate_fn, shuffle=True, batch_size=4)
    val_loader = DataLoader(val_dataset, num_workers=0, collate_fn=collate_fn, shuffle=False, batch_size=4)

    model = gen_model.FragGNN(
        hidden_size=128, layers=3, dropout=0.1, mpnn_type="GGNN",
        set_layers=0, learning_rate=1e-3, lr_decay_rate=0.9,
        weight_decay=0.0, node_feats=train_dataset.get_node_feats(),
        pe_embed_k=pe_embed_k, pool_op="avg", root_encode=root_encode,
        inject_early=False, embed_adduct=True, embed_collision=True,
        embed_instrument=True, embed_elem_group=embed_elem_group,
        encode_forms=True, add_hs=add_hs,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    save_dir = f"results/smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=tb_logger.log_dir,
        filename="best", save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=20)

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[earlystop_callback, checkpoint_callback],
        gradient_clip_val=5, min_epochs=1, max_epochs=NUM_TRAIN_EPOCHS,
        gradient_clip_algorithm="value", num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score
    print(f"  Best model: {best_path}, val_loss={best_score}")

    loaded_model = gen_model.FragGNN.load_from_checkpoint(best_path)
    loaded_model.eval()

    test_loader = DataLoader(
        val_dataset, num_workers=0, collate_fn=collate_fn, shuffle=False, batch_size=4,
    )
    trainer.test(model=loaded_model, dataloaders=test_loader)

    print("[SMOKE] Training completed successfully!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("[SMOKE] ms-pred (ICEBERG) Smoke Test")
    print("=" * 60)

    import torch
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Step 1: Create dataset
    print("[SMOKE] Step 1/3: Prepare smoke dataset")
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    df = create_labels_tsv()
    create_spec_files_hdf5(df)
    create_splits(df)
    print()

    # Step 2: Run MAGMa
    print("[SMOKE] Step 2/3: Run MAGMa fragmentation")
    n_trees = run_magma(df)
    if n_trees == 0:
        print("  ERROR: MAGMa produced 0 trees. Cannot proceed with training.")
        sys.exit(1)
    print()

    # Step 3: Train
    print("[SMOKE] Step 3/3: Train ICEBERG gen model")
    success = run_training()
    print()

    if success:
        print("=" * 60)
        print("[SMOKE] All ms-pred smoke tests PASSED!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[SMOKE] ms-pred smoke test had issues (see above)")
        print("=" * 60)
        sys.exit(1)
