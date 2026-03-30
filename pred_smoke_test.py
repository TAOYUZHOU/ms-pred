"""
ms-pred ICEBERG Prediction Smoke Test:
  1. Fix magma trees (add raw_spec, collision_energy, instrument)
  2. Re-train FragGNN gen model (10 epochs)
  3. Train IntenGNN intensity model (10 epochs)
  4. Run joint prediction using gen+inten checkpoints
Usage:
    /root/autodl-tmp/taoyuzhou/conda_envs/ms-gen/bin/python pred_smoke_test.py
"""
import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SMOKE_DIR = Path("data/spec_datasets/smoke")
NUM_TRAIN_EPOCHS = 10


def fix_magma_trees():
    """Add raw_spec, collision_energy, instrument to magma trees."""
    print("[PRED-SMOKE] Fixing magma trees (adding raw_spec, collision_energy, instrument) ...")
    import ms_pred.common as common

    labels_df = pd.read_csv(SMOKE_DIR / "labels.tsv", sep="\t")
    spec_to_ce = {}
    spec_to_instrument = {}
    for _, row in labels_df.iterrows():
        ces = json.loads(row["collision_energies"].replace("'", '"'))
        ce_val = float(ces[0]) if ces else 30.0
        spec_to_ce[row["spec"]] = ce_val
        spec_to_instrument[row["spec"]] = row.get("instrument", "Orbitrap")

    spec_h5 = common.HDF5Dataset(SMOKE_DIR / "spec_files.hdf5")
    spec_names = spec_h5.get_all_names()
    spec_data = {}
    for sn in spec_names:
        raw = spec_h5.read_str(sn)
        mzs, intens = [], []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith(">") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mzs.append(float(parts[0]))
                    intens.append(float(parts[1]))
                except ValueError:
                    pass
        if mzs:
            max_i = max(intens)
            if max_i > 0:
                intens = [x / max_i for x in intens]
            spec_data[sn] = list(zip(mzs, intens))
    spec_h5.close()

    magma_h5_path = SMOKE_DIR / "magma_outputs" / "magma_tree.hdf5"
    magma_h5 = common.HDF5Dataset(magma_h5_path)
    tree_names = magma_h5.get_all_names()

    updated_trees = {}
    for tn in tree_names:
        tree = json.loads(magma_h5.read_str(tn))
        stem = Path(tn).stem
        parts = stem.split("_collision")
        spec_id = parts[0] if parts else stem

        ce = spec_to_ce.get(spec_id, 30.0)
        tree["collision_energy"] = ce
        tree["instrument"] = spec_to_instrument.get(spec_id, "Orbitrap")

        if spec_id in spec_data:
            tree["raw_spec"] = spec_data[spec_id]
        else:
            tree["raw_spec"] = [[100.0, 0.5], [200.0, 1.0]]

        new_name = f"{spec_id}_collision {ce}.json"
        updated_trees[new_name] = json.dumps(tree)
    magma_h5.close()

    dt = h5py.special_dtype(vlen=bytes)
    with h5py.File(magma_h5_path, "w") as f:
        for name, data in updated_trees.items():
            f.create_dataset(name, data=np.array([data.encode("utf-8")], dtype=object), dtype=dt)

    print(f"  Fixed {len(updated_trees)} trees in {magma_h5_path}")
    return len(updated_trees)


def fix_labels():
    """Update labels.tsv to match expected format."""
    print("[PRED-SMOKE] Fixing labels.tsv ...")
    df = pd.read_csv(SMOKE_DIR / "labels.tsv", sep="\t")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.to_csv(SMOKE_DIR / "labels.tsv", sep="\t", index=False)
    print(f"  Labels: {len(df)} entries")
    return df


def fix_splits():
    """Fix splits to use correct column names."""
    print("[PRED-SMOKE] Fixing splits ...")
    split_path = SMOKE_DIR / "splits" / "split_1.tsv"
    sdf = pd.read_csv(split_path, sep="\t")
    if "name" in sdf.columns and "spec" not in sdf.columns:
        sdf = sdf.rename(columns={"name": "spec", "split": "fold_0"})
        sdf.to_csv(split_path, sep="\t", index=False)
        print(f"  Fixed split columns -> spec, fold_0")
    elif "spec" in sdf.columns:
        print(f"  Split already has 'spec' column")
    else:
        print(f"  WARNING: Split columns: {list(sdf.columns)}")
    return sdf


def train_gen_model():
    """Train FragGNN gen model on smoke data."""
    print(f"[PRED-SMOKE] Training FragGNN gen model ({NUM_TRAIN_EPOCHS} epochs) ...")
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    import ms_pred.common as common
    from ms_pred.dag_pred import dag_data, gen_model

    pl.seed_everything(42)

    data_dir = common.get_data_dir("smoke")
    labels = data_dir / "labels.tsv"
    split_file = data_dir / "splits" / "split_1.tsv"

    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    magma_h5_path = data_dir / "magma_outputs" / "magma_tree.hdf5"
    magma_tree_h5 = common.HDF5Dataset(magma_h5_path)
    name_to_json = {Path(i).stem: i for i in magma_tree_h5.get_all_names()}

    pe_embed_k = 0
    root_encode = "gnn"
    add_hs = True
    embed_elem_group = True

    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode,
        add_hs=add_hs, embed_elem_group=embed_elem_group,
    )

    train_dataset = dag_data.GenDataset(
        train_df, magma_h5=magma_h5_path, magma_map=name_to_json,
        num_workers=0, tree_processor=tree_processor,
    )
    val_dataset = dag_data.GenDataset(
        val_df, magma_h5=magma_h5_path, magma_map=name_to_json,
        num_workers=0, tree_processor=tree_processor,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("  WARNING: Empty dataset, skipping gen training.")
        return None

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

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
    print(f"  FragGNN parameters: {num_params:,}")

    save_dir = f"results/smoke_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print(f"  Best FragGNN: {best_path}, val_loss={best_score}")
    print("[PRED-SMOKE] FragGNN training completed!")
    return best_path


def train_inten_model():
    """Train IntenGNN on the smoke dataset with raw_spec."""
    print(f"[PRED-SMOKE] Training IntenGNN ({NUM_TRAIN_EPOCHS} epochs) ...")
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    import ms_pred.common as common
    from ms_pred.dag_pred import dag_data, inten_model

    pl.seed_everything(42)

    data_dir = common.get_data_dir("smoke")
    labels = data_dir / "labels.tsv"
    split_file = data_dir / "splits" / "split_1.tsv"

    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]

    magma_h5_path = data_dir / "magma_outputs" / "magma_tree.hdf5"
    magma_tree_h5 = common.HDF5Dataset(magma_h5_path)
    name_to_json = {Path(i).stem: i for i in magma_tree_h5.get_all_names()}

    pe_embed_k = 0
    root_encode = "gnn"
    add_hs = True
    embed_elem_group = True
    binned_targs = True

    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=pe_embed_k, root_encode=root_encode,
        binned_targs=binned_targs, add_hs=add_hs,
        embed_elem_group=embed_elem_group,
    )

    train_dataset = dag_data.IntenDataset(
        train_df, magma_h5=magma_h5_path, magma_map=name_to_json,
        num_workers=0, tree_processor=tree_processor,
    )
    val_dataset = dag_data.IntenDataset(
        val_df, magma_h5=magma_h5_path, magma_map=name_to_json,
        num_workers=0, tree_processor=tree_processor,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("  WARNING: Empty dataset, cannot train IntenGNN.")
        return None

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(train_dataset, num_workers=0, collate_fn=collate_fn, shuffle=True, batch_size=4)
    val_loader = DataLoader(val_dataset, num_workers=0, collate_fn=collate_fn, shuffle=False, batch_size=4)

    model = inten_model.IntenGNN(
        hidden_size=128, gnn_layers=3, mlp_layers=2,
        set_layers=1, frag_set_layers=2,
        dropout=0.1, mpnn_type="GGNN",
        learning_rate=7e-4, lr_decay_rate=0.9,
        weight_decay=0.0,
        node_feats=train_dataset.get_node_feats(),
        pe_embed_k=pe_embed_k, pool_op="avg",
        loss_fn="cosine", root_encode=root_encode,
        inject_early=False, embed_adduct=True,
        embed_collision=True, embed_instrument=True,
        embed_elem_group=embed_elem_group,
        include_unshifted_mz=True,
        binned_targs=binned_targs,
        encode_forms=True, add_hs=add_hs,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  IntenGNN parameters: {num_params:,}")

    save_dir = f"results/smoke_inten_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print(f"  Best IntenGNN: {best_path}, val_loss={best_score}")
    print("[PRED-SMOKE] IntenGNN training completed!")
    return best_path


def run_joint_prediction(gen_ckpt, inten_ckpt):
    """Run joint ICEBERG prediction with gen + inten checkpoints."""
    print("[PRED-SMOKE] Running joint ICEBERG prediction ...")
    import torch
    import ms_pred.common as common
    from ms_pred.dag_pred import gen_model, inten_model, joint_model

    gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_ckpt, map_location="cpu")
    inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_ckpt, map_location="cpu")

    model = joint_model.JointModel(
        gen_model_obj=gen_model_obj,
        inten_model_obj=inten_model_obj,
    )
    model.eval()
    model.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_molecules = [
        ("OC(=O)C1=CC=CC=C1OC(C)=O", "[M+H]+", "Orbitrap", 181.0501, 30.0),
        ("CC(=O)NC1=CC=C(O)C=C1", "[M+H]+", "Orbitrap", 152.0712, 30.0),
        ("OC(=O)C1=CC=CC=C1O", "[M+H]+", "Orbitrap", 139.0395, 30.0),
    ]

    print(f"  Predicting spectra for {len(test_molecules)} molecules ...")
    save_dir = Path(f"results/smoke_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for smi, adduct, instrument, precursor_mz, ce in test_molecules:
        try:
            out = model.predict_mol(
                smi=smi,
                collision_eng=ce,
                precursor_mz=precursor_mz,
                adduct=adduct,
                instrument=instrument,
                threshold=0.0,
                device=device,
                max_nodes=100,
                binned_out=False,
                adduct_shift=True,
            )
            if "spec" in out:
                spec = out["spec"]
                n_peaks = spec.shape[0] if hasattr(spec, 'shape') else len(spec)
                results.append({
                    "smiles": smi,
                    "adduct": adduct,
                    "n_peaks": n_peaks,
                    "status": "OK",
                })
                print(f"    {smi[:40]:40s} -> {n_peaks} peaks predicted")
            else:
                results.append({"smiles": smi, "adduct": adduct, "n_peaks": 0, "status": "no_spec"})
                print(f"    {smi[:40]:40s} -> no spec key in output")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"smiles": smi, "adduct": adduct, "n_peaks": 0, "status": f"error: {e}"})
            print(f"    {smi[:40]:40s} -> ERROR: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "prediction_results.csv", index=False)
    print(f"  Results saved to {save_dir / 'prediction_results.csv'}")

    n_success = sum(1 for r in results if r["status"] == "OK")
    print(f"  Success: {n_success}/{len(results)}")
    return n_success > 0


if __name__ == "__main__":
    print("=" * 60)
    print("[PRED-SMOKE] ms-pred ICEBERG Prediction Smoke Test")
    print("=" * 60)

    import torch
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Step 0: Fix data
    print("[PRED-SMOKE] Step 0: Fix data (labels, splits, magma trees)")
    fix_labels()
    fix_splits()
    n_trees = fix_magma_trees()
    if n_trees == 0:
        print("ERROR: No magma trees after fixing")
        sys.exit(1)
    print()

    # Check for existing checkpoints (skip training if available)
    import glob
    existing_gen = sorted(glob.glob("results/smoke_gen_*/version_0/best.ckpt"))
    existing_inten = sorted(glob.glob("results/smoke_inten_*/version_0/best.ckpt"))

    if existing_gen:
        gen_ckpt = existing_gen[-1]
        print(f"[PRED-SMOKE] Reusing existing gen checkpoint: {gen_ckpt}")
    else:
        print("[PRED-SMOKE] Step 1/3: Train FragGNN gen model")
        gen_ckpt = train_gen_model()
        if gen_ckpt is None:
            print("ERROR: Failed to train FragGNN")
            sys.exit(1)
    print()

    if existing_inten:
        inten_ckpt = existing_inten[-1]
        print(f"[PRED-SMOKE] Reusing existing inten checkpoint: {inten_ckpt}")
    else:
        print("[PRED-SMOKE] Step 2/3: Train IntenGNN intensity model")
        inten_ckpt = train_inten_model()
        if inten_ckpt is None:
            print("ERROR: Failed to train IntenGNN")
            sys.exit(1)
    print()

    # Step 3: Run joint prediction
    print("[PRED-SMOKE] Step 3/3: Run Joint Prediction")
    success = run_joint_prediction(gen_ckpt, inten_ckpt)
    print()

    if success:
        print("=" * 60)
        print("[PRED-SMOKE] ICEBERG Prediction Smoke Test PASSED!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[PRED-SMOKE] ICEBERG Prediction Smoke Test FAILED!")
        print("=" * 60)
        sys.exit(1)
