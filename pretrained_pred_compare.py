"""
Compare ICEBERG prediction: pretrained (Dropbox) vs smoke-trained (10 epochs).
Uses the same 3 test molecules from pred_smoke_test.py.
"""
import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MSPRED_ROOT = Path("/root/autodl-tmp/taoyuzhou/ms-pred")
PRETRAINED_GEN = MSPRED_ROOT / "pretrained_weights" / "iceberg_dag_gen_msg_best.ckpt"
PRETRAINED_INTEN = MSPRED_ROOT / "pretrained_weights" / "iceberg_dag_inten_msg_best.ckpt"

import glob
SMOKE_GEN_CKPTS = sorted(glob.glob(str(MSPRED_ROOT / "results" / "smoke_gen_*" / "version_0" / "best.ckpt")))
SMOKE_INTEN_CKPTS = sorted(glob.glob(str(MSPRED_ROOT / "results" / "smoke_inten_*" / "version_0" / "best.ckpt")))

TEST_MOLECULES = [
    {"smiles": "OC(=O)C1=CC=CC=C1OC(C)=O", "name": "Aspirin", "adduct": "[M+H]+",
     "instrument": "Orbitrap", "precursor_mz": 181.0501, "ce": 30.0},
    {"smiles": "CC(=O)NC1=CC=C(O)C=C1", "name": "Acetaminophen", "adduct": "[M+H]+",
     "instrument": "Orbitrap", "precursor_mz": 152.0712, "ce": 30.0},
    {"smiles": "OC(=O)C1=CC=CC=C1O", "name": "Salicylic acid", "adduct": "[M+H]+",
     "instrument": "Orbitrap", "precursor_mz": 139.0395, "ce": 30.0},
]

OUTPUT_DIR = MSPRED_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_predict(gen_ckpt, inten_ckpt, label="model"):
    from ms_pred.dag_pred import gen_model, inten_model, joint_model

    print(f"\n{'='*60}")
    print(f"Loading {label} ...")
    print(f"  gen_ckpt:   {gen_ckpt}")
    print(f"  inten_ckpt: {inten_ckpt}")

    gen_obj = gen_model.FragGNN.load_from_checkpoint(str(gen_ckpt), map_location="cpu")
    inten_obj = inten_model.IntenGNN.load_from_checkpoint(str(inten_ckpt), map_location="cpu")

    gen_params = sum(p.numel() for p in gen_obj.parameters())
    inten_params = sum(p.numel() for p in inten_obj.parameters())
    print(f"  FragGNN params:  {gen_params:,}")
    print(f"  IntenGNN params: {inten_params:,}")

    model = joint_model.JointModel(gen_model_obj=gen_obj, inten_model_obj=inten_obj)
    model.eval()
    model.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = []
    for mol in TEST_MOLECULES:
        try:
            out = model.predict_mol(
                smi=mol["smiles"], collision_eng=mol["ce"],
                precursor_mz=mol["precursor_mz"], adduct=mol["adduct"],
                instrument=mol["instrument"], threshold=0.0,
                device=device, max_nodes=100,
                binned_out=False, adduct_shift=True,
            )
            if "spec" in out:
                spec = out["spec"]
                n_peaks = spec.shape[0] if hasattr(spec, 'shape') else len(spec)
                if hasattr(spec, 'shape') and len(spec.shape) == 2:
                    mzs = spec[:, 0].tolist() if spec.shape[0] > 0 else []
                    ints = spec[:, 1].tolist() if spec.shape[0] > 0 else []
                else:
                    mzs, ints = [], []

                top5_idx = np.argsort(ints)[-5:][::-1] if len(ints) > 0 else []
                top5 = [(round(mzs[i], 4), round(ints[i], 4)) for i in top5_idx]

                results.append({
                    "name": mol["name"], "smiles": mol["smiles"],
                    "n_peaks": n_peaks, "status": "OK",
                    "top5_peaks": top5,
                    "total_intensity": round(sum(ints), 4) if ints else 0,
                    "max_mz": round(max(mzs), 4) if mzs else 0,
                    "min_mz": round(min(mzs), 4) if mzs else 0,
                    "full_spec": list(zip([round(m, 4) for m in mzs], [round(i, 6) for i in ints])),
                })
                print(f"  {mol['name']:20s} -> {n_peaks:5d} peaks | top m/z: {top5[:3]}")
            else:
                results.append({"name": mol["name"], "smiles": mol["smiles"],
                                "n_peaks": 0, "status": "no_spec", "top5_peaks": [], "full_spec": []})
                print(f"  {mol['name']:20s} -> no spec")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"name": mol["name"], "smiles": mol["smiles"],
                            "n_peaks": 0, "status": f"error: {e}", "top5_peaks": [], "full_spec": []})
            print(f"  {mol['name']:20s} -> ERROR: {e}")

    del model, gen_obj, inten_obj
    torch.cuda.empty_cache()
    return results, gen_params, inten_params


def compare_and_report(pretrained_results, smoke_results, pretrained_params, smoke_params):
    print(f"\n{'='*60}")
    print("COMPARISON: Pretrained (MassSpecGym) vs Smoke-trained (10 epochs)")
    print(f"{'='*60}\n")

    print(f"{'':20s} | {'Pretrained':>20s} | {'Smoke-trained':>20s}")
    print(f"{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    print(f"{'FragGNN params':20s} | {pretrained_params[0]:>20,} | {smoke_params[0]:>20,}")
    print(f"{'IntenGNN params':20s} | {pretrained_params[1]:>20,} | {smoke_params[1]:>20,}")
    print()

    rows = []
    for pr, sr in zip(pretrained_results, smoke_results):
        name = pr["name"]
        print(f"\n--- {name} ({pr['smiles']}) ---")
        print(f"  {'Metric':20s} | {'Pretrained':>15s} | {'Smoke':>15s}")
        print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*15}")
        print(f"  {'Peaks predicted':20s} | {pr['n_peaks']:>15d} | {sr['n_peaks']:>15d}")
        print(f"  {'Status':20s} | {pr['status']:>15s} | {sr['status']:>15s}")
        print(f"  {'Total intensity':20s} | {pr.get('total_intensity',0):>15.4f} | {sr.get('total_intensity',0):>15.4f}")
        print(f"  {'m/z range':20s} | {pr.get('min_mz',0):.1f}-{pr.get('max_mz',0):.1f} | {sr.get('min_mz',0):.1f}-{sr.get('max_mz',0):.1f}")

        if pr.get("top5_peaks"):
            print(f"\n  Top-5 peaks (pretrained):")
            for mz, inten in pr["top5_peaks"][:5]:
                print(f"    m/z={mz:>10.4f}  intensity={inten:.6f}")
        if sr.get("top5_peaks"):
            print(f"  Top-5 peaks (smoke-trained):")
            for mz, inten in sr["top5_peaks"][:5]:
                print(f"    m/z={mz:>10.4f}  intensity={inten:.6f}")

        rows.append({
            "molecule": name, "smiles": pr["smiles"],
            "pretrained_peaks": pr["n_peaks"], "smoke_peaks": sr["n_peaks"],
            "pretrained_status": pr["status"], "smoke_status": sr["status"],
            "pretrained_total_int": pr.get("total_intensity", 0),
            "smoke_total_int": sr.get("total_intensity", 0),
        })

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "pretrained_vs_smoke_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nComparison saved to: {out_csv}")

    # Save detailed spectra
    for label, results in [("pretrained", pretrained_results), ("smoke", smoke_results)]:
        for r in results:
            spec_file = OUTPUT_DIR / f"{label}_{r['name'].lower().replace(' ','_')}_spectrum.json"
            with open(spec_file, "w") as f:
                json.dump({"name": r["name"], "smiles": r["smiles"],
                          "n_peaks": r["n_peaks"], "spectrum": r.get("full_spec", [])}, f, indent=2)

    return df


if __name__ == "__main__":
    print("="*60)
    print("ICEBERG Prediction: Pretrained vs Smoke-trained Comparison")
    print("="*60)
    print(f"Device: {'cuda (' + torch.cuda.get_device_name() + ')' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")

    # 1. Pretrained model prediction
    pretrained_results, pt_gen_p, pt_inten_p = load_and_predict(
        PRETRAINED_GEN, PRETRAINED_INTEN, label="Pretrained (MassSpecGym)")

    # 2. Smoke-trained model prediction
    if SMOKE_GEN_CKPTS and SMOKE_INTEN_CKPTS:
        smoke_results, sm_gen_p, sm_inten_p = load_and_predict(
            SMOKE_GEN_CKPTS[-1], SMOKE_INTEN_CKPTS[-1], label="Smoke-trained (10 epochs)")
    else:
        print("\nWARNING: No smoke-trained checkpoints found, skipping comparison.")
        smoke_results = [{"name": m["name"], "smiles": m["smiles"], "n_peaks": 0,
                          "status": "no_ckpt", "top5_peaks": [], "full_spec": []}
                         for m in TEST_MOLECULES]
        sm_gen_p, sm_inten_p = 0, 0

    # 3. Compare
    compare_and_report(
        pretrained_results, smoke_results,
        (pt_gen_p, pt_inten_p), (sm_gen_p, sm_inten_p)
    )

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
