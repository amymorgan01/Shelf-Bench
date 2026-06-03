"""
test_all_sweeps.py — Evaluate the best W&B sweep models for Shelf-BENCH.

Output CSVs (written to ROOT_LOCAL/visualisation_panels/):
    sweep_selected_models.csv     — the selected checkpoints + val + test metrics
    sweep_test_rankings.csv       — all evaluated models ranked by test mIoU

Sweep layout expected:
    SWEEP_BASE_DIR/
        <sweep_id>/
            <architecture>/
                <architecture>_run_<run_id>_best_iou.pth
                <architecture>_run_<run_id>_best_loss.pth
"""

import gc
import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_processing.ice_data import IceDataset
from load_functions import get_loss_function, load_full_model_state, load_model
from metrics import (
    accumulate_confusion_matrix,
    accumulate_iou_components,
    calculate_iou_from_components,
    calculate_metrics_from_confusion_matrix,
)
from paths import ROOT_GWS, ROOT_LOCAL
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SWEEP_BASE_DIR = ROOT_GWS / "benchmark_data_CB" / "model_outputs" / "wandb_sweeps"

# Class 0 = Ocean, 1 = Ice 
CLASS_NAMES = ["ocean", "ice"]

ARCH_FOLDER_MAP = {
    "ViT": "ViT",
    "Unet": "Unet",
    "DeepLabV3": "DeepLabV3",
    "FPN": "FPN",
    "DinoV3": "DinoV3",
}

HPARAM_KEYS = [
    "learning_rate",
    "weight_decay",
    "batch_size",
    "loss_function",
    "optimizer",
    "epochs",
    "freeze_backbone",
    "seed",
]

VAL_IOU_KEYS  = ["val_iou", "best_val_iou", "val_mean_iou", "best_iou"]
VAL_LOSS_KEYS = ["val_loss", "best_val_loss", "min_val_loss", "best_loss"]


def round_sigfigs(value, sigfigs: int = 3):
    """Round numeric values to a fixed number of significant figures."""
    if value is None:
        return None
    try:
        return float(f"{float(value):.{sigfigs}g}")
    except (TypeError, ValueError):
        return value


# ── Hyperparameter extraction ─────────────────────────────────────────────────
def extract_hparams(checkpoint: dict) -> dict:
    out = {f"cfg_{k}": None for k in HPARAM_KEYS}
    out["cfg_architecture"] = None
    out["saved_epoch"] = None   # <-- add this

    raw = checkpoint.get("cfg", None)
    if raw is None:
        raw = checkpoint.get("config", None)
    if raw is None:
        log.debug("Checkpoint has no 'cfg' or 'config' key — hyperparams will be None")
        return out

    if hasattr(raw, "_metadata"):
        cfg_dict = OmegaConf.to_container(raw, resolve=True)
    elif isinstance(raw, dict):
        cfg_dict = raw
    else:
        log.debug(f"Unrecognised cfg type in checkpoint: {type(raw)}")
        return out

    training  = cfg_dict.get("training", {}) or {}
    model_cfg = cfg_dict.get("model", {}) or {}

    out["cfg_learning_rate"]   = round_sigfigs(training.get("learning_rate"), 3)
    out["cfg_weight_decay"]    = training.get("weight_decay")
    out["cfg_batch_size"]      = training.get("batch_size")
    out["cfg_loss_function"]   = training.get("loss_function")
    out["cfg_optimizer"]       = training.get("optimizer")
    out["cfg_epochs"]          = training.get("epochs")
    out["cfg_freeze_backbone"] = model_cfg.get("freeze_backbone")
    out["cfg_architecture"]    = model_cfg.get("name")
    out["cfg_seed"]            = cfg_dict.get("seed")

    out["saved_epoch"] = extract_saved_epoch(checkpoint) 
    return out

def extract_saved_epoch(checkpoint: dict):
    epoch = checkpoint.get("epoch", None)
    if epoch is None:
        return None
    try:
        if torch.is_tensor(epoch):
            epoch = epoch.item()
        return int(epoch)
    except (TypeError, ValueError):
        return epoch

def extract_val_metrics(checkpoint: dict) -> tuple[float | None, float | None]:
    """
    Read validation IoU and validation loss from a checkpoint dict.

    Returns (val_iou, val_loss) — either may be None if not found.
    """
    val_iou  = None
    val_loss = None

    for key in VAL_IOU_KEYS:
        if key in checkpoint:
            try:
                val_iou = float(checkpoint[key])
            except (TypeError, ValueError):
                pass
            break

    for key in VAL_LOSS_KEYS:
        if key in checkpoint:
            try:
                val_loss = float(checkpoint[key])
            except (TypeError, ValueError):
                pass
            break

    return val_iou, val_loss


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def discover_checkpoints(sweep_base: Path) -> list[dict]:
    """
    Walk SWEEP_BASE_DIR and collect _best_iou / _best_loss checkpoints.

    Returns list of dicts: sweep_id, architecture, run_id,
                           checkpoint_type, path, model_name.
    """
    checkpoints = []

    for sweep_dir in sorted(sweep_base.iterdir()):
        if not sweep_dir.is_dir():
            continue
        sweep_id = sweep_dir.name

        for arch_dir in sorted(sweep_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            arch_name = arch_dir.name
            if arch_name not in ARCH_FOLDER_MAP:
                log.debug(f"Skipping unknown architecture folder: {arch_dir}")
                continue

            for pth_file in sorted(arch_dir.glob("*.pth")):
                stem = pth_file.stem
                if stem.endswith("_best_iou"):
                    ckpt_type = "best_iou"
                elif stem.endswith("_best_loss"):
                    ckpt_type = "best_loss"
                else:
                    log.debug(f"Skipping non-standard checkpoint: {pth_file.name}")
                    continue

                parts  = stem.split("_run_")
                run_id = parts[1].rsplit("_best_", 1)[0] if len(parts) == 2 else "unknown"

                checkpoints.append({
                    "sweep_id":        sweep_id,
                    "architecture":    ARCH_FOLDER_MAP[arch_name],
                    "run_id":          run_id,
                    "checkpoint_type": ckpt_type,
                    "path":            pth_file,
                    "model_name":      stem,
                })

    log.info(f"Discovered {len(checkpoints)} checkpoints under {sweep_base}")
    return checkpoints


# ── Val-metric selection ──────────────────────────────────────────────────────

def select_best_checkpoints(checkpoints: list[dict]) -> list[dict]:
    """
    For each architecture, select:
      - the checkpoint with the highest val IoU  (selection_criterion = "val_iou")
      - the checkpoint with the lowest  val loss (selection_criterion = "val_loss")

    If both criteria point to the same file it is included once.
    Checkpoints without val metrics are de-prioritised (ranked last) but still
    considered if no better alternative exists.

    Each returned dict gains three extra keys:
        val_iou, val_loss, selection_criterion
    """
    # Group by architecture
    by_arch: dict[str, list[dict]] = {}
    for ckpt in checkpoints:
        by_arch.setdefault(ckpt["architecture"], []).append(ckpt)

    selected: list[dict] = []

    for arch, ckpts in sorted(by_arch.items()):
        log.info(f"\nSelecting best checkpoints for {arch} ({len(ckpts)} candidates)")

        # Read val metrics from every checkpoint file
        candidates = []
        for ckpt in ckpts:
            try:
                raw = torch.load(ckpt["path"], map_location="cpu", weights_only=False)
                val_iou, val_loss = extract_val_metrics(raw)
                del raw
            except Exception as e:
                log.warning(f"  Could not read {ckpt['path'].name}: {e}")
                val_iou, val_loss = None, None

            candidates.append({**ckpt, "val_iou": val_iou, "val_loss": val_loss})
            gc.collect()

        with_iou  = [c for c in candidates if c["val_iou"]  is not None]
        without_iou = [c for c in candidates if c["val_iou"] is None]
        pool_iou = with_iou if with_iou else without_iou   # fallback to full set

        best_iou_ckpt = max(
            pool_iou,
            key=lambda c: (c["val_iou"] is not None, c["val_iou"] or 0.0),
        )
        best_iou_ckpt = {**best_iou_ckpt, "selection_criterion": "val_iou"}
        selected.append(best_iou_ckpt)

        log.info(
            f"  Best val IoU  → {best_iou_ckpt['model_name']}  "
            f"(val_iou={best_iou_ckpt['val_iou']}, val_loss={best_iou_ckpt['val_loss']})"
        )

        with_loss   = [c for c in candidates if c["val_loss"] is not None]
        without_loss = [c for c in candidates if c["val_loss"] is None]
        pool_loss = with_loss if with_loss else without_loss

        best_loss_ckpt = min(
            pool_loss,
            key=lambda c: (c["val_loss"] is None, c["val_loss"] or float("inf")),
        )

        if best_loss_ckpt["path"] != best_iou_ckpt["path"]:
            best_loss_ckpt = {**best_loss_ckpt, "selection_criterion": "val_loss"}
            selected.append(best_loss_ckpt)
            log.info(
                f"  Best val loss → {best_loss_ckpt['model_name']}  "
                f"(val_iou={best_loss_ckpt['val_iou']}, val_loss={best_loss_ckpt['val_loss']})"
            )
        else:
            log.info(
                f"  Best val loss → same checkpoint as best val IoU; not duplicated."
            )

    log.info(f"\nTotal checkpoints selected for evaluation: {len(selected)}")
    return selected


def flat_per_class(values, metric: str, class_names: list[str] = CLASS_NAMES) -> dict:
    """
    Expand a per-class array into flat dict keys.

    flat_per_class([0.91, 0.87], "iou") → {"ocean_iou": 0.91, "ice_iou": 0.87}
    """
    if hasattr(values, "tolist"):
        values = values.tolist()
    return {f"{cls}_{metric}": float(values[i]) for i, cls in enumerate(class_names)}


def evaluate_single_model(
    ckpt: dict,
    test_loader,
    device,
    cfg,
    class_names: list[str] = CLASS_NAMES,
) -> dict | None:
    """
    Evaluate one checkpoint and return a flat metrics dict.

    Returns None on any unrecoverable error.
    """
    arch       = ckpt["architecture"]
    model_name = ckpt["model_name"]
    model_path = ckpt["path"]

    log.info(f"\nEvaluating [{arch}] {model_name}")
    log.info(f"  Path: {model_path}")

    try:
        # ── Load model ────────────────────────────────────────────────────────
        cfg_copy = cfg.copy()
        cfg_copy.model.name = arch
        model = load_model(cfg_copy, device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = load_full_model_state(model, checkpoint["model_state_dict"], model_name)
        model.eval()

        hparams          = extract_hparams(checkpoint)
        val_iou_saved, val_loss_saved = extract_val_metrics(checkpoint)

        num_classes   = cfg.model.classes
        loss_function = get_loss_function(cfg)

        running_loss         = 0.0
        num_batches          = 0
        total_correct_pixels = 0
        total_pixels         = 0

        confusion_matrix = {
            "tp": torch.zeros(num_classes, device=device),
            "fp": torch.zeros(num_classes, device=device),
            "fn": torch.zeros(num_classes, device=device),
            "tn": torch.zeros(num_classes, device=device),
        }
        iou_components = {
            "intersection": torch.zeros(num_classes, device=device),
            "union":        torch.zeros(num_classes, device=device),
        }

        sample_every_n = max(1, len(test_loader) // 20)
        sklearn_preds, sklearn_targets = [], []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                if batch_idx >= 100:
                    break

                images = images.to(device)
                masks  = masks.to(device)

                if masks.max() > 1:
                    masks = masks / 255.0
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                masks = masks.long()

                outputs = model(images)

                try:
                    loss = loss_function(outputs, masks)
                except Exception:
                    try:
                        masks_oh = (
                            F.one_hot(masks, num_classes=num_classes)
                            .permute(0, 3, 1, 2).float()
                        )
                        loss = loss_function(outputs, masks_oh)
                    except Exception:
                        loss = torch.tensor(0.0)

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                total_correct_pixels += (preds == masks).sum().item()
                total_pixels         += masks.numel()

                confusion_matrix = accumulate_confusion_matrix(
                    masks, preds, num_classes, confusion_matrix
                )
                iou_components = accumulate_iou_components(
                    masks, preds, num_classes, iou_components
                )

                if batch_idx % sample_every_n == 0:
                    pf = preds.view(-1).cpu().numpy()
                    mf = masks.view(-1).cpu().numpy()
                    n  = mf.shape[0]
                    if n > 10_000:
                        idx = np.random.choice(n, 10_000, replace=False)
                        pf, mf = pf[idx], mf[idx]
                    sklearn_preds.extend(pf)
                    sklearn_targets.extend(mf)
                    if len(sklearn_preds) > 200_000:
                        sklearn_preds   = sklearn_preds[-100_000:]
                        sklearn_targets = sklearn_targets[-100_000:]

                num_batches += 1

        if num_batches == 0 or total_pixels == 0:
            log.error(f"No batches processed for {model_name}")
            return None

        avg_loss       = running_loss / num_batches
        pixel_accuracy = total_correct_pixels / total_pixels

        avg_precision, avg_recall, avg_f1 = calculate_metrics_from_confusion_matrix(
            confusion_matrix, num_classes, device
        )
        class_ious, mean_iou = calculate_iou_from_components(
            iou_components, num_classes, device
        )

        # optional sklearn
        sk_acc = sk_prec = sk_rec = sk_f1 = sk_jac = None
        if sklearn_preds:
            sp = np.array(sklearn_preds)
            st = np.array(sklearn_targets)
            sk_acc  = accuracy_score(st, sp)
            sk_prec = precision_score(st, sp, average=None, zero_division=0)
            sk_rec  = recall_score(st, sp,    average=None, zero_division=0)
            sk_f1   = f1_score(st, sp,        average=None, zero_division=0)
            sk_jac  = jaccard_score(st, sp,   average=None, zero_division=0)


        row = {
            # Identity
            "sweep_id":           ckpt["sweep_id"],
            "run_id":             ckpt["run_id"],
            "architecture":       arch,
            "checkpoint_type":    ckpt["checkpoint_type"],
            "selection_criterion": ckpt.get("selection_criterion"),
            "model_name":         model_name,
            "model_path":         str(model_path),


            "val_iou_saved":  val_iou_saved,
            "val_loss_saved": val_loss_saved,

            **hparams,

            # Test-set overall metrics
            "test_loss":           avg_loss,
            "test_pixel_accuracy": pixel_accuracy,
            "test_mean_iou":       float(mean_iou),
            "test_mean_precision": float(avg_precision.mean()),
            "test_mean_recall":    float(avg_recall.mean()),
            "test_mean_f1":        float(avg_f1.mean()),

            # sklearn overall
            "test_sklearn_pixel_accuracy": sk_acc,
        }

        # Per-class flat columns
        row.update(flat_per_class(class_ious,   "test_iou",       class_names))
        row.update(flat_per_class(avg_precision, "test_precision", class_names))
        row.update(flat_per_class(avg_recall,    "test_recall",    class_names))
        row.update(flat_per_class(avg_f1,        "test_f1",        class_names))

        if sk_prec is not None:
            row.update(flat_per_class(sk_prec, "sklearn_precision", class_names))
            row.update(flat_per_class(sk_rec,  "sklearn_recall",    class_names))
            row.update(flat_per_class(sk_f1,   "sklearn_f1",        class_names))
            row.update(flat_per_class(sk_jac,  "sklearn_iou",       class_names))
        else:
            for cls in class_names:
                for m in ["sklearn_precision", "sklearn_recall", "sklearn_f1", "sklearn_iou"]:
                    row[f"{cls}_{m}"] = None

        log.info(
            f"  ✓ test mIoU: {mean_iou:.4f}  |  "
            + "  ".join(
                f"{cls.capitalize()} IoU: {float(class_ious[i]):.4f}"
                for i, cls in enumerate(class_names)
            )
        )

        del model, checkpoint, outputs, preds
        torch.cuda.empty_cache()
        gc.collect()
        return row

    except Exception as e:
        log.error(f"  ✗ Failed — {e}", exc_info=True)
        torch.cuda.empty_cache()
        gc.collect()
        return None


def save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        log.info(f"✓ {label} → {path}")
    except Exception as e:
        log.error(f"Failed to save {label}: {e}")
        fallback = path.with_suffix(".txt")
        try:
            fallback.write_text(df.to_string())
            log.info(f"  Fallback text → {fallback}")
        except Exception as e2:
            log.error(f"  Fallback also failed: {e2}")
            print(df.to_string())


def _print_section(title: str) -> None:
    print(f"\n{'='*100}")
    print(title)
    print("=" * 100)

def print_model_block(row: pd.Series, rank: int, class_names: list[str]) -> None:
    print(
        f"\n  Rank {rank}  ──  {row['architecture']}  "
        f"(selected by {row.get('selection_criterion', '?')})"
    )
    print(f"  Sweep:           {row['sweep_id']}  |  Run: {row['run_id']}")
    print(f"  Model:           {row['model_name']}")
    print(
        f"  Saved epoch:     {row.get('saved_epoch')}  |  "
        f"Config epochs: {row.get('cfg_epochs')}"
    )
    print(
        f"  Val IoU saved:   {row.get('val_iou_saved')}  |  "
        f"Val loss saved: {row.get('val_loss_saved')}"
    )

def run_sweep_testing(cfg, class_names: list[str] = CLASS_NAMES) -> None:
    """
    1. Discover all checkpoints.
    2. Select the best val-IoU and best val-loss checkpoint per architecture.
    3. Evaluate the selected set on the shared test set.
    4. Report and save results, including a test-metric ranking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    all_checkpoints = discover_checkpoints(SWEEP_BASE_DIR)
    if not all_checkpoints:
        log.error(f"No checkpoints found under {SWEEP_BASE_DIR}. Exiting.")
        return


    selected = select_best_checkpoints(all_checkpoints)

    _print_section("SELECTED CHECKPOINTS FOR EVALUATION")
    for s in selected:
        print(
            f"  [{s['architecture']:>12}]  criterion={s['selection_criterion']:<9}  "
            f"val_iou={str(s['val_iou']):<8}  val_loss={str(s['val_loss']):<10}  "
            f"{s['model_name']}"
        )

    parent_dir    = ROOT_GWS / "benchmark_data_CB" / "ICE-BENCH"
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset  = list(test_datasets.values())[0]
    log.info(f"\nTest dataset: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    log.info(f"Test batches: {len(test_loader)}")

    all_results: list[dict] = []
    failed:      list[str]  = []
    total = len(selected)

    for i, ckpt in enumerate(selected, 1):
        log.info(
            f"\n[{i}/{total}]  arch={ckpt['architecture']}  "
            f"criterion={ckpt.get('selection_criterion')}  "
            f"run={ckpt['run_id']}  sweep={ckpt['sweep_id']}"
        )
        row = evaluate_single_model(ckpt, test_loader, device, cfg, class_names)
        if row is not None:
            all_results.append(row)
        else:
            failed.append(
                f"{ckpt['sweep_id']}/{ckpt['architecture']}/{ckpt['model_name']}"
            )

    del test_loader
    torch.cuda.empty_cache()
    gc.collect()

    if not all_results:
        log.warning("No results to save — all evaluations failed.")
        return

    output_dir = Path(ROOT_LOCAL) / "visualisation_panels"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)

    final_table_cols = [
        "architecture",
        "selection_criterion",
        "sweep_id",
        "run_id",
        "saved_epoch",
        "val_iou_saved",
        "val_loss_saved",
        "test_mean_iou",
        "test_loss",
        "model_name",
        "model_path",
    ]
    final_table_cols = [c for c in final_table_cols if c in results_df.columns]

    final_models_df = (
        results_df[final_table_cols]
        .sort_values(["architecture", "selection_criterion"], ascending=[True, True])
        .reset_index(drop=True)
    )

    save_csv(
        final_models_df,
        output_dir / "sweep_final_saved_models_table.csv",
        "Final saved models table",
    )

    _print_section("FINAL SAVED MODELS TABLE")
    print(final_models_df.to_string(index=False))


    ranking_cols = (
        ["architecture", "selection_criterion", "sweep_id", "run_id",
         "val_iou_saved", "val_loss_saved",
         "test_mean_iou", "test_pixel_accuracy", "test_mean_f1", "test_loss"]
        + [f"{c}_test_iou" for c in class_names]
        + [f"{c}_test_f1"  for c in class_names]
        + ["cfg_learning_rate", "cfg_weight_decay", "cfg_batch_size",
           "cfg_loss_function", "cfg_optimizer", "cfg_epochs",
           "cfg_freeze_backbone", "model_name", "model_path"]
    )
    ranking_cols = [c for c in ranking_cols if c in results_df.columns]

    rankings_df = (
        results_df[ranking_cols]
        .sort_values("test_mean_iou", ascending=False)
        .reset_index(drop=True)
    )
    rankings_df.insert(0, "test_rank", rankings_df.index + 1)

    save_csv(rankings_df, output_dir / "sweep_test_rankings.csv", "Test rankings")


    _print_section("RESULTS BY ARCHITECTURE  (best val-IoU and best val-loss models)")

    for arch in sorted(results_df["architecture"].unique()):
        sub = results_df[results_df["architecture"] == arch].sort_values(
            "test_mean_iou", ascending=False
        )
        print(f"\n  ━━ {arch} ({'─'*60})")
        for rank_local, (_, row) in enumerate(sub.iterrows(), 1):
            print_model_block(row, rank_local, class_names)


    _print_section("OVERALL TEST RANKING  (all selected models, ranked by test mIoU)")

    for _, row in rankings_df.iterrows():
        rank = int(row["test_rank"])
        print(
            f"  #{rank:<3}  {row['architecture']:<12}  "
            f"criterion={row.get('selection_criterion', '?'):<9}  "
            f"test_mIoU={row['test_mean_iou']:.4f}  "
            f"test_F1={row['test_mean_f1']:.4f}  "
            f"px_acc={row['test_pixel_accuracy']:.4f}  "
            + "  ".join(
                f"{c}_iou={row.get(f'{c}_test_iou', float('nan')):.4f}"
                for c in class_names
            )
        )


    if failed:
        _print_section(f"FAILED CHECKPOINTS  ({len(failed)})")
        for f in failed:
            print(f"  ✗ {f}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    run_sweep_testing(cfg, class_names=CLASS_NAMES)


if __name__ == "__main__":
    main()