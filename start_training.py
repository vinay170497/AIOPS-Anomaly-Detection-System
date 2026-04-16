"""
start_training.py
------------------
Resume the pipeline from the training step.

Current state:
  [DONE]  ingest-big         1,000,000 rows in DuckDB
  [DONE]  ingest-small A     100,000 rows in DuckDB
  [DONE]  ingest-small B     100,000 labelled embeddings in ChromaDB
  [NEXT]  Training           AE1 -> IF -> AE2  (this script)

CRASH-SAFE RESUME:
  Feature matrix is saved to models/feature_matrix.npy after extraction.
  If training crashes mid-way, re-running this script skips re-extraction
  and resumes from whichever model checkpoint exists in models/.

  Checkpoints saved:
    models/scaler.pkl          after Step 1 (scaling)
    models/ae1_checkpoint.pt   best epoch during AE1 training
    models/ae2_checkpoint.pt   best epoch during AE2 training
    models/ae1.pt, ae2.pt      after full training completes

Usage:
    python start_training.py              # normal run / resume
    python start_training.py --clean      # wipe model cache, start fresh
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
        logging.FileHandler("aiops.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("StartTraining")

DEFAULT_CFG = {
    "db_path":            "aiops_logs.duckdb",
    "model_dir":          "models",
    "chroma_dir":         "chromadb_store",
    "report_dir":         "reports",
    "drain_config":       "miner_config.json",
    "drain_state":        "drain_state.bin",
    "tmp_dir":            "data",
    "chunk_size":         10_000,
    "embed_batch_size":   256,
    "ae1_epochs":         30,
    "ae2_epochs":         40,
    "train_batch_size":   512,
    "if_contamination":   0.05,
    "anomaly_percentile": 95.0,
    "knn_k":              5,
    "label_confidence":   0.6,
    "ollama_model":       "deepseek-r1:8b",
    "ollama_url":         "http://localhost:11434",
}


def inspect_resume_state(model_dir: Path) -> dict:
    """
    Check which training artefacts exist and report what will be skipped vs re-run.
    Returns a dict of what's present.
    """
    checks = {
        "feature_matrix":  (model_dir / "feature_matrix.npy").exists(),
        "feature_meta":    (model_dir / "feature_meta.pkl").exists(),
        "scaler":          (model_dir / "scaler.pkl").exists(),
        "ae1_checkpoint":  (model_dir / "ae1_checkpoint.pt").exists(),
        "ae1_final":       (model_dir / "ae1.pt").exists(),
        "ae2_checkpoint":  (model_dir / "ae2_checkpoint.pt").exists(),
        "ae2_final":       (model_dir / "ae2.pt").exists(),
        "iso_forest":      (model_dir / "isolation_forest.pkl").exists(),
        "pipeline_meta":   (model_dir / "pipeline_meta.pkl").exists(),
    }
    return checks


def clean_model_cache(model_dir: Path):
    """Remove all training artefacts so the next run starts completely fresh."""
    to_remove = [
        "feature_matrix.npy", "feature_meta.pkl",
        "scaler.pkl", "ae1_checkpoint.pt", "ae1.pt",
        "ae2_checkpoint.pt", "ae2.pt", "ae2_finetuned.pt",
        "isolation_forest.pkl", "pipeline_meta.pkl",
    ]
    removed = []
    for name in to_remove:
        p = model_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
    if removed:
        logger.info(f"Cleaned: {removed}")
    else:
        logger.info("Nothing to clean — model_dir was already empty")


def main():
    parser = argparse.ArgumentParser(
        description="Resume/run training pipeline",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete all cached model artefacts and re-run from scratch"
    )
    parser.add_argument(
        "--config", default="config.json",
        help="JSON config file (optional)"
    )
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            cfg.update(json.load(f))
        logger.info(f"Config loaded from {args.config}")

    model_dir = Path(cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle --clean flag
    if args.clean:
        logger.info("--clean flag set: wiping model cache for fresh training run")
        clean_model_cache(model_dir)

    # Inspect and report resume state
    state = inspect_resume_state(model_dir)

    print()
    print("=" * 60)
    print("  AIOps — Training Pipeline")
    print("=" * 60)
    print("  Pre-conditions:")

    # Verify DuckDB
    try:
        import duckdb
        con = duckdb.connect(cfg["db_path"], read_only=True)
        n   = con.execute("SELECT COUNT(*) FROM raw_logs").fetchone()[0]
        con.close()
        print(f"    DuckDB rows        : {n:,}  [OK]")
        if n < 100_000:
            logger.error(f"DuckDB has only {n:,} rows. Run ingestion first.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Cannot read DuckDB: {e}")
        sys.exit(1)

    # Verify ChromaDB
    try:
        from knowledge_manager import KnowledgeManager
        km    = KnowledgeManager(persist_dir=cfg["chroma_dir"])
        stats = km.get_statistics()
        print(f"    ChromaDB templates : {stats['templates']:,}  [OK]")
    except Exception as e:
        logger.warning(f"ChromaDB check failed: {e}")

    print()
    print("  Resume state (existing artefacts):")
    skip_extraction = state["feature_matrix"] and state["feature_meta"]
    print(f"    Feature matrix     : {'[CACHED - skip extraction]' if skip_extraction else '[will extract]'}")
    print(f"    Scaler             : {'[CACHED - skip fit]'        if state['scaler']          else '[will fit]'}")
    print(f"    AE1                : {'[CHECKPOINT exists]'        if state['ae1_checkpoint']  else '[will train]'}")
    print(f"    AE2                : {'[CHECKPOINT exists]'        if state['ae2_checkpoint']  else '[will train]'}")
    print(f"    Full pipeline      : {'[COMPLETE]'                 if state['pipeline_meta']   else '[pending]'}")
    print()

    if state["pipeline_meta"]:
        print("  Pipeline already fully trained!")
        print("  Use --clean to retrain from scratch, or proceed to:")
        print("    python main.py --mode status")
        print("    python main.py --mode analyse --input data/log_files.csv")
        print("    python main.py --mode chat")
        return

    print("  Training stages:")
    print("    Phase 1/4 : Feature extraction (or load cached matrix)")
    print("    Phase 2/4 : AE1 -> IF -> AE2  (GPU if available)")
    print("    Phase 3/4 : Knowledge base population")
    print("    Phase 4/4 : KNN label propagation")
    print("=" * 60)
    print()

    # Run training via main.py's run_train (single source of truth)
    from main import run_train
    run_train(cfg)

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print()
    print("  Next commands:")
    print("    python main.py --mode status")
    print('    python main.py --mode analyse --input data/log_files.csv \\')
    print('        --query "Which services have the highest error rates?"')
    print("    python main.py --mode chat")
    print()


if __name__ == "__main__":
    main()
