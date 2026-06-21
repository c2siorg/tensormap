#!/usr/bin/env python
"""Backfill script to populate graph_ir column from existing model_configs rows.

This script is part of the Week 3 migration from model_configs KV table to
graph_ir JSON column. It reads all models where graph_ir IS NULL, reconstructs
the graph from model_configs or graph_json, converts to IRGraph, and updates
the graph_ir column.

Usage:
    python -m scripts.backfill_graph_ir [--dry-run] [--limit N]

Options:
    --dry-run: Show what would be updated without committing changes
    --limit N: Only process first N models (useful for testing)
"""

import argparse
import sys
from datetime import datetime

from sqlmodel import Session, select

from app.database import engine
from app.ir.schema import IRGraph
from app.ir.translator import reactflow_to_ir
from app.models.ml import ModelBasic, ModelConfigs
from app.services.deep_learning import _unflatten_model_configs
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def backfill_graph_ir(dry_run: bool = False, limit: int | None = None) -> dict:
    """Backfill graph_ir column for models that don't have it populated.

    Args:
        dry_run: If True, don't commit changes to database
        limit: Maximum number of models to process (None = all)

    Returns:
        Dict with statistics: {success: int, failed: int, skipped: int, failed_models: list}
    """
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "failed_models": [],
        "start_time": datetime.now(),
    }

    with Session(engine) as session:
        # Find all models with null graph_ir
        stmt = select(ModelBasic).where(ModelBasic.graph_ir == None)  # noqa: E711
        if limit:
            stmt = stmt.limit(limit)

        models = session.exec(stmt).all()
        total = len(models)

        logger.info("Found %d models with NULL graph_ir to process", total)

        for idx, model in enumerate(models, 1):
            logger.info("[%d/%d] Processing model: %s (id=%d)", idx, total, model.model_name, model.id)

            try:
                graph_ir_data = _backfill_single_model(model, session)

                if graph_ir_data is None:
                    logger.warning("Skipping model %s: no graph data available", model.model_name)
                    stats["skipped"] += 1
                    stats["failed_models"].append(
                        {"id": model.id, "name": model.model_name, "error": "No graph data available"}
                    )
                    continue

                # Validate the IRGraph before saving
                graph = IRGraph(**graph_ir_data)
                logger.debug(
                    "IRGraph validated for model %s: %d nodes, %d edges",
                    model.model_name,
                    len(graph.nodes),
                    len(graph.edges),
                )

                # Update the model
                model.graph_ir = graph_ir_data

                if not dry_run:
                    session.add(model)
                    session.commit()
                    logger.info("✓ Successfully backfilled model %s", model.model_name)
                else:
                    logger.info("✓ [DRY-RUN] Would backfill model %s", model.model_name)

                stats["success"] += 1

            except Exception as e:
                logger.error("✗ Failed to backfill model %s: %s", model.model_name, str(e), exc_info=True)
                stats["failed"] += 1
                stats["failed_models"].append({"id": model.id, "name": model.model_name, "error": str(e)})

                if not dry_run:
                    session.rollback()

    stats["end_time"] = datetime.now()
    stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()

    return stats


def _backfill_single_model(model: ModelBasic, session: Session) -> dict | None:
    """Reconstruct IRGraph for a single model from available sources.

    Priority order:
    1. graph_json (if populated) → convert to IRGraph
    2. model_configs KV table → unflatten → convert to IRGraph

    Args:
        model: The ModelBasic instance to backfill
        session: SQLModel session for querying model_configs

    Returns:
        IRGraph dict representation or None if no data available
    """
    # Try graph_json first (faster, no join needed)
    if model.graph_json is not None:
        try:
            graph = reactflow_to_ir(model.graph_json)
            return graph.model_dump()
        except Exception as e:
            logger.warning("Failed to convert graph_json for model %s: %s", model.model_name, str(e))

    # Fall back to model_configs KV table
    configs = session.exec(select(ModelConfigs).where(ModelConfigs.model_id == model.id)).all()
    if not configs:
        return None

    try:
        graph_dict = _unflatten_model_configs(configs)
        graph = reactflow_to_ir(graph_dict)
        return graph.model_dump()
    except Exception as e:
        logger.warning("Failed to reconstruct from model_configs for model %s: %s", model.model_name, str(e))
        return None


def _print_report(stats: dict, dry_run: bool = False) -> None:
    """Print a human-readable summary of the backfill operation."""
    mode = "[DRY-RUN] " if dry_run else ""
    print("\n" + "=" * 60)
    print(f"{mode}Backfill Complete")
    print("=" * 60)
    print(f"Duration:       {stats['duration_seconds']:.2f} seconds")
    print(f"Success:        {stats['success']}")
    print(f"Failed:         {stats['failed']}")
    print(f"Skipped:        {stats['skipped']}")
    print(f"Total:          {stats['success'] + stats['failed'] + stats['skipped']}")

    if stats["failed_models"]:
        print("\nFailed Models:")
        print("-" * 60)
        for failed in stats["failed_models"]:
            print(f"  • {failed['name']} (id={failed['id']})")
            print(f"    Error: {failed['error']}")

    print("=" * 60)


def main():
    """CLI entry point for backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill graph_ir column from model_configs KV table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without committing")
    parser.add_argument("--limit", type=int, help="Maximum number of models to process")

    args = parser.parse_args()

    logger.info("Starting graph_ir backfill (dry_run=%s, limit=%s)", args.dry_run, args.limit)

    try:
        stats = backfill_graph_ir(dry_run=args.dry_run, limit=args.limit)
        _print_report(stats, dry_run=args.dry_run)

        # Exit with non-zero if any failures
        sys.exit(0 if stats["failed"] == 0 else 1)

    except KeyboardInterrupt:
        logger.warning("Backfill interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Backfill failed with unexpected error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
