"""
Production Mode - Database-to-Database Classification Pipeline.

This is the production pipeline that:
- Fetches unclassified rows directly from the database
- Classifies products in batches
- Writes results back to the database immediately
- NEVER stores data locally

Safety Features:
- Only processes rows where target column IS NULL or empty
- Only updates rows where target column IS NULL or empty (double-check)
- Idempotent: safe to re-run multiple times

Usage:
    python app/main.py

Configuration:
    config.yaml  — table, columns, labels, prompts, batch size
    .env         — credentials and infrastructure settings
"""
from __future__ import annotations

import time

import pandas as pd
from dotenv import load_dotenv

from config.exceptions import PipelineError
from config.loader import PipelineConfig, load_config
from db.db_connector import DBConnector
from services import AzureClient, Batcher, Parser, PromptBuilder, run_classification
from utils.logging import get_logger, init_logging
from utils.console import console


# -------------------- Setup -------------------- #
init_logging("main")
logger = get_logger(__name__)


def classify_batch_and_update(
    connector: DBConnector,
    df: pd.DataFrame,
    batcher: Batcher,
    cfg: PipelineConfig,
    batch_num: int,
    total_batches: int,
    row_offset: int,
) -> int:
    """Classify a batch and write results back to DB.

    Returns number of rows successfully updated.
    """
    if df.empty:
        return 0

    target_col = cfg.target.column

    # Add target column if missing
    if target_col not in df.columns:
        df[target_col] = None

    # Run classification (modifies df in place)
    run_classification(
        batcher,
        df,
        context=cfg.prompt.classification_context,
        label_names=cfg.label_names,
        label_descriptions=cfg.label_descriptions,
        target_col=target_col,
        batch_size=cfg.batch_size,
        display_column=cfg.display_column,
        partial_output_json=None,  # No local JSON in production
        show_console_start=False,
        progress_batch_offset=batch_num - 1,
        progress_total_batches=total_batches,
        display_row_offset=row_offset,
    )

    # Prepare updates for DB
    updates = []
    for _, row in df.iterrows():
        label_value = row.get(target_col)
        pk_value = row.get(cfg.source.primary_key)

        if pd.notna(label_value) and label_value != "" and pk_value is not None:
            updates.append({cfg.source.primary_key: pk_value, "label": label_value})

    if not updates:
        logger.warning("No classifications to update")
        return 0

    # Write back to DB
    updated = connector.update_classifications(
        updates,
        table=cfg.source.table,
        schema=cfg.source.schema,
        target_col=target_col,
        primary_key=cfg.source.primary_key,
    )

    return updated


def main() -> int:
    """
    Run the production pipeline.

    Fetches unclassified rows from DB, classifies them, and writes back.
    Only updates rows where target column is NULL or empty.

    Returns exit code.
    """
    pipeline_start = time.time()

    try:
        load_dotenv()
        cfg = load_config()

        logger.info("[PRODUCTION MODE] Pipeline starting")
        logger.info(
            "Config: schema=%s, table=%s, target=%s, batch_size=%d, pk=%s",
            cfg.source.schema,
            cfg.source.table,
            cfg.target.column,
            cfg.batch_size,
            cfg.source.primary_key,
        )

        console.start(
            "Production Pipeline",
            f"Connecting to [{cfg.source.schema}].[{cfg.source.table}]...",
        )

        # Initialize DB connection
        connector = DBConnector()
        connector.connect_and_verify(schema=cfg.source.schema, table=cfg.source.table)

        # Count unclassified rows
        total_unclassified = connector.count_unclassified_rows(
            table=cfg.source.table,
            schema=cfg.source.schema,
            target_col=cfg.target.column,
        )

        if total_unclassified == 0:
            logger.info("No unclassified rows found. Nothing to do.")
            console.info("Complete", "No unclassified rows found.")
            console.pipeline_finished(success=True)
            return 0

        logger.info("Found %d unclassified rows to process", total_unclassified)
        console.info("Unclassified Rows", f"{total_unclassified} rows to process")

        # Initialize LLM client
        client = AzureClient.from_env(
            system_message=cfg.prompt.system_message,
            max_rpm=cfg.max_rpm,
        )
        if client is None:
            raise PipelineError("Azure OpenAI not configured (missing env vars)")

        logger.info(
            "Azure client initialized: deployment=%s, rate_limit=%d RPM",
            client.deployment,
            cfg.max_rpm,
        )

        builder = PromptBuilder(
            context_columns=cfg.context_columns,
            instructions_template=cfg.prompt.instructions_template,
            response_key=cfg.prompt.response_key,
            fallback_label=cfg.fallback_label,
        )
        parser = Parser(response_key=cfg.prompt.response_key)
        batcher = Batcher(client=client, parser=parser, builder=builder)

        # Process in batches
        total_updated = 0
        batch_num = 0
        rows_processed = 0

        # Calculate total batches for progress
        total_batches = (total_unclassified + cfg.batch_size - 1) // cfg.batch_size
        console.classification_start(
            total_unclassified, cfg.batch_size, total_unclassified
        )

        while True:
            batch_num += 1
            batch_start = time.time()

            # Fetch batch of unclassified rows
            # Always offset=0 since we update rows (they won't appear in next query)
            df = connector.fetch_unclassified_batch(
                batch_size=cfg.batch_size,
                offset=0,
                table=cfg.source.table,
                schema=cfg.source.schema,
                target_col=cfg.target.column,
                primary_key=cfg.source.primary_key,
            )

            if df.empty:
                logger.info("No more unclassified rows. Stopping.")
                break

            logger.info("Processing batch %d: %d rows", batch_num, len(df))

            # Classify and update
            updated = classify_batch_and_update(
                connector,
                df,
                batcher,
                cfg,
                batch_num=batch_num,
                total_batches=total_batches,
                row_offset=rows_processed,
            )
            total_updated += updated
            rows_processed += len(df)

            batch_elapsed = time.time() - batch_start
            remaining = total_unclassified - total_updated

            logger.info(
                "Batch %d complete: %d updated in %.1fs. Total: %d/%d. Remaining: ~%d",
                batch_num,
                updated,
                batch_elapsed,
                total_updated,
                total_unclassified,
                remaining,
            )

            # Safety: if we updated 0 rows but had rows in batch, something is wrong
            if updated == 0 and len(df) > 0:
                logger.warning(
                    "Batch had %d rows but 0 updates. Breaking to avoid infinite loop.",
                    len(df),
                )
                break

        total_elapsed = time.time() - pipeline_start

        logger.info("[PRODUCTION MODE] Pipeline complete in %.1fs", total_elapsed)
        logger.info("Total rows updated: %d", total_updated)
        logger.info("Rate limiter stats: %s", client.rate_limiter_stats)

        console.info("Complete", f"Updated {total_updated} rows in {total_elapsed:.1f}s")
        console.pipeline_finished(success=True)
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        console.interrupted()
        return 130
    except PipelineError as e:
        logger.error("Pipeline error: %s", e)
        console.error("Pipeline Error", str(e))
        console.pipeline_finished(success=False)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        console.error("Unexpected Error", str(e))
        console.pipeline_finished(success=False)
        return 1


if __name__ == "__main__":
    exit(main())
