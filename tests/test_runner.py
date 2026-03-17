"""
Test Mode Runner - Local JSON-based classification pipeline.

This replicates the original pipeline behavior:
- Downloads data from DB to local JSON
- Runs classification on local DataFrame
- Outputs results to local JSON
- NEVER writes back to the database

Usage:
    python tests/test_runner.py

Configuration:
    config.yaml      — table, columns, labels, prompts, batch size
    .env             — credentials only
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

# Resolve project root (../..)
# This ensures we work correctly regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add app directory to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "app"))

try:
    import pandas as pd
except ImportError as e:
    print(f"\n[ERROR] Failed to import pandas: {e}")
    print(f"Please check your Python environment (Python {sys.version.split()[0]}).\n")
    sys.exit(1)

from dotenv import load_dotenv

from config.exceptions import PipelineError
from config.loader import PipelineConfig, load_config
from db.db_connector import DBConnector
from helpers.data_operations import (
    JsonManager,
    validate_classification_output,
)
from services import AzureClient, Batcher, Parser, PromptBuilder, run_classification
from utils.logging import get_logger, init_logging
from utils.console import console


# Test mode data directories (located in project root's data folder)
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test_run"
TEST_INPUT_JSON = TEST_DATA_DIR / "input.json"
TEST_OUTPUT_JSON = TEST_DATA_DIR / "output.json"


# Setup logging
init_logging("test")
logger = get_logger(__name__)


def export_data(
    connector: DBConnector, cfg: PipelineConfig, row_limit: int
) -> tuple[int, int, float]:
    """Export raw data from SQL table into JSON. Returns (rows, cols, elapsed)."""
    TEST_INPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    target_col = cfg.target.column

    # Fetch data directly as DataFrame
    # Note: We fetch 'top' rows regardless of status, then clear the target column locally
    # to simulate a fresh classification run on this sample.
    df = connector.fetch_table(
        table=cfg.source.table, schema=cfg.source.schema, top=row_limit
    )

    # Force target column to be empty for re-classification
    if target_col in df.columns:
        df[target_col] = None
        logger.info("Cleared updated values in '%s'", target_col)
    else:
        df[target_col] = None

    # Save to JSON (Test Mode specific logic)
    df.to_json(TEST_INPUT_JSON, orient="records", indent=2, force_ascii=False)

    rows, cols = df.shape
    elapsed = time.time() - start_time
    logger.info(
        "Exported to %s: %d rows, %d cols in %.1fs",
        TEST_INPUT_JSON,
        rows,
        cols,
        elapsed,
    )
    return rows, cols, elapsed


def load_json_data(json_file: Path, target_col: str) -> pd.DataFrame:
    """Load JSON into DataFrame and add target column if needed."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    logger.debug("Loaded %d rows from %s", len(data), json_file)
    df = pd.DataFrame(data)

    # Add target column if it doesn't exist
    if target_col not in df.columns:
        df[target_col] = None

    logger.info("Loaded %d rows with target column '%s'", len(df), target_col)
    return df


def classify_data(df: pd.DataFrame, cfg: PipelineConfig) -> float:
    """Run classification and save output. Returns elapsed time."""
    start_time = time.time()
    jm = JsonManager()
    target_col = cfg.target.column

    def _on_interrupt(signum, frame):
        try:
            jm.write(TEST_OUTPUT_JSON, df)
            logger.info("Partial data saved on interrupt to %s", TEST_OUTPUT_JSON)
        except Exception as e:
            logger.warning("Failed to save on interrupt: %s", e)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _on_interrupt)

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

    run_classification(
        batcher,
        df,
        context=cfg.prompt.classification_context,
        label_names=cfg.label_names,
        label_descriptions=cfg.label_descriptions,
        target_col=target_col,
        batch_size=cfg.batch_size,
        display_column=cfg.display_column,
        partial_output_json=str(TEST_OUTPUT_JSON),
    )

    elapsed = time.time() - start_time
    logger.info("Classification complete in %.1fs", elapsed)

    # Get validation stats for console summary
    stats = validate_classification_output(
        df,
        target_col=target_col,
        expected_options=cfg.label_names,
        as_dict=True,
    )

    if stats:
        console.classification_summary(
            total_rows=stats["total_rows"],
            classified=stats["classified_rows"],
            unique_categories=stats["unique_categories"],
            total_categories=len(cfg.label_names),
            top_categories=stats["top_frequencies"],
            unexpected=stats.get("unexpected_values", []),
            output_path=str(TEST_OUTPUT_JSON),
            elapsed=elapsed,
        )

    jm.write(TEST_OUTPUT_JSON, df)
    logger.info("Output saved to %s", TEST_OUTPUT_JSON)
    return elapsed


def run_test_mode() -> int:
    """
    Run the test mode pipeline.

    This is the original pipeline behavior:
    - Downloads data from DB to local JSON
    - Classifies products locally
    - Saves results to local JSON
    - NEVER writes to database

    Returns exit code.
    """
    pipeline_start = time.time()

    try:
        load_dotenv()
        cfg = load_config()
        row_limit = cfg.test_row_limit

        logger.info(
            "[TEST MODE] Pipeline starting with config: schema=%s, table=%s, "
            "limit=%d, batch_size=%d",
            cfg.source.schema,
            cfg.source.table,
            row_limit,
            cfg.batch_size,
        )

        console.start(
            "Test Mode Pipeline",
            f"Connecting to [{cfg.source.schema}].[{cfg.source.table}]...",
        )

        connector = DBConnector()
        connector.connect_and_verify(schema=cfg.source.schema, table=cfg.source.table)

        rows, cols, export_time = export_data(connector, cfg, row_limit)
        console.data_loaded(
            source=f"[{cfg.source.schema}].[{cfg.source.table}]",
            rows=rows,
            columns=cols,
            elapsed=export_time,
        )

        df = load_json_data(TEST_INPUT_JSON, target_col=cfg.target.column)
        classify_data(df, cfg)

        total_elapsed = time.time() - pipeline_start
        logger.info("[TEST MODE] Pipeline finished successfully in %.1fs", total_elapsed)
        logger.info("[TEST MODE] Output saved to: %s", TEST_OUTPUT_JSON)
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
    exit(run_test_mode())
