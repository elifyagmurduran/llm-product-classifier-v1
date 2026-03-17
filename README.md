# llm-classifies-data

A production-ready Python pipeline that uses Azure OpenAI to automatically classify rows in an Azure SQL database — with no code changes required between use cases.

You describe the classification task entirely in `config.yaml`: which table to read, which columns to use as evidence, what labels are valid, and what the LLM prompt says. The pipeline handles everything else — fetching data, batching it, calling the LLM, parsing responses, and writing results back to the database.

> **New here?** Start with [Getting Started](#getting-started). You can be running a classification in under 10 minutes once credentials are in hand.

---

## Table of Contents

- [What This Does](#what-this-does)
- [What Was Built](#what-was-built)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone and install](#1-clone-and-install)
  - [2. Create your `.env` file](#2-create-your-env-file)
  - [3. Configure `config.yaml`](#3-configure-configyaml)
  - [4. Run in test mode first](#4-run-in-test-mode-first)
  - [5. Run in production](#5-run-in-production)
- [Configuration Reference](#configuration-reference)
- [Running the Pipeline](#running-the-pipeline)
- [Adapting to a New Use Case](#adapting-to-a-new-use-case)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Key Features](#key-features)

---

## What This Does

Many datasets contain rows that need to be labelled — products by category, tickets by topic, records by status. Doing this manually is slow and inconsistent. Writing a hard-coded script means touching code every time the table, columns, or category list change.

This pipeline solves that by making the entire classification task **data-driven**:

1. You write a `config.yaml` describing the task (table, columns, labels, prompt wording).
2. You run the pipeline.
3. The LLM reads each row, picks the best label from your list, and the result is written back to the database.

No Python code changes required — ever. Swap `config.yaml` to classify a completely different table with different labels.

---

## What Was Built

This project was built from scratch in Python and demonstrates several production engineering practices:

### Technologies used

| Technology | Role |
|---|---|
| **Python 3.11+** | Core language |
| **Azure OpenAI (GPT)** | LLM classification engine — called directly via the REST API using `requests` |
| **Azure SQL Database** | Source and target data store — connected via `pyodbc` and `SQLAlchemy` |
| **pandas** | In-memory DataFrame operations: batching, label assignment, validation |
| **PyYAML** | Parsing and validating `config.yaml` at startup |
| **python-dotenv** | Loading credentials from `.env` without hardcoding secrets |
| **azure-identity** | Azure service principal authentication for SQL |

### Engineering highlights

- **Config-driven design** — `PipelineConfig` is a validated Python dataclass built from YAML. Every behavioural concern lives in one file; the code never needs to change.
- **Layered architecture** — the config, database, and LLM layers are fully decoupled. Neither layer knows about the others; they communicate only through the orchestrator.
- **Stateless, idempotent operation** — the pipeline only operates on rows where the target column is `NULL`. It is safe to stop and re-start at any point without duplicating work.
- **Two run modes** — a safe `test_runner.py` mode that never writes to the database, and a production `main.py` mode that writes results back immediately after each batch.
- **Batch checkpointing** — in test mode, progress is saved to `output.json` after every batch. A `Ctrl+C` interrupt saves whatever has been classified so far.
- **Startup validation** — `config.yaml` is fully validated before any database or API calls are made. Misconfigured YAML produces a clear error message at startup, not mid-run.
- **Structured logging** — all run activity is written to timestamped files in `logs/` alongside live terminal output.
- **Prompt templating** — prompts are assembled dynamically from config values. Label descriptions, context columns, fallback behaviour, and response format are all controlled from `config.yaml`.

---

## How It Works

```
config.yaml
    │
    ▼
Load & validate PipelineConfig
    │
    ├─→ Connect to Azure SQL
    │       └─→ Count / fetch unclassified rows (WHERE target IS NULL)
    │
    ├─→ Build LLM prompt from context columns + label descriptions
    │
    ├─→ Batch rows → send to Azure OpenAI → parse JSON response
    │       └─→ Assign label to each row in DataFrame
    │
    └─→ [Production] Write labels back to DB immediately per batch
        [Test Mode]  Save results to data/test_run/output.json
```

The pipeline loops until no unclassified rows remain. It is **idempotent** — already-classified rows are never re-processed.

---

## Getting Started

### Prerequisites

Before you begin, make sure you have:

- **Python 3.11 or later** — check with `python --version`
- **ODBC Driver 18 for SQL Server** — required to connect to Azure SQL
  - Windows: [Download from Microsoft](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
  - macOS: `brew install microsoft/mssql-release/msodbcsql18`
  - Linux: [Microsoft's installation guide](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)
- **An Azure SQl Database** with a table you want to classify
- **An Azure OpenAI resource** with a GPT model deployed (e.g. `gpt-4o`)

---

### 1. Clone and install

```bash
git clone https://github.com/your-org/llm-classifies-data.git
cd llm-classifies-data

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Create your `.env` file

The `.env` file holds your credentials. It is gitignored and should **never** be committed.

Create a file named `.env` in the project root with the following content, replacing the placeholder values:

```env
# ── Azure SQL ──────────────────────────────────────────────
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your-database-name
AZURE_SQL_CLIENT_ID=your-service-principal-client-id
AZURE_SQL_CLIENT_SECRET=your-service-principal-secret
AZURE_SQL_TIMEOUT=30

# ── Azure OpenAI ───────────────────────────────────────────
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-deployment-name        # e.g. gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

```

Where to find these values:
- **Azure SQL** credentials — your Azure service principal details from the Azure portal or your DBA
- **Azure OpenAI** — from your Azure OpenAI resource under *Keys and Endpoint*, and the deployment name from *Model deployments*

> The `.env` file is for **credentials and infrastructure only**. Table names, columns, labels, batch size, and row limits are all set in `config.yaml`.

---

### 3. Configure `config.yaml`

Open `config.yaml` in the project root and edit it to describe your classification task. This is the **only file you need to change** between use cases.

The minimum things to set:
- `source.schema` and `source.table` — point at your Azure SQL table
- `source.primary_key` — the column used to identify rows when writing results back
- `target.column` — the column that will be populated with labels (must already exist in the table, or will be created locally in test mode)
- `context_columns` — which columns the LLM should read when classifying each row
- `labels` — the valid labels you want assigned, with optional descriptions

See the full [Configuration Reference](#configuration-reference) below for every option.

---

### 4. Run in test mode first

**Always run test mode before production.** Test mode reads from the database but **never writes back** — results are saved locally to `data/test_run/output.json`.

```bash
python tests/test_runner.py
```

Open `data/test_run/output.json` to inspect the assigned labels. If the results look wrong, adjust your label descriptions or prompt wording in `config.yaml` and run again.

---

### 5. Run in production

Once you are satisfied with the test results:

```bash
python app/main.py
```

The pipeline will classify all `NULL` rows and write the labels back to the database. Progress is logged to the terminal and to `logs/`.

---

## Configuration Reference

All pipeline behaviour is controlled by `config.yaml`. Nothing in the Python code needs to change when you adapt the pipeline to a new table or taxonomy.

```yaml
# ── Source Table ───────────────────────────────────────────
source:
  schema: "dbo"               # SQL schema
  table: "my_products"        # Table to read from
  primary_key: "id"           # Primary key column (used for writing results back)

# ── Classification Target ──────────────────────────────────
target:
  column: "product_category"  # Column to populate with labels (NULL rows are processed)

# ── Context Columns ────────────────────────────────────────
# These columns are sent to the LLM as evidence for each classification.
# They must exist in the source table.
context_columns:
  - "product_name"
  - "brand"
  - "description"

# ── Display Column ─────────────────────────────────────────
# Column used for human-readable console/log output during a run.
# Falls back to the primary key if not present.
display_column: "product_name"

# ── Labels ─────────────────────────────────────────────────
# The valid labels the LLM can assign. Each has an optional description
# included in the prompt — more detail generally means better accuracy.
labels:
  - name: "Electronics"
    description: >
      Consumer electronics such as phones, tablets, laptops, TVs,
      headphones, cameras, and related accessories.
  - name: "Food & Beverages"
    description: >
      Edible products including snacks, fresh produce, dairy,
      frozen meals, soft drinks, and water.
  - name: "Other"
    description: >
      Products that do not clearly fit into any of the above categories.
      Use this ONLY when you cannot confidently assign the product.

# ── Fallback Label ─────────────────────────────────────────
# The label the LLM is instructed to use when it cannot classify confidently.
# Must exactly match one of the label names above.
fallback_label: "Other"

# ── Prompt Configuration ───────────────────────────────────
prompt:
  # The system role message sent to the LLM at the start of each request.
  system_message: >
    You are a precise product classification assistant...

  # General framing included at the top of every batch prompt.
  classification_context: >
    Classify each product into ONE of the available labels...

  # Template for per-batch instructions. Three placeholders are required:
  #   {columns}        — substituted with the context_columns list
  #   {fallback_label} — substituted with the fallback_label value
  #   {response_key}   — substituted with the response_key value below
  instructions_template: >
    Use ONLY these columns as context: {columns}
    ...
    If unsure, use '{fallback_label}'.
    Return ONLY a JSON array: [{"row_id": int, "{response_key}": string}]

  # The JSON key the LLM uses in its response for the assigned label.
  # Must match the placeholder used in instructions_template.
  response_key: "category"

# ── Pipeline Settings ──────────────────────────────────────
settings:
  batch_size: 10              # Rows sent to the LLM per API call
  max_rpm: 30                 # Max LLM requests per minute (rate limiter)
```

### Config validation

The pipeline validates `config.yaml` at startup before making any database or API calls:

- `fallback_label` must exactly match a name in `labels`
- `context_columns` must not be empty
- `instructions_template` must contain the `{columns}`, `{fallback_label}`, and `{response_key}` placeholders
- All required sections (`source`, `target`, `labels`, `prompt`, `settings`) must be present

If anything is misconfigured, the pipeline exits immediately with a clear error message.

### Overriding the config path

By default the pipeline looks for `config.yaml` in the project root. To use a different file (e.g. for a different environment or use case):

```bash
CONFIG_PATH=/path/to/other-config.yaml python app/main.py
```

---

## Running the Pipeline

### Test Mode (always run this first)

```bash
python tests/test_runner.py
```

- Fetches up to `settings.test_row_limit` rows (set in `config.yaml`, default: 100) from the source table
- Clears the target column in memory to force re-classification
- Runs classification and saves results to `data/test_run/output.json`
- **Never writes anything back to the database**
- Saves partial results automatically if interrupted with `Ctrl+C`

Use test mode to validate your label definitions and prompt wording before touching production data.

### Production Mode

```bash
python app/main.py
```

- Fetches only rows where the target column is `NULL` or empty
- Classifies in batches and writes results back to the database immediately after each batch
- Loops until no unclassified rows remain
- Safe to interrupt and re-run — already-classified rows are always skipped

### Controlling batch size and rate limits are both set `config.yaml`. Larger batches mean fewer API calls but larger prompts per call. A value of 10–20 rows is a good starting point for most models and table schemas.

### Controlling the rate limit

Set `max_rpm` in `config.yaml` under `settings`. This controls how many requests per minute the pipeline sends to Azure OpenAI. The default is 30 RPM. If you notice 429 errors in the logs, lower this value. If your Azure deployment has a higher quota, you can increase it for faster throughput.

---

## Adapting to a New Use Case

To classify a completely different table with different labels, edit only `config.yaml`. No Python changes required.

**Example: classifying support tickets by topic**

```yaml
source:
  schema: "dbo"
  table: "support_tickets"
  primary_key: "ticket_id"

target:
  column: "topic"

context_columns:
  - "subject"
  - "body_snippet"

display_column: "subject"

labels:
  - name: "Billing"
    description: Issues related to invoices, payments, and charges.
  - name: "Technical"
    description: Software bugs, crashes, and integration errors.
  - name: "Account"
    description: Login, password reset, and access management.
  - name: "Other"
    description: Anything that does not fit the above.

fallback_label: "Other"
```

Everything else — the database connection, batching logic, LLM client, prompt assembly, retry handling, logging, and console output — stays exactly the same.

---

## Project Structure

```
llm-classifies-data/
├── config.yaml                         # ← Primary configuration (edit this)
├── .env                                # ← Credentials (create this, never commit)
│
├── app/
│   ├── main.py                         # Production entry point
│   ├── config/
│   │   ├── loader.py                   # Loads & validates config.yaml → PipelineConfig
│   │   ├── constants.py                # Path constants and environment helpers
│   │   └── exceptions.py              # PipelineError
│   ├── db/
│   │   └── db_connector.py             # Azure SQL: connect, fetch unclassified, update
│   ├── helpers/
│   │   └── data_operations.py          # JsonManager, output validation helpers
│   ├── services/llm/
│   │   ├── azure_client.py             # Azure OpenAI REST client
│   │   ├── prompt_builder.py           # Assembles per-batch prompts from config
│   │   └── classification_orchestrator.py  # Batcher, Parser, run_classification
│   └── utils/
│       ├── console.py                  # Formatted terminal output
│       ├── logging.py                  # File + console structured logging
│       └── rate_limiter.py             # Token-bucket RPM rate limiter
│
├── tests/
│   └── test_runner.py                  # Safe test mode — read from DB, write to JSON
│
├── data/
│   └── test_run/
│       ├── input.json                  # Raw rows exported during last test run
│       └── output.json                 # Classified results from last test run
│
├── logs/                               # Auto-generated log files (gitignored)
└── requirements.txt
```

---

## Architecture

The pipeline is split into four independent layers. Each layer only depends on the layer below it; none of them know about each other directly.

```
┌──────────────────────────────────────────────────────────┐
│           main.py / test_runner.py                       │
│    Entry points — load config, wire components, run      │
└──────────────────────────┬───────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
│   config/    │  │    db/       │  │     services/llm/    │
│  loader.py   │  │ db_connector │  │  azure_client        │
│  (YAML →     │  │ (fetch/write │  │  prompt_builder      │
│  dataclass)  │  │  SQL rows)   │  │  orchestrator        │
└──────────────┘  └──────────────┘  └──────────────────────┘
```

**Key design decisions:**
- `PipelineConfig` is a plain Python dataclass — no global state, no singletons, easy to test
- All components receive their config explicitly via constructor injection
- The DB layer and LLM layer are fully decoupled — swapping either has no impact on the other
- `run_classification` operates on a pandas DataFrame in place and optionally writes checkpoints to JSON

---

## Key Features

| Feature | Detail |
|---|---|
| **Config-driven** | Change table, columns, labels, and prompts in `config.yaml` — no code edits |
| **Idempotent** | Only processes `NULL` rows; safe to stop and re-run at any point |
| **Two run modes** | `test_runner.py` for safe local testing; `main.py` for production DB writes |
| **Partial save on interrupt** | In test mode, `Ctrl+C` saves whatever was classified so far to `output.json` |
| **Config validation** | Startup checks catch misconfigured YAML before any DB or API calls |
| **Configurable fallback** | Any label can be the fallback — the LLM is explicitly instructed to use it when uncertain |
| **Batch checkpointing** | Progress is written to `output.json` after every batch in test mode |
| **Formatted console output** | Per-row classification results visible in the terminal as each batch completes |
| **Structured logging** | Full run logs written to `logs/` with timestamps and batch-level detail |
| **Rate limiting** | Built-in token-bucket rate limiter prevents Azure 429 throttling; auto-retries with `Retry-After` if it occurs |
| **Flexible config path** | Override via `CONFIG_PATH` env var to use a different YAML for different environments |

