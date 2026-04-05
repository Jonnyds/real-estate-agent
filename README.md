# Real Estate Asset Management - Multi-Agent System

Natural language interface for querying a real estate financial ledger. Built with LangGraph, Azure OpenAI, and Streamlit. Ask about P&L, compare buildings, break things down by quarter or tenant - the system figures out what you mean and gives you the numbers.

## Setup

```bash
git clone <repo-url>
cd real-estate-agent
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2025-03-01-preview
DATA_PATH=data/ledger.parquet
```

Place the dataset and run:

```bash
mkdir -p data
cp <your-parquet-file> data/ledger.parquet
streamlit run app.py
```

## Architecture

Six nodes in a LangGraph state machine. Four use LLM calls, two are pure Python:

```
User → Router → Retriever → Analyst → Validator → Responder → Memory → Answer
        (LLM)   (Python)     (LLM)     (LLM)       (LLM)      (LLM\Python)
```

Not every query goes through the full pipeline. The Router classifies the query first - general knowledge questions ("what is P&L?") and unclear queries without any data entities skip straight to the Responder. Everything else goes through the data pipeline.

```
general/unclear, no entities  →  Router → Responder → Memory → END
data query                    →  Router → Retriever → Analyst → Validator → Responder → Memory → END
```

### Router

The Router is the entry point. It reads the user's question and the conversation history, then outputs a JSON with three things: the intent (data query, comparison, general knowledge, unclear), the detail level (summary or detailed), and the extracted entities (which buildings, years, tenants, ledger categories, etc.).

The Router also handles follow-ups. If the user asked about Building 17 and then says "2024?", the Router reads the conversation history and outputs the complete entity set - Building 17 + 2024. It resolves time references too: "this year" becomes the actual calendar year (2026), "latest" becomes the most recent year in the data.

The classification step matters because it controls routing. "General" intent with no entities skips the data pipeline entirely - no point filtering a dataframe for "what does P&L stand for?" But if the Router extracts any entity from the query, it goes through the Retriever regardless of intent, because we want the data to validate the response rather than letting the LLM make things up.

### Retriever

Pure Python, no LLM. Takes the extracted entities and filters the parquet dataframe using pandas. Builds two things the downstream agents need: a scope string ("Buildings: Building 17 | Years: 2024") that tells the Responder what to mention, and a compact data summary (unique values, counts per dimension) that tells the Analyst what's in the data without sending raw rows.

The Retriever has one important guard: if the Router lists values that cover everything available within the current filters (like listing all 8 tenants of Building 17), the filter is redundant. Applying it with `isin()` would silently drop NaN rows - entity-level expenses like insurance or bank charges where `tenant_name` is empty. The Retriever detects this and removes the redundant filter so those rows survive.

When filters produce zero rows, the Retriever tries dropping each filter one at a time to figure out which one caused the problem. It reports this as a hint: "this data exists without the buildings filter, try asking without it." The user decides what to do - the system doesn't silently change their query.

### Analyst

This is where the split between LLM reasoning and Python execution happens. The Analyst LLM receives the data summary and decides what to compute - it outputs a plan like `["total", "group_by:quarter"]`. Then a Python executor maps each operation to a pandas call. The LLM plans, Python computes. No hallucinated math.

Two operation types: `"total"` computes overall P&L, revenue, and expenses. `"group_by:<column>"` groups by any column and returns P&L breakdown per group. The column name gets validated against a whitelist derived from the dataset config. The Analyst also receives the previous turn's operations, so follow-ups reuse the same analysis type without the LLM having to re-derive it.

### Validator

Checks if the Analyst's output actually answers the question. If the user asked "split by tenant" but the results don't have a tenant breakdown, the Validator sends feedback back to the Analyst: "Add group_by:tenant_name". The Analyst retries with this feedback (max once). This is the agent-to-agent communication loop - one LLM reviewing another's work.

The Validator is conservative - it passes through if in doubt. Only fails validation when a clearly needed operation is missing. If the Validator itself errors out, it passes through so the pipeline doesn't break.

### Responder

Takes the computed results, the scope, and the data summary, and formats a readable answer. It only mentions dimensions that appear in the scope - so "sum all expenses" without a building filter produces "Total expenses are EUR 1,354,321.02" without listing every building. For factual questions like "how many tenants does Building 120 have?", it reads the count from the data summary rather than the financial results. The ledger descriptions are in Dutch which suggests EUR as currency.

### Memory

Stores conversation history and compresses old turns. Each user entry includes the extracted entities so the Router can see what was queried before. Every 6 turns, older history gets compressed into a summary paragraph by an LLM call. The last 2 turns stay verbatim. This keeps the context window small while preserving what matters.

## Project Structure

```
├── app.py                  Streamlit chat UI
├── graph.py                LangGraph state machine + routing logic
├── data_loader.py          Dataset config + parquet loader + generic filter
├── agents/
│   ├── __init__.py         Exports all node functions
│   ├── helpers.py          LLM client with retry, history formatting, JSON parsing
│   ├── router.py           Intent classification + entity extraction
│   ├── retriever.py        Dataframe filtering + scope building
│   ├── analyst.py          Analysis planning + pandas execution
│   ├── validator.py        Output quality check + retry logic
│   ├── responder.py        Natural language formatting
│   └── memory.py           Conversation history + summarization
├── requirements.txt
├── .env.example
└── README.md
```

## Dataset Config

All column references flow through one config in `data_loader.py`:

```python
COLUMN_MAP = {
    "buildings": "property_name",
    "tenants": "tenant_name",
    "years": "year",
    ...
}
VALUE_COLUMN = "profit"
```

The retriever, analyst, scope builder, and data summary all loop over this map. Swapping datasets means editing these two values. The agent code doesn't change.

## Notes

**Local file, no API.** The dataset is a parquet file loaded directly into pandas. No external data API, no database. The Retriever filters the dataframe in memory. This is the simplest approach for a single-file dataset and avoids adding infrastructure that doesn't serve the demo.

**Python logic over tools/MCP.** I considered using LLM tool calling or MCP to let the model query the data directly. For a single parquet file with ~4K rows, that's unnecessary infrastructure - tools and MCP shine when you have multiple data sources the LLM needs to discover dynamically. More importantly, financial calculations can't afford hallucination. `df.groupby().sum()` gives the same answer every time. An LLM with a SQL tool might write slightly different queries on each run or misinterpret column semantics. The split is deliberate: LLMs handle natural language (classification, planning, formatting), Python handles data (filtering, aggregation, computation).

**GPT-4.1 mini for all agents.** Each query makes 4 LLM calls (Router, Analyst, Validator, Responder), sometimes 5 on a retry. All are text-only classification or formatting tasks - mini handles them well at low cost and latency. The Analyst never sees raw data rows, only the compact summary. The architecture supports using different models per agent - for example, a cheaper model for the Validator which just does a pass/fail check - but for this scale there's no need.

## Challenges

**Follow-up queries.** The hardest part. "And 2024?" after discussing Building 17 needs to resolve to Building 17 in 2024. I tried Python-based context merging with structured entity memory - it kept creating edge cases. Sticky filters, time hierarchy conflicts, ambiguous phrases like "in general." The final approach is simpler: pass the full conversation history to the Router with entity annotations on each turn, and let the LLM read the context naturally. No Python override, no separate memory structures.

**Entity-level records.** The dataset has rows where `property_name` is NaN - entity-level expenses like bank charges and director's fees. These carry real financial data. Filtering by building with `isin()` drops them because NaN never matches. The contextual check in the Retriever catches this: if the listed values are ALL values available in the current context, the filter is redundant and gets removed. NaN rows survive.

**LLM hallucination.** The Responder would sometimes invent numbers or claim data doesn't exist when it does. Two guards: the prompt says to only use numbers from the Results field, and the routing logic sends any query with extracted entities through the Retriever to validate against actual data - even if the intent is unclear. The Responder never answers a data question without the Retriever having run first.

**Wrong operations.** "Split by tenants" sometimes produced only totals. The Validator catches this by checking if the result keys match what was asked, and sends specific feedback back to the Analyst. One retry is enough.

## What I'd Improve

**Evaluation framework.** I tested manually with a validation set but there's no automated suite. A proper eval would run 20-30 queries, compare outputs against expected values, and catch regressions. I didn't build it because the system was changing too fast during development - manual testing was faster for iteration. First thing to add for production.

**Smarter matching.** Filtering is exact string matching. Works here because 5 buildings with simple names - the Router LLM normalizes "building 160" to "Building 160." With hundreds of properties or free-text addresses, you'd need approximate matching or search indexing in the Retriever instead of relying on the LLM to get the exact name right.


**Streaming.** The user waits for all 4 agents to finish. Streaming each agent's output as it completes would feel much faster.

**Data preparation.** The raw dataset goes straight in - NaN values, no validation. A production system would have an ETL step: normalize text, validate types, handle missing values upfront instead of at query time.

**More operations.** Currently `total` and `group_by`. With more time: percentages, year-over-year changes, rolling averages, pivot tables. The executor pattern scales - add a new operation type without touching the LLM prompts.

**Caching.** Same query runs the full pipeline every time. Hashing entities + intent and returning cached results would cut cost and latency. Skipped because it adds state complexity for a demo.
