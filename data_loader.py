"""
Loads the financial ledger from parquet and provides filtered views.
All column references go through COLUMN_MAP - swap dataset by editing the map.
"""

import os
import logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "buildings": "property_name",
    "tenants": "tenant_name",
    "years": "year",
    "quarters": "quarter",
    "months": "month",
    "ledger_type": "ledger_type",
    "ledger_groups": "ledger_group",
    "ledger_categories": "ledger_category",
    "ledger_codes": "ledger_code",
    "ledger_descriptions": "ledger_description",
}

VALUE_COLUMN = "profit"

_df: pd.DataFrame | None = None


def get_df() -> pd.DataFrame:
    """Return the full ledger dataframe, loading from disk on first call."""
    global _df
    if _df is None:
        path = os.getenv("DATA_PATH", "data/ledger.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}. Check DATA_PATH in .env")
        _df = pd.read_parquet(path)
        logger.info(f"Loaded {len(_df)} rows from {path}")
    return _df


def get_metadata() -> dict:
    """Return unique values for each mapped column."""
    df = get_df()
    return {
        key: sorted(df[col].dropna().unique().tolist())
        for key, col in COLUMN_MAP.items()
        if col in df.columns
    }


def filter_df(**filters) -> pd.DataFrame:
    """Apply filters using COLUMN_MAP. Lists use isin(), scalars use ==."""
    df = get_df()
    for key, vals in filters.items():
        if not vals or key not in COLUMN_MAP:
            continue
        col = COLUMN_MAP[key]
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in dataframe, skipping filter '{key}'")
            continue
        if isinstance(vals, list):
            df = df[df[col].isin(vals)]
        else:
            df = df[df[col] == vals]
    return df
