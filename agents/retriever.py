"""Retriever - filters the dataframe by extracted entities (pure Python, no LLM)."""

import logging
from data_loader import filter_df, get_metadata, COLUMN_MAP

logger = logging.getLogger(__name__)


def retriever_node(state: dict) -> dict:
    """Filter dataframe, build scope string and data summary for Analyst."""
    ent = state.get("entities", {})
    steps = list(state.get("steps", []))

    try:
        meta = get_metadata()
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        steps.append({"agent": "Retriever", "reasoning": f"Failed to load dataset: {e}"})
        return {"data": None, "error": f"Dataset error: {e}", "steps": steps}

    # Build filters - validate scalars against metadata
    filters = {}
    for key in COLUMN_MAP:
        vals = ent.get(key) or None
        if not vals:
            continue
        if isinstance(vals, list):
            filters[key] = vals
        elif key in meta and vals in meta[key]:
            filters[key] = vals

    # If listed values cover everything available within the other filters,
    # the filter is redundant. Removing it preserves NaN rows (entity-level expenses).
    for key in list(filters.keys()):
        if not isinstance(filters[key], list):
            continue
        try:
            other_filters = {k: v for k, v in filters.items() if k != key}
            other_df = filter_df(**other_filters)
            available = set(other_df[COLUMN_MAP[key]].dropna().unique())
            if available and set(filters[key]) >= available:
                del filters[key]
        except Exception as e:
            logger.warning(f"Contextual check failed for {key}: {e}")

    try:
        df = filter_df(**filters)
    except Exception as e:
        logger.error(f"Filter failed: {e}")
        steps.append({"agent": "Retriever", "reasoning": f"Filter error: {e}"})
        return {"data": None, "error": f"Filter error: {e}", "steps": steps}

    # Empty - return error with available alternatives and hint
    if df.empty:
        error_parts = ["No data found for the specified filters."]
        for key in filters:
            if key in meta:
                error_parts.append(f"Available {key}: {meta[key]}.")
        for key in list(filters.keys()):
            try:
                relaxed_df = filter_df(**{k: v for k, v in filters.items() if k != key})
                if not relaxed_df.empty:
                    error_parts.append(
                        f"Note: this data exists without the {key} filter "
                        f"({filters[key]}), try asking without it.")
                    break
            except Exception:
                pass
        steps.append({"agent": "Retriever", "reasoning": f"No matching rows. {' '.join(error_parts)}"})
        return {"data": None, "error": " ".join(error_parts), "steps": steps}

    # Build scope - only active filters
    scope_parts = []
    for key, vals in filters.items():
        label = vals if isinstance(vals, str) else ", ".join(str(v) for v in sorted(vals))
        scope_parts.append(f"{key.title()}: {label}")

    reasoning = f"Filtered {len(df)} rows. {'; '.join(scope_parts)}."
    steps.append({"agent": "Retriever", "reasoning": reasoning})

    # Data summary - compact stats for Analyst and Responder
    data_summary = {"row_count": len(df)}
    for key, col in COLUMN_MAP.items():
        vals = sorted(df[col].dropna().unique().tolist())
        data_summary[key] = vals
        data_summary[f"n_{key}"] = len(vals)

    return {
        "data": df.to_dict(orient="records"),
        "data_summary": data_summary,
        "scope": " | ".join(scope_parts),
        "steps": steps,
    }
