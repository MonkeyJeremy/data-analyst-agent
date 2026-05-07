from __future__ import annotations

import io
import json
import pathlib

import pandas as pd


def load_tabular(
    file: object,
    filename: str,
    header: int | None = 0,
    skiprows: int | None = None,
) -> pd.DataFrame:
    """Load a CSV, Excel, or JSON file into a DataFrame.

    Parameters
    ----------
    file:
        File-like object or path (e.g. from st.file_uploader or open()).
    filename:
        Original filename used to detect format.
    header:
        Row number(s) to use as column names (passed to pandas for CSV/Excel).
        Default 0 = first row.  Pass the detected row index when the file
        has a non-standard layout.  Not used for JSON files.
    skiprows:
        Number of rows to skip before the header (passed to pandas for CSV/Excel).
        Rarely needed — prefer adjusting *header* instead.  Not used for JSON.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame.

    Raises
    ------
    ValueError
        If the file format is unsupported or the file cannot be parsed.
    """
    name = filename.lower()

    # Accept pathlib.Path objects transparently
    if isinstance(file, pathlib.Path):
        file = str(file)

    # ── JSON ──────────────────────────────────────────────────────────────────
    if name.endswith(".json"):
        return _load_json(file, filename)

    # ── CSV / Excel ──────────────────────────────────────────────────────────
    kwargs: dict = {"header": header}
    if skiprows is not None:
        kwargs["skiprows"] = skiprows

    try:
        if name.endswith(".csv"):
            return pd.read_csv(file, **kwargs)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: '{filename}'. "
                "Please upload a .csv, .xlsx, .xls, or .json file."
            )
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not parse '{filename}': {exc}. "
            "Ensure the file is a valid CSV or Excel file and is not corrupted."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Failed to load '{filename}': {exc}") from exc


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _load_json(file: object, filename: str) -> pd.DataFrame:
    """Load a JSON file into a DataFrame via json.loads + json_normalize.

    Supported top-level structures:
    - Array of objects: ``[{"a": 1}, ...]``
    - Dict with a known array wrapper key (``"data"``, ``"records"``,
      ``"results"``, or ``"rows"``): ``{"data": [...]}``
    - Single object: ``{"a": 1, "b": 2}``  → one-row DataFrame

    Raises
    ------
    ValueError
        If the JSON is malformed or the top-level structure is not supported.
    """
    # Read raw bytes
    if hasattr(file, "seek"):
        file.seek(0)  # type: ignore[union-attr]
    if hasattr(file, "read"):
        raw = file.read()  # type: ignore[union-attr]
    else:
        with open(str(file), "rb") as f:
            raw = f.read()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{filename}': {exc}") from exc

    # ── Dispatch by top-level structure ──────────────────────────────────────
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        # Normalise: list of dicts (standard) or list of scalars
        if all(isinstance(item, dict) for item in data):
            return pd.json_normalize(data)
        # List of scalars → single-column DataFrame
        return pd.DataFrame({"value": data})

    if isinstance(data, dict):
        # Try common wrapper keys first
        for key in ("data", "records", "results", "rows"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                if not items:
                    return pd.DataFrame()
                return pd.json_normalize(items)
        # Single object → one-row DataFrame
        return pd.json_normalize([data])

    raise ValueError(
        f"Cannot parse JSON structure in '{filename}'. "
        "Expected a JSON array or object at the top level."
    )
