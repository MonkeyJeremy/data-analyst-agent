from __future__ import annotations

import pathlib

import pandas as pd


def load_tabular(
    file: object,
    filename: str,
    header: int | None = 0,
    skiprows: int | None = None,
) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    file:
        File-like object or path (e.g. from st.file_uploader or open()).
    filename:
        Original filename used to detect format.
    header:
        Row number(s) to use as column names (passed to pandas).
        Default 0 = first row.  Pass the detected row index when the file
        has a non-standard layout.
    skiprows:
        Number of rows to skip before the header (passed to pandas).
        Rarely needed — prefer adjusting *header* instead.

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
                f"Unsupported file format: '{filename}'. Please upload a .csv or .xlsx file."
            )
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not parse '{filename}': {exc}. "
            "Ensure the file is a valid CSV or Excel file and is not corrupted."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Failed to load '{filename}': {exc}") from exc
