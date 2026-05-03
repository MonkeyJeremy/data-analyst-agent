import pandas as pd


def load_tabular(file: object, filename: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Args:
        file: File-like object (e.g. from st.file_uploader or open()).
        filename: Original filename used to detect format.

    Returns:
        Parsed DataFrame.

    Raises:
        ValueError: If the file format is unsupported or the file cannot be parsed.
    """
    name = filename.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
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
