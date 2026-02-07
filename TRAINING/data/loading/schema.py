# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Schema validation for loaded data.

Provides utilities for validating DataFrame schemas against requirements.
Supports auto-detection of time columns and optional auto-fixing of
common schema issues.

Example:
    ```python
    from TRAINING.data.loading.schema import validate_dataframe, DEFAULT_SCHEMA

    df = pd.read_csv("data.csv")
    validated_df = validate_dataframe(df, DEFAULT_SCHEMA, strict=False, auto_fix=True)
    ```
"""

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from .interface import SchemaRequirement, SchemaValidationError

logger = logging.getLogger(__name__)


# Default schema for FoxML pipeline
DEFAULT_SCHEMA = SchemaRequirement(
    required_columns={"timestamp"},
    time_column="timestamp",
    index_column=None,
    dtypes=None,
)


# Common time column names for auto-detection (in preference order)
TIME_COLUMN_CANDIDATES = [
    "timestamp",
    "ts",
    "time",
    "datetime",
    "date",
    "ts_pred",
    "Timestamp",
    "DateTime",
    "Date",
    "Time",
]


def validate_dataframe(
    df: pd.DataFrame,
    schema: SchemaRequirement,
    strict: bool = False,
    auto_fix: bool = True,
) -> pd.DataFrame:
    """Validate and optionally fix dataframe schema.

    Args:
        df: DataFrame to validate
        schema: Schema requirements
        strict: If True, raise on validation failure
        auto_fix: If True, attempt to fix common issues

    Returns:
        Validated (and possibly fixed) DataFrame

    Raises:
        SchemaValidationError: If strict mode and validation fails
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Make a copy if we might modify it
    if auto_fix:
        df = df.copy()

    # Check for time column
    time_col = schema.time_column
    if time_col not in df.columns:
        # Try auto-detection
        detected = _detect_time_column(df.columns.tolist())

        if detected:
            if auto_fix:
                df = df.rename(columns={detected: time_col})
                warnings.append(f"Renamed '{detected}' to '{time_col}'")
            else:
                errors.append(
                    f"Time column '{time_col}' not found (detected: {detected})"
                )
        else:
            errors.append(f"Time column '{time_col}' not found")

    # Check required columns
    missing = schema.required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    # Check dtypes if specified
    if schema.dtypes:
        for col, expected_dtype in sorted(schema.dtypes.items()):
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not _dtype_compatible(actual_dtype, expected_dtype):
                    if auto_fix:
                        try:
                            df[col] = df[col].astype(expected_dtype)
                            warnings.append(
                                f"Converted '{col}' from {actual_dtype} to {expected_dtype}"
                            )
                        except (ValueError, TypeError) as e:
                            errors.append(
                                f"Cannot convert '{col}' to {expected_dtype}: {e}"
                            )
                    else:
                        errors.append(
                            f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
                        )

    # Log warnings
    for warning in warnings:
        logger.warning(f"Schema auto-fix: {warning}")

    # Handle errors
    if errors:
        error_msg = f"Schema validation failed: {'; '.join(errors)}"
        if strict:
            raise SchemaValidationError(error_msg)
        else:
            logger.warning(error_msg)

    return df


def _detect_time_column(columns: List[str]) -> Optional[str]:
    """Detect time column from column names.

    Args:
        columns: List of column names

    Returns:
        Detected time column name or None
    """
    columns_set = set(columns)
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in columns_set:
            return candidate
    return None


def _dtype_compatible(actual: str, expected: str) -> bool:
    """Check if dtypes are compatible.

    Args:
        actual: Actual dtype string
        expected: Expected dtype string

    Returns:
        True if compatible
    """
    # Normalize dtype strings
    actual = actual.lower()
    expected = expected.lower()

    # Direct match
    if actual == expected:
        return True

    # Numeric compatibility
    numeric_types = {"float64", "float32", "int64", "int32", "int16", "float", "int"}
    if actual in numeric_types and expected in numeric_types:
        return True

    # String compatibility
    string_types = {"object", "string", "str"}
    if actual in string_types and expected in string_types:
        return True

    # Datetime compatibility
    datetime_types = {"datetime64[ns]", "datetime64", "datetime"}
    if actual in datetime_types or expected in datetime_types:
        # Allow if either is datetime-like
        if any(actual.startswith(dt) for dt in ["datetime64"]):
            return True
        if any(expected.startswith(dt) for dt in ["datetime64"]):
            return True

    return False


def infer_schema(df: pd.DataFrame) -> SchemaRequirement:
    """Infer schema from a dataframe.

    Useful for documenting existing data format or generating
    schema requirements from sample data.

    Args:
        df: DataFrame to infer schema from

    Returns:
        SchemaRequirement with inferred values
    """
    # Find time column
    time_col = _detect_time_column(df.columns.tolist())

    return SchemaRequirement(
        required_columns=set(df.columns),
        time_column=time_col or "timestamp",
        dtypes={col: str(df[col].dtype) for col in sorted(df.columns)},
    )


def check_schema(df: pd.DataFrame, schema: SchemaRequirement) -> List[str]:
    """Check schema without modifying dataframe.

    Args:
        df: DataFrame to check
        schema: Schema requirements

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []

    # Check time column
    if schema.time_column not in df.columns:
        errors.append(f"Missing time column: {schema.time_column}")

    # Check required columns
    missing = schema.required_columns - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    # Check dtypes
    if schema.dtypes:
        for col, expected in sorted(schema.dtypes.items()):
            if col in df.columns:
                actual = str(df[col].dtype)
                if not _dtype_compatible(actual, expected):
                    errors.append(f"Column '{col}': expected {expected}, got {actual}")

    return errors
