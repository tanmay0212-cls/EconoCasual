"""Dataset validation service enforcing schema, domain, and causal integrity rules."""

import io
import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from fastapi import HTTPException, status, UploadFile

from src.api.config import settings
from src.api.schemas.data import (
    DatasetValidationResponse,
    DatasetValidationSummary,
    UploadSummaryResponse,
)

logger = logging.getLogger("econocausal.data_service")

# Required columns for multi-tier causal datasets (Week 3/4)
MULTI_TIER_REQUIRED_COLS: List[str] = [
    "customer_id",
    "age",
    "loyalty_score",
    "previous_purchases",
    "customer_segment",
    "discount_level",
    "purchase",
    "product_price",
    "gross_margin_rate",
]

# Required columns for binary treatment campaign datasets (Week 1/2)
BINARY_REQUIRED_COLS: List[str] = [
    "customer_id",
    "prior_purchases",
    "prior_spend",
    "engagement_score",
    "customer_tenure",
    "discount_offered",
    "purchased",
]


class DataService:
    """Service to safely load, validate, and check schemas of retail customer datasets."""

    @staticmethod
    def resolve_dataset_reference(reference_name: str) -> pd.DataFrame:
        """Resolve an approved server-side dataset identifier to a DataFrame.
        
        Strictly rejects arbitrary paths, traversal strings (e.g. '../'), or unknown names.
        """
        clean_ref = str(reference_name).strip()
        if clean_ref not in settings.allowed_dataset_references:
            logger.warning("Rejected unauthorized dataset reference attempt: %s", clean_ref)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid dataset reference '{clean_ref}'. "
                    f"Allowed references are: {list(settings.allowed_dataset_references.keys())}"
                ),
            )

        file_path = settings.allowed_dataset_references[clean_ref]
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Approved dataset reference '{clean_ref}' was not found on disk at configured path.",
            )

        try:
            return pd.read_csv(file_path)
        except Exception as exc:
            logger.error("Error reading approved dataset %s: %s", clean_ref, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read dataset reference '{clean_ref}': {str(exc)}",
            )

    @staticmethod
    async def read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
        """Read and decode an uploaded CSV file with size and format guards."""
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for '{file.filename}'. Only .csv files are supported.",
            )

        contents = await file.read()
        if len(contents) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File size ({round(len(contents) / (1024 * 1024), 2)} MB) exceeds "
                    f"maximum allowed upload size ({settings.max_upload_size_bytes // (1024 * 1024)} MB)."
                ),
            )

        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty.",
            )

        try:
            df = pd.read_csv(io.BytesIO(contents))
            return df
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed CSV content: {str(exc)}",
            )

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, dataset_name: str) -> DatasetValidationResponse:
        """Perform comprehensive causal schema and domain validation on a DataFrame."""
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or len(df) == 0:
            return DatasetValidationResponse(
                is_valid=False,
                dataset_name=dataset_name,
                total_rows=0,
                valid_rows=0,
                errors=["Dataset contains 0 rows."],
                warnings=[],
                summary=DatasetValidationSummary(
                    total_rows=0,
                    valid_rows=0,
                    columns=[],
                    column_count=0,
                    missing_value_counts={},
                ),
            )

        total_rows = len(df)
        cols = list(df.columns)

        # Detect schema type: multi-tier vs binary
        is_multi_tier = "discount_level" in cols or "age" in cols
        required_cols = MULTI_TIER_REQUIRED_COLS if is_multi_tier else BINARY_REQUIRED_COLS

        # 1. Missing columns check
        missing = [c for c in required_cols if c not in cols]
        if missing:
            errors.append(f"Missing required schema columns: {missing}")

        # 2. Missing value counts
        missing_counts = {c: int(df[c].isnull().sum()) for c in cols if df[c].isnull().sum() > 0}
        critical_missing = {c: missing_counts[c] for c in required_cols if c in missing_counts}
        if critical_missing:
            errors.append(f"Missing (null) values detected in required columns: {critical_missing}")

        # 3. Treatment and outcome domain checks
        treatment_dist: Dict[str, int] = {}
        conversion_rate: float = 0.0
        avg_order_value = None

        if is_multi_tier and "discount_level" in cols:
            levels = set(df["discount_level"].dropna().unique())
            treatment_dist = {str(k): int((df["discount_level"] == k).sum()) for k in sorted(levels)}
            if not levels.issubset({0, 10, 20}):
                errors.append(f"Invalid discount_level values: {levels}. Must be a subset of {{0, 10, 20}}.")

        if "purchase" in cols:
            outcomes = set(df["purchase"].dropna().unique())
            if not outcomes.issubset({0, 1}):
                errors.append(f"Invalid purchase outcome values: {outcomes}. Must be binary {{0, 1}}.")
            else:
                conversion_rate = round(float(df["purchase"].mean()), 4)

        if "discount_offered" in cols:
            binary_treat = set(df["discount_offered"].dropna().unique())
            treatment_dist = {str(k): int((df["discount_offered"] == k).sum()) for k in sorted(binary_treat)}
            if not binary_treat.issubset({0, 1}):
                errors.append(f"Invalid discount_offered values: {binary_treat}. Must be binary {{0, 1}}.")

        if "purchased" in cols:
            outcomes = set(df["purchased"].dropna().unique())
            if not outcomes.issubset({0, 1}):
                errors.append(f"Invalid purchased outcome values: {outcomes}. Must be binary {{0, 1}}.")
            else:
                conversion_rate = round(float(df["purchased"].mean()), 4)

        # 4. Numerical range bounds checks
        if "age" in cols:
            if (df["age"] < 18).any() or (df["age"] > 105).any():
                errors.append("Column 'age' contains values outside plausible range [18, 105].")
        if "loyalty_score" in cols:
            if (df["loyalty_score"] < 0).any() or (df["loyalty_score"] > 100).any():
                errors.append("Column 'loyalty_score' contains values outside [0, 100].")
        if "product_price" in cols:
            if (df["product_price"] < 0).any():
                errors.append("Column 'product_price' contains negative values.")
            avg_order_value = round(float(df["product_price"].mean()), 2)
        if "gross_margin_rate" in cols:
            if (df["gross_margin_rate"] < 0).any() or (df["gross_margin_rate"] > 1.0).any():
                errors.append("Column 'gross_margin_rate' contains values outside [0.0, 1.0].")

        # 5. Customer ID checks
        if "customer_id" in cols:
            dups = int(df["customer_id"].duplicated().sum())
            if dups > 0:
                warnings.append(f"Found {dups} duplicate customer_id records.")

        # Determine valid row count
        valid_rows = total_rows if len(errors) == 0 else 0

        summary = DatasetValidationSummary(
            total_rows=total_rows,
            valid_rows=valid_rows,
            columns=cols,
            column_count=len(cols),
            missing_value_counts=missing_counts,
            treatment_distribution=treatment_dist,
            conversion_rate=conversion_rate,
            average_order_value=avg_order_value,
        )

        return DatasetValidationResponse(
            is_valid=len(errors) == 0,
            dataset_name=dataset_name,
            total_rows=total_rows,
            valid_rows=valid_rows,
            errors=errors,
            warnings=warnings,
            summary=summary,
        )

    @classmethod
    def process_upload(cls, df: pd.DataFrame, filename: str) -> UploadSummaryResponse:
        """Process and validate an uploaded customer CSV file with quarantine and schema checks."""
        errors: List[str] = []
        warnings: List[str] = []
        quarantined: List[str] = []

        if df is None or len(df) == 0:
            return UploadSummaryResponse(
                status="error",
                filename=filename,
                total_rows=0,
                valid_rows=0,
                invalid_rows=0,
                columns_detected=[],
                missing_values_by_column={},
                duplicate_customer_ids=0,
                data_types={},
                validation_passed=False,
                quarantined_columns=[],
                errors=["Uploaded file contains no rows or data."],
                warnings=[],
                preview=[],
            )

        total_rows = len(df)
        cols = list(df.columns)
        data_types = {col: str(df[col].dtype) for col in cols}

        # 1. Anti-leakage / quarantine of ground-truth or potential-outcome columns
        leakage_prefixes = ("potential_", "counterfactual_", "ground_truth_", "tau_", "actual_outcome", "actual_purchase")
        for col in cols:
            col_lower = col.lower()
            if any(col_lower.startswith(p) for p in leakage_prefixes):
                quarantined.append(col)

        if quarantined:
            warnings.append(
                f"Quarantined {len(quarantined)} ground-truth/counterfactual columns from operational decision inputs: {quarantined}"
            )

        # 2. Check for customer identifier
        duplicate_ids = 0
        if "customer_id" in cols:
            duplicate_ids = int(df["customer_id"].duplicated().sum())
            if duplicate_ids > 0:
                warnings.append(f"Detected {duplicate_ids} duplicate customer_id records. Recommend deduplication before optimization.")
        else:
            warnings.append("No 'customer_id' column found; autogenerated indices will be assigned.")

        # 3. Check required causal feature columns
        required_causal = ["age", "loyalty_score", "previous_purchases", "customer_segment"]
        missing_causal = [c for c in required_causal if c not in cols]
        if missing_causal:
            errors.append(f"Missing core customer feature columns required for DML: {missing_causal}")

        # 4. Missing value checks
        missing_counts = {c: int(df[c].isnull().sum()) for c in cols if df[c].isnull().sum() > 0}
        critical_nulls = {c: missing_counts[c] for c in required_causal if c in missing_counts}
        if critical_nulls:
            errors.append(f"Null values detected in core features: {critical_nulls}")

        # 5. Row-level invalid record detection
        invalid_mask = pd.Series(False, index=df.index)
        if "age" in cols:
            age_numeric = pd.to_numeric(df["age"], errors="coerce")
            invalid_age = (age_numeric < 18) | (age_numeric > 105) | age_numeric.isna()
            if invalid_age.any():
                invalid_mask |= invalid_age
                warnings.append(f"Found {int(invalid_age.sum())} rows with out-of-range age (<18 or >105).")

        if "loyalty_score" in cols:
            loyalty_numeric = pd.to_numeric(df["loyalty_score"], errors="coerce")
            invalid_loyalty = (loyalty_numeric < 0) | (loyalty_numeric > 100) | loyalty_numeric.isna()
            if invalid_loyalty.any():
                invalid_mask |= invalid_loyalty
                warnings.append(f"Found {int(invalid_loyalty.sum())} rows with out-of-range loyalty_score (<0 or >100).")

        if "product_price" in cols:
            price_numeric = pd.to_numeric(df["product_price"], errors="coerce")
            invalid_price = (price_numeric < 0) | price_numeric.isna()
            if invalid_price.any():
                invalid_mask |= invalid_price
                errors.append(f"Found {int(invalid_price.sum())} rows with negative or invalid product_price.")

        invalid_rows_count = int(invalid_mask.sum())
        valid_rows_count = max(0, total_rows - invalid_rows_count)

        validation_passed = len(errors) == 0
        status_str = "success" if (validation_passed and len(warnings) == 0) else "warning" if validation_passed else "error"

        # 6. Generate 5-row preview (sanitizing NaN to None)
        preview_data = df.head(5).replace({np.nan: None}).to_dict(orient="records")

        return UploadSummaryResponse(
            status=status_str,
            filename=filename,
            total_rows=total_rows,
            valid_rows=valid_rows_count,
            invalid_rows=invalid_rows_count,
            columns_detected=cols,
            missing_values_by_column=missing_counts,
            duplicate_customer_ids=duplicate_ids,
            data_types=data_types,
            validation_passed=validation_passed,
            quarantined_columns=quarantined,
            errors=errors,
            warnings=warnings,
            preview=preview_data,
        )


