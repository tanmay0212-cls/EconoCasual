"""Service for budget-constrained discount optimization and allocation retrieval."""

import os
import json
import uuid
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from fastapi import HTTPException, status

from src.api.config import settings
from src.causal.multi_tier_dml import MultiTierCausalDML
from src.optimization.budget_optimizer import SciPyBudgetOptimizer
from src.api.schemas.optimize import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationKPIs,
    CustomerAllocationRecord,
)

logger = logging.getLogger("econocausal.optimize_service")


class OptimizeService:
    """Service to execute SciPy MILP discount optimization and manage persisted runs."""

    # In-memory LRU / runs cache
    _runs_cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def _ensure_runs_dir(cls) -> str:
        os.makedirs(settings.runs_dir, exist_ok=True)
        return settings.runs_dir

    @classmethod
    def run_optimization(
        cls,
        request: OptimizationRequest,
        model: MultiTierCausalDML,
        data_service_ref_resolver,
    ) -> OptimizationResponse:
        """Run SciPy MILP discount knapsack optimization."""
        # 1. Resolve input DataFrame
        if request.records is not None and len(request.records) > 0:
            if len(request.records) > settings.max_optimize_customers:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"Requested customer count ({len(request.records)}) exceeds "
                        f"maximum optimization limit ({settings.max_optimize_customers})."
                    ),
                )
            data_dicts = [r.model_dump() for r in request.records]
            df = pd.DataFrame(data_dicts)
        elif request.dataset_reference:
            df = data_service_ref_resolver(request.dataset_reference)
            if len(df) > settings.max_optimize_customers:
                # Use held-out or top slice up to max limit
                df = df.head(settings.max_optimize_customers)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'records' or an approved 'dataset_reference' must be provided.",
            )

        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer population to optimize is empty.",
            )

        # Ensure required pricing columns exist
        if "product_price" not in df.columns:
            df["product_price"] = 85.0
        if "gross_margin_rate" not in df.columns:
            df["gross_margin_rate"] = 0.40
        if "campaign_channel" not in df.columns:
            df["campaign_channel"] = "email"

        start_time = time.perf_counter()
        budget_type_applied = getattr(request, "budget_type", "redemption")

        # 2. Execute SciPy MILP optimization
        try:
            optimizer = SciPyBudgetOptimizer(
                budget=request.budget,
                discount_tiers=request.discount_tiers,
                objective_type=request.objective_type,
                budget_type=budget_type_applied,
            )
            opt_res = optimizer.optimize(df_customers=df, model=model)
            opt_df = opt_res.allocation_matrix
        except Exception as exc:
            logger.error("SciPy MILP optimization failure: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization solver failure: {str(exc)}",
            )

        solve_time = round(time.perf_counter() - start_time, 4)
        run_id = str(uuid.uuid4())

        # 3. Format allocation records
        allocation_records: List[CustomerAllocationRecord] = []
        for _, row in opt_df.iterrows():
            disc = int(row["recommended_discount"])
            prob = float(
                row["predicted_purchase_probability_20"] if disc == 20
                else row["predicted_purchase_probability_10"] if disc == 10
                else row["predicted_purchase_probability_0"]
            )
            allocation_records.append(
                CustomerAllocationRecord(
                    customer_id=str(row["customer_id"]),
                    recommended_discount=disc,
                    predicted_purchase_probability=round(prob, 6),
                    expected_revenue=round(float(row["expected_revenue"]), 2),
                    expected_discount_cost=round(float(row["expected_discount_cost"]), 2),
                    expected_net_profit=round(float(row["expected_net_profit"]), 2),
                )
            )

        # 4. Extract KPI summary
        counts = opt_res.tier_counts
        avg_conversion = round(float(np.mean([r.predicted_purchase_probability for r in allocation_records])), 4) if allocation_records else 0.0
        kpis = OptimizationKPIs(
            total_customers=len(opt_df),
            count_discount_0=int(counts.get(0, 0)),
            count_discount_10=int(counts.get(10, 0)),
            count_discount_20=int(counts.get(20, 0)),
            expected_revenue=round(float(opt_res.expected_gross_revenue), 2),
            expected_discount_expenditure=round(float(opt_res.total_cost), 2),
            expected_net_profit=round(float(opt_res.expected_net_profit), 2),
            budget_cap=round(float(opt_res.budget), 2),
            remaining_budget=round(float(max(0.0, opt_res.budget - opt_res.total_cost)), 2),
            budget_utilization_pct=round(float(opt_res.budget_utilization_pct), 2),
            conversion_rate=avg_conversion,
        )

        response = OptimizationResponse(
            run_id=run_id,
            solver_status=str(opt_res.status_message),
            objective_type=request.objective_type,
            budget_type=budget_type_applied,
            solve_time_seconds=solve_time,
            kpis=kpis,
            allocation=allocation_records,
        )

        # 5. Save run to cache and disk
        cls._save_run(run_id, response)
        return response

    @classmethod
    def _save_run(cls, run_id: str, response: OptimizationResponse) -> None:
        """Persist optimization results in memory and on disk."""
        data_dict = response.model_dump()
        cls._runs_cache[run_id] = data_dict

        try:
            runs_dir = cls._ensure_runs_dir()
            run_file = os.path.join(runs_dir, f"{run_id}.json")
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2)
            logger.info("Saved optimization run %s to %s", run_id, run_file)
        except Exception as exc:
            logger.warning("Could not persist run to disk: %s", exc)

    @classmethod
    def get_allocation_run(cls, run_id: str) -> OptimizationResponse:
        """Retrieve a saved optimization run by run_id."""
        clean_id = str(run_id).strip()

        # Check in-memory cache
        if clean_id in cls._runs_cache:
            return OptimizationResponse.model_validate(cls._runs_cache[clean_id])

        # Check disk storage
        run_file = os.path.join(settings.runs_dir, f"{clean_id}.json")
        if os.path.exists(run_file):
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cls._runs_cache[clean_id] = data
                return OptimizationResponse.model_validate(data)
            except Exception as exc:
                logger.error("Failed to read run file %s: %s", run_file, exc)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization run '{clean_id}' was not found. Please verify the run_id.",
        )

    @classmethod
    def get_latest_or_default_run(cls, run_id: Optional[str] = None) -> OptimizationResponse:
        """Retrieve a specific run by run_id, or the most recent run available in cache or disk."""
        if run_id:
            return cls.get_allocation_run(run_id)

        # 1. Check in-memory cache for latest inserted run
        if cls._runs_cache:
            latest_id = list(cls._runs_cache.keys())[-1]
            return OptimizationResponse.model_validate(cls._runs_cache[latest_id])

        # 2. Check disk runs directory for latest modified run file
        runs_dir = cls._ensure_runs_dir()
        if os.path.exists(runs_dir):
            json_files = [
                os.path.join(runs_dir, f)
                for f in os.listdir(runs_dir)
                if f.endswith(".json") and not f.startswith("drift")
            ]
            if json_files:
                json_files.sort(key=os.path.getmtime, reverse=True)
                for fpath in json_files:
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if "allocation" in data and "kpis" in data:
                            run_obj = OptimizationResponse.model_validate(data)
                            cls._runs_cache[run_obj.run_id] = data
                            return run_obj
                    except Exception:
                        continue

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No previous optimization run found. Please execute an optimization run first via POST /api/optimize.",
        )

    @classmethod
    def export_allocation_csv(cls, run_id: Optional[str] = None) -> Tuple[str, str]:
        """Export optimization allocation records as a CSV formatted string and return (csv_content, filename)."""
        run = cls.get_latest_or_default_run(run_id)
        records = [r.model_dump() for r in run.allocation]
        if not records:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization run '{run.run_id}' contains no allocation records to export.",
            )

        df_export = pd.DataFrame(records)
        cols_order = [
            "customer_id",
            "recommended_discount",
            "predicted_purchase_probability",
            "expected_revenue",
            "expected_discount_cost",
            "expected_net_profit",
        ]
        present_cols = [c for c in cols_order if c in df_export.columns] + [
            c for c in df_export.columns if c not in cols_order
        ]
        df_export = df_export[present_cols]
        csv_str = df_export.to_csv(index=False)
        filename = f"optimized_discount_allocation_{run.run_id[:8]}.csv"
        return csv_str, filename

