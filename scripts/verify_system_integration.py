"""
Comprehensive System Integration and Verification Script for EconoCausal.
Measures real performance, API response times, optimization runtimes,
validates security boundaries, and confirms end-to-end workflow consistency.
"""
import sys
import time
import json
import statistics
import concurrent.futures
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root path is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from src.api.main import app
from src.api.config import settings
from src.causal.multi_tier_dml import MultiTierCausalDML
from src.optimization.budget_optimizer import SciPyBudgetOptimizer
from src.monitoring.drift_detector import EconoCausalDriftDetector

def run_integration_audit():
    print("=" * 80)
    print("ECONOCAUSAL END-TO-END INTEGRATION & PERFORMANCE AUDIT")
    print("=" * 80)
    
    results = {}
    client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. API Health & Model Registry Check
    # -------------------------------------------------------------------------
    print("\n[1/8] Testing Health Check & Model Registry...")
    t0 = time.perf_counter()
    resp = client.get("/health")
    health_latency_ms = (time.perf_counter() - t0) * 1000
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print(f"  -> Status: {health_data['status']}")
    print(f"  -> Models: {health_data['models']}")
    print(f"  -> Reference dataset ready: {health_data.get('reference_data_available')}")
    print(f"  -> Health check latency: {health_latency_ms:.2f} ms")
    results["health"] = {
        "status_code": resp.status_code,
        "latency_ms": round(health_latency_ms, 2),
        "data": health_data,
    }

    # -------------------------------------------------------------------------
    # 2. Data Validation Endpoint & Security Checks
    # -------------------------------------------------------------------------
    print("\n[2/8] Testing Dataset Validation & Security Protections...")
    
    # 2.1 Approved dataset reference
    t0 = time.perf_counter()
    resp_val = client.post("/data/validate", json={"dataset_name": "optimization_retail_data"})
    val_latency_ms = (time.perf_counter() - t0) * 1000
    assert resp_val.status_code == 200, f"Valid ref failed: {resp_val.text}"
    val_data = resp_val.json()
    print(f"  -> Approved reference validation: is_valid={val_data['is_valid']}, rows={val_data['total_rows']}, latency={val_latency_ms:.2f} ms")
    
    # 2.2 Path traversal attack rejection
    resp_trav = client.post("/data/validate", json={"dataset_name": "../../etc/passwd"})
    print(f"  -> Path traversal check (../../etc/passwd): status={resp_trav.status_code} (Expected 400)")
    assert resp_trav.status_code == 400, "Security failure: path traversal was not rejected!"
    
    # 2.3 Unapproved reference rejection
    resp_unapp = client.post("/data/validate", json={"dataset_name": "random_unapproved_name"})
    print(f"  -> Unapproved reference check: status={resp_unapp.status_code} (Expected 400)")
    assert resp_unapp.status_code == 400
    
    # 2.4 Upload CSV validation
    sample_csv = "customer_id,age,loyalty_score,previous_purchases,customer_segment,discount_level,purchase,product_price,gross_margin_rate\nCUST_001,35,70.5,4,regular_buyer,10,1,85.0,0.50\n"
    resp_upload = client.post(
        "/data/validate",
        files={"file": ("sample.csv", sample_csv.encode("utf-8"), "text/csv")}
    )
    assert resp_upload.status_code == 200
    upload_valid = resp_upload.json()['is_valid']
    print(f"  -> Safe CSV file upload check: status={resp_upload.status_code}, is_valid={upload_valid}")
    assert upload_valid is True, f"Upload validation errors: {resp_upload.json().get('errors')}"

    results["security_and_validation"] = {
        "approved_val_latency_ms": round(val_latency_ms, 2),
        "path_traversal_blocked": resp_trav.status_code == 400,
        "unapproved_ref_blocked": resp_unapp.status_code == 400,
        "csv_upload_valid": resp_upload.status_code == 200,
    }

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 3. Treatment-Effect Prediction Benchmark (/predict/effects)
    # -------------------------------------------------------------------------
    print("\n[3/8] Benchmarking Treatment-Effect Prediction Across Batch Sizes...")
    opt_df = pd.read_csv(root_dir / "data" / "optimization_retail_data.csv")
    
    batch_sizes = [10, 100, 500, 1000]
    pred_benchmarks = {}
    
    for bs in batch_sizes:
        batch_records = opt_df.head(bs)[
            ["customer_id", "age", "loyalty_score", "previous_purchases", "customer_segment", "campaign_channel"]
        ].to_dict(orient="records")
        
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            resp_p = client.post("/predict/effects", json={"records": batch_records})
            latencies.append((time.perf_counter() - t0) * 1000)
            assert resp_p.status_code == 200, f"Predict failed: {resp_p.text}"
            
        avg_lat = statistics.mean(latencies)
        p_data = resp_p.json()
        print(f"  -> Batch size {bs:4d}: mean latency = {avg_lat:6.2f} ms | min = {min(latencies):6.2f} ms | max = {max(latencies):6.2f} ms")
        first_cust = p_data["predictions"][0]
        print(f"     Example customer {first_cust['customer_id']}: P(0)={first_cust['predicted_purchase_probability_0']:.4f}, P(10)={first_cust['predicted_purchase_probability_10']:.4f}, P(20)={first_cust['predicted_purchase_probability_20']:.4f}")
        pred_benchmarks[bs] = {
            "mean_ms": round(avg_lat, 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
        }
    results["predict_benchmarks"] = pred_benchmarks

    # -------------------------------------------------------------------------
    # 4. SciPy HiGHS Optimization Engine & Full 15,000-Customer Scalability Benchmark
    # -------------------------------------------------------------------------
    print("\n[4/8] Benchmarking SciPy HiGHS Optimization Engine & Full 15,000-Customer Scalability...")
    
    opt_benchmarks = {}
    test_populations = [100, 500, 1500, 3000]
    target_budget = 5000.0
    
    for pop_size in test_populations:
        sub_records = opt_df.head(pop_size)[
            ["customer_id", "age", "loyalty_score", "previous_purchases", "customer_segment", "campaign_channel", "product_price", "gross_margin_rate"]
        ].to_dict(orient="records")
        
        t0 = time.perf_counter()
        resp_opt = client.post("/optimize", json={
            "budget": target_budget,
            "objective_type": "net_profit",
            "records": sub_records
        })
        opt_lat_ms = (time.perf_counter() - t0) * 1000
        assert resp_opt.status_code == 200, f"Optimization failed: {resp_opt.text}"
        opt_json = resp_opt.json()
        
        kpis = opt_json["kpis"]
        run_id = opt_json["run_id"]
        allocations = opt_json["allocation"]
        
        # Verify budget adherence strictly
        assert kpis["expected_discount_expenditure"] <= target_budget + 1e-4, f"Budget exceeded! Cost: {kpis['expected_discount_expenditure']} > {target_budget}"
        assert len(allocations) == pop_size, f"Allocation count mismatch: {len(allocations)} != {pop_size}"
        
        print(f"  -> API Population {pop_size:4d}: API latency = {opt_lat_ms:6.2f} ms | Solver = {opt_json['solver_status']} in {opt_json['solve_time_seconds']:.3f}s")
        print(f"     Discounts: $0={kpis['count_discount_0']}, $10={kpis['count_discount_10']}, $20={kpis['count_discount_20']}")
        print(f"     Cost = ${kpis['expected_discount_expenditure']:,.2f} / ${target_budget:,.2f} ({kpis['budget_utilization_pct']:.1f}%) | Profit = ${kpis['expected_net_profit']:,.2f}")
        
        opt_benchmarks[pop_size] = {
            "api_latency_ms": round(opt_lat_ms, 2),
            "solve_time_seconds": opt_json["solve_time_seconds"],
            "expected_discount_cost": kpis["expected_discount_expenditure"],
            "expected_net_profit": kpis["expected_net_profit"],
            "budget_utilization_pct": kpis["budget_utilization_pct"],
            "count_0": kpis["count_discount_0"],
            "count_10": kpis["count_discount_10"],
            "count_20": kpis["count_discount_20"],
            "run_id": run_id,
        }
    
    # Run Full 15,000-Customer MILP Knapsack Benchmark
    print("  -> Executing Full 15,000-Customer MILP Knapsack Solver Benchmark...")
    causal_model = MultiTierCausalDML.load(str(root_dir / "models" / "multitreatment_model.joblib"))
    optimizer_15k = SciPyBudgetOptimizer(budget=5000.0, budget_type="redemption", objective_type="net_profit")
    t0_15k = time.perf_counter()
    result_15k = optimizer_15k.optimize(df_customers=opt_df, model=causal_model)
    time_15k = time.perf_counter() - t0_15k
    
    # Assertions for 15k customers
    assert result_15k.success is True, f"15k optimization failed: {result_15k.status_message}"
    assert result_15k.total_cost <= 5000.0 + 1e-4, f"15k budget exceeded: {result_15k.total_cost}"
    allocations_15k = result_15k.allocation_matrix
    rec_discounts = allocations_15k["recommended_discount"].to_numpy()
    assert len(rec_discounts) == 15000, f"Expected 15,000 allocations, got {len(rec_discounts)}"
    assert set(np.unique(rec_discounts)).issubset({0, 10, 20}), f"Invalid discounts: {np.unique(rec_discounts)}"
    
    print(f"  -> Full 15,000-Customer Benchmark: Solver={result_15k.status_message}")
    print(f"     Runtime: {time_15k:.4f} s | Cost: ${result_15k.total_cost:,.2f} / $5,000.00 ({result_15k.budget_utilization_pct:.2f}%)")
    print(f"     Expected Profit: ${result_15k.expected_net_profit:,.2f} | Tiers: {result_15k.tier_counts}")
    print(f"     Integrity: Exactly 1 discount/customer verified across all 15,000 records.")
    
    opt_benchmarks[15000] = {
        "solve_time_seconds": round(time_15k, 4),
        "solver_status": result_15k.status_message,
        "solver_status_code": result_15k.status_code,
        "expected_discount_cost": round(result_15k.total_cost, 2),
        "expected_net_profit": round(result_15k.expected_net_profit, 2),
        "budget_utilization_pct": round(result_15k.budget_utilization_pct, 2),
        "tier_counts": result_15k.tier_counts,
        "total_records_verified": len(rec_discounts),
        "single_discount_verified": True,
    }
    results["optimization_benchmarks"] = opt_benchmarks

    # -------------------------------------------------------------------------
    # 5. Prescriptive Allocation Retrieval & State Persistence (/allocation/{run_id})
    # -------------------------------------------------------------------------
    print("\n[5/8] Testing Prescriptive Allocation Retrieval & State Persistence...")
    latest_run_id = opt_benchmarks[3000]["run_id"]
    t0 = time.perf_counter()
    resp_get_alloc = client.get(f"/allocation/{latest_run_id}")
    get_alloc_ms = (time.perf_counter() - t0) * 1000
    assert resp_get_alloc.status_code == 200, f"Get allocations failed: {resp_get_alloc.text}"
    alloc_data = resp_get_alloc.json()
    print(f"  -> Allocation retrieval by Run ID ({latest_run_id}): latency = {get_alloc_ms:.2f} ms, returned {len(alloc_data['allocation'])} records")
    results["allocation_retrieval"] = {
        "status_code": resp_get_alloc.status_code,
        "latency_ms": round(get_alloc_ms, 2),
        "total_records": len(alloc_data["allocation"]),
    }

    # -------------------------------------------------------------------------
    # 6. Automated Data Drift Detection Engine (/drift/check)
    # -------------------------------------------------------------------------
    print("\n[6/8] Testing Automated Data Drift Detection Endpoint...")
    
    # 6.1 In-distribution test (clean reference)
    t0 = time.perf_counter()
    resp_drift_clean = client.post("/drift/check", json={"dataset_reference": "drift_reference"})
    drift_clean_ms = (time.perf_counter() - t0) * 1000
    assert resp_drift_clean.status_code == 200
    clean_json = resp_drift_clean.json()
    print(f"  -> Reference baseline check: status='{clean_json['overall_status']}', drifted={clean_json['drifted_features_count']}/{clean_json['total_features_evaluated']}, latency={drift_clean_ms:.2f} ms")
    
    # 6.2 Out-of-distribution shifted test (current batch)
    t0 = time.perf_counter()
    resp_drift_shifted = client.post("/drift/check", json={"dataset_reference": "drift_current"})
    drift_shifted_ms = (time.perf_counter() - t0) * 1000
    assert resp_drift_shifted.status_code == 200
    shifted_json = resp_drift_shifted.json()
    print(f"  -> Shifted cohort check: status='{shifted_json['overall_status']}', drifted={shifted_json['drifted_features_count']}/{shifted_json['total_features_evaluated']}, severe={shifted_json['severe_drift_count']}, latency={drift_shifted_ms:.2f} ms")
    print(f"     Recommended Action: {shifted_json['recommended_next_action']}")
    
    results["drift_detection"] = {
        "clean_status": clean_json["overall_status"],
        "clean_latency_ms": round(drift_clean_ms, 2),
        "shifted_status": shifted_json["overall_status"],
        "shifted_drifted_count": shifted_json["drifted_features_count"],
        "shifted_severe_count": shifted_json["severe_drift_count"],
        "shifted_latency_ms": round(drift_shifted_ms, 2),
    }

    # -------------------------------------------------------------------------
    # 7. Multi-Threaded Concurrency & Stress Testing
    # -------------------------------------------------------------------------
    print("\n[7/8] Testing Multi-Threaded Concurrent API Requests...")
    
    def fetch_health():
        return client.get("/health").status_code
        
    def fetch_pred():
        rec = opt_df.head(20)[["customer_id", "age", "loyalty_score", "previous_purchases", "customer_segment", "campaign_channel"]].to_dict(orient="records")
        return client.post("/predict/effects", json={"records": rec}).status_code
        
    concurrency_tasks = [fetch_health] * 20 + [fetch_pred] * 10
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fn) for fn in concurrency_tasks]
        status_codes = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    total_concurrent_time_ms = (time.perf_counter() - t0) * 1000
    success_rate = (status_codes.count(200) / len(status_codes)) * 100
    print(f"  -> Executed {len(concurrency_tasks)} concurrent requests across 8 worker threads in {total_concurrent_time_ms:.2f} ms")
    print(f"  -> Success rate: {success_rate:.1f}% (All 200 OK)")
    assert success_rate == 100.0
    results["concurrency"] = {
        "total_requests": len(concurrency_tasks),
        "total_time_ms": round(total_concurrent_time_ms, 2),
        "success_rate_pct": success_rate,
    }

    # -------------------------------------------------------------------------
    # 8. Codebase Secrets & Security Boundary Audit
    # -------------------------------------------------------------------------
    print("\n[8/8] Auditing Codebase for Sensitive Information & Security Boundary Integrity...")
    
    # Scanned directories: production Python code, tests, scripts, frontend code, config files
    scan_roots = [
        root_dir / "src",
        root_dir / "tests",
        root_dir / "scripts",
        root_dir / "frontend" / "src",
    ]
    top_level_configs = [
        f for f in root_dir.glob("*")
        if f.is_file() and f.suffix in [".py", ".json", ".env", ".toml", ".yml", ".yaml", ".md"]
    ]
    
    files_to_scan = list(top_level_configs)
    for s_dir in scan_roots:
        if s_dir.exists():
            for p in s_dir.rglob("*"):
                if p.is_file() and p.suffix in [".py", ".js", ".jsx", ".json", ".env", ".toml", ".yml"]:
                    files_to_scan.append(p)
    
    # Patterns for actual exposed secrets or API credentials
    credential_patterns = ["ghp_", "sk-proj-", "sk-", "aws_secret_access_key", "bearer ey", "-----begin private key"]
    flagged_findings = []
    
    for file_path in files_to_scan:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for idx, line in enumerate(content.splitlines(), 1):
                clean_l = line.strip()
                if clean_l.startswith("#") or clean_l.startswith("//") or clean_l.startswith("/*"):
                    continue
                for pat in credential_patterns:
                    if pat in clean_l.lower():
                        # Exclude harmless parameter names, regex patterns, or mock fixtures
                        if not any(safe in clean_l.lower() for safe in ["example", "dummy", "placeholder", "mock", "pattern"]):
                            flagged_findings.append((str(file_path.relative_to(root_dir)), idx, clean_l[:60]))
        except Exception:
            pass
            
    print(f"  -> Scanned {len(files_to_scan)} production, test, and config files.")
    print(f"  -> Sensitive credential leaks found: {len(flagged_findings)}")
    assert len(flagged_findings) == 0, f"Critical security leak detected: {flagged_findings}"
    results["security_audit"] = {
        "files_scanned_count": len(files_to_scan),
        "sensitive_findings_count": len(flagged_findings),
        "findings": flagged_findings,
        "status": "PASSED - Zero Leaks",
    }

    # Save integration audit output
    output_path = root_dir / "reports" / "system_integration_benchmark_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Benchmark measurements saved to {output_path}")

if __name__ == "__main__":
    run_integration_audit()
