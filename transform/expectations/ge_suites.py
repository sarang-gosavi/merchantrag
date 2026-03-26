"""
transform/expectations/ge_suites.py — Great Expectations data quality suites.

Validates merchant transaction data at ingestion time.
Checks: null, MCC code validation, amount range, merchant_id referential integrity.

Run standalone:
    python transform/expectations/ge_suites.py --suite raw
    python transform/expectations/ge_suites.py --suite staging
    python transform/expectations/ge_suites.py --validate-file data/staging/merchant_txn_*.csv

Integrates with dbt via dbt-expectations package for CI/CD.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Valid MCC codes (ISO 18245 subset) ────────────────────────────────────────
VALID_MCC_CODES = {
    "5812", "5813", "5814", "5411", "5912", "7011",
    "5941", "5921", "5999", "7299", "7996", "4111",
    "5045", "5651", "5661", "5732", "5734",
}

VALID_ACCEPTANCE_METHODS = {
    "chip", "contactless", "online", "swipe", "mobile_pay", "keyed"
}

VALID_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}


# ── Expectation result ─────────────────────────────────────────────────────────

class ExpectationResult:
    def __init__(self, name: str, success: bool, observed: Any,
                 expected: Any = None, details: str = ""):
        self.name = name
        self.success = success
        self.observed = observed
        self.expected = expected
        self.details = details

    def to_dict(self) -> Dict:
        return {
            "expectation": self.name,
            "success": self.success,
            "observed": self.observed,
            "expected": self.expected,
            "details": self.details,
        }


class ExpectationSuite:
    """
    Lightweight Great Expectations-compatible suite.
    Mirrors GE API: expect_column_values_to_not_be_null, etc.
    Install great_expectations for production-grade HTML reports.
    """

    def __init__(self, suite_name: str, records: List[Dict]):
        self.suite_name = suite_name
        self.records = records
        self.results: List[ExpectationResult] = []
        self.run_at = datetime.utcnow().isoformat()

    def _col(self, col: str) -> List[Any]:
        return [r.get(col) for r in self.records]

    def expect_column_values_to_not_be_null(self, col: str) -> "ExpectationSuite":
        vals = self._col(col)
        nulls = sum(1 for v in vals if v is None or v == "")
        pct = nulls / len(vals) if vals else 0
        self.results.append(ExpectationResult(
            name=f"expect_column_values_to_not_be_null({col})",
            success=nulls == 0,
            observed=f"{nulls}/{len(vals)} null ({pct:.1%})",
            expected="0 nulls",
        ))
        return self

    def expect_column_values_to_be_in_set(self, col: str, value_set: set) -> "ExpectationSuite":
        vals = self._col(col)
        invalid = [v for v in vals if v is not None and str(v).lower() not in
                   {str(x).lower() for x in value_set}]
        self.results.append(ExpectationResult(
            name=f"expect_column_values_to_be_in_set({col})",
            success=len(invalid) == 0,
            observed=f"{len(invalid)} invalid values: {invalid[:5]}",
            expected=f"values in {sorted(value_set)[:5]}...",
        ))
        return self

    def expect_column_values_to_be_between(self, col: str, min_val: float, max_val: float) -> "ExpectationSuite":
        vals = [v for v in self._col(col) if v is not None]
        out_of_range = [v for v in vals if not (min_val <= float(v) <= max_val)]
        self.results.append(ExpectationResult(
            name=f"expect_column_values_to_be_between({col}, {min_val}, {max_val})",
            success=len(out_of_range) == 0,
            observed=f"{len(out_of_range)} out-of-range: {out_of_range[:5]}",
            expected=f"[{min_val}, {max_val}]",
        ))
        return self

    def expect_column_values_to_be_unique(self, col: str) -> "ExpectationSuite":
        vals = [v for v in self._col(col) if v is not None]
        dupes = len(vals) - len(set(vals))
        self.results.append(ExpectationResult(
            name=f"expect_column_values_to_be_unique({col})",
            success=dupes == 0,
            observed=f"{dupes} duplicates",
            expected="0 duplicates",
        ))
        return self

    def expect_column_pair_values_to_not_be_null(self, col_a: str, col_b: str) -> "ExpectationSuite":
        """Both columns must be non-null when col_a is non-null (referential integrity)."""
        issues = [
            r for r in self.records
            if r.get(col_a) is not None and r.get(col_b) is None
        ]
        self.results.append(ExpectationResult(
            name=f"expect_referential_integrity({col_a} → {col_b})",
            success=len(issues) == 0,
            observed=f"{len(issues)} records with {col_a} set but {col_b} null",
            expected="0 referential integrity violations",
        ))
        return self

    def expect_table_row_count_to_be_between(self, min_rows: int, max_rows: int) -> "ExpectationSuite":
        n = len(self.records)
        self.results.append(ExpectationResult(
            name="expect_table_row_count_to_be_between",
            success=min_rows <= n <= max_rows,
            observed=n,
            expected=f"[{min_rows}, {max_rows}]",
        ))
        return self

    def to_report(self) -> Dict:
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        return {
            "suite_name": self.suite_name,
            "run_at": self.run_at,
            "success": passed == total,
            "statistics": {
                "evaluated_expectations": total,
                "successful_expectations": passed,
                "unsuccessful_expectations": total - passed,
                "success_percent": f"{100 * passed / total:.1f}%" if total else "N/A",
            },
            "results": [r.to_dict() for r in self.results],
        }

    def print_report(self) -> bool:
        report = self.to_report()
        icon = "✅" if report["success"] else "❌"
        print(f"\n{icon} Suite: {self.suite_name} | "
              f"{report['statistics']['successful_expectations']}/{report['statistics']['evaluated_expectations']} passed")
        for r in self.results:
            status = "  ✓" if r.success else "  ✗"
            print(f"{status} {r.name}")
            if not r.success:
                print(f"      Observed: {r.observed}")
                print(f"      Expected: {r.expected}")
        return report["success"]


# ── Suite definitions ─────────────────────────────────────────────────────────

def build_raw_suite(records: List[Dict]) -> ExpectationSuite:
    """
    RAW layer suite — validates ingestion-time data quality.
    Catches schema violations before they pollute the warehouse.
    """
    suite = ExpectationSuite("raw_merchant_transactions", records)
    return (
        suite
        .expect_table_row_count_to_be_between(1, 10_000_000)
        .expect_column_values_to_not_be_null("transaction_id")
        .expect_column_values_to_not_be_null("merchant_id")
        .expect_column_values_to_not_be_null("transaction_amount")
        .expect_column_values_to_not_be_null("mcc_code")
        .expect_column_values_to_not_be_null("city")
        .expect_column_values_to_be_unique("transaction_id")
        .expect_column_values_to_be_in_set("mcc_code", VALID_MCC_CODES)
        .expect_column_values_to_be_in_set("acceptance_method", VALID_ACCEPTANCE_METHODS)
        .expect_column_values_to_be_between("transaction_amount", 0.01, 99999.99)
        .expect_column_values_to_be_between("stars", 1.0, 5.0)
        .expect_column_values_to_be_between("price_range", 1, 4)
        .expect_column_pair_values_to_not_be_null("merchant_id", "merchant_name")
    )


def build_staging_suite(records: List[Dict]) -> ExpectationSuite:
    """
    STAGING layer suite — validates post-transformation correctness.
    Enforces data contracts for downstream RAG pipeline.
    """
    suite = ExpectationSuite("stg_merchant_transactions", records)
    return (
        suite
        .expect_column_values_to_not_be_null("transaction_id")
        .expect_column_values_to_not_be_null("merchant_id")
        .expect_column_values_to_be_unique("transaction_id")
        .expect_column_values_to_be_in_set("transaction_value_tier", {"low", "mid", "high"})
        .expect_column_values_to_be_in_set("transaction_quarter", {"Q1", "Q2", "Q3", "Q4"})
        .expect_column_values_to_be_between("transaction_amount", 0.01, 99999.99)
    )


def validate_records(records: List[Dict], suite_name: str = "raw") -> Tuple[bool, Dict]:
    """Validate a list of records against the named suite."""
    builder = build_raw_suite if suite_name == "raw" else build_staging_suite
    suite = builder(records)
    report = suite.to_report()
    return report["success"], report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import csv
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["raw", "staging"], default="raw")
    parser.add_argument("--validate-file", help="Path to CSV file to validate")
    parser.add_argument("--db-path", default="./data/merchantrag.sqlite")
    args = parser.parse_args()

    if args.validate_file:
        with open(args.validate_file, encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
    else:
        conn = sqlite3.connect(args.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM raw_merchant_transactions LIMIT 10000")
        records = [dict(row) for row in cur.fetchall()]
        conn.close()

    if not records:
        print("No records found — run the pipeline first.")
        sys.exit(1)

    print(f"Validating {len(records):,} records with suite: {args.suite}")
    builder = build_raw_suite if args.suite == "raw" else build_staging_suite
    suite = builder(records)
    ok = suite.print_report()

    report_path = Path(f"./data/ge_report_{args.suite}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(suite.to_report(), indent=2))
    print(f"\nReport saved → {report_path}")

    sys.exit(0 if ok else 1)
