"""
Great Expectations Validation Script
Assignment 2 - MAI201 MLOps
Somayeh Balashi - 147025241
"""

import great_expectations as gx
import pandas as pd
import re
import os
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "customer_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ge_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(filepath):
    """Load the raw dataset as strings so no information is lost/altered."""
    return pd.read_csv(filepath, dtype=str, keep_default_na=False)


def is_missing(x):
    return x is None or str(x).strip() == ""


def analyze_data_quality(df):
    """Column-by-column, rule-based data quality analysis (independent of GE)."""
    issues = {}

    for col in df.columns:
        count = int(df[col].apply(is_missing).sum())
        if count > 0:
            issues[f"Missing values in '{col}'"] = count

    non_null_ids = df[~df["customer_id"].apply(is_missing)]
    dup_count = int(non_null_ids.duplicated(subset=["customer_id"]).sum())
    if dup_count > 0:
        issues["Duplicate customer_id records"] = dup_count

    def age_val(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    ages = df["age"].apply(age_val)
    age_oor = int(ages.apply(lambda v: v is not None and (v < 0 or v > 120)).sum())
    if age_oor > 0:
        issues["Out-of-range age (>120 or <0)"] = age_oor

    non_numeric_age = int(df["age"].apply(lambda x: (not is_missing(x)) and age_val(x) is None).sum())
    if non_numeric_age > 0:
        issues["Non-numeric age values"] = non_numeric_age

    email_pattern = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

    def bad_email(x):
        if is_missing(x):
            return False
        return not bool(email_pattern.match(x.strip()))

    invalid_emails = int(df["email"].apply(bad_email).sum())
    if invalid_emails > 0:
        issues["Invalid email format"] = invalid_emails

    def sal_val(x):
        try:
            return float(re.sub(r"[$,]", "", x))
        except (TypeError, ValueError):
            return None

    def bad_salary(x):
        if is_missing(x):
            return False
        v = sal_val(x)
        return v is None or v < 0

    bad_salary_count = int(df["salary"].apply(bad_salary).sum())
    if bad_salary_count > 0:
        issues["Invalid salary (negative or non-numeric)"] = bad_salary_count

    valid_countries = {"USA", "Canada", "UK", "Australia"}

    def bad_country(x):
        if is_missing(x):
            return False
        return x.strip() not in valid_countries

    invalid_country = int(df["country"].apply(bad_country).sum())
    if invalid_country > 0:
        issues["Invalid/unsupported country value"] = invalid_country

    def bad_date(x):
        if is_missing(x):
            return False
        try:
            pd.Timestamp(x)
            return False
        except (ValueError, TypeError):
            return True

    bad_dates = int(df["signup_date"].apply(bad_date).sum())
    if bad_dates > 0:
        issues["Invalid signup_date value"] = bad_dates

    formula_phones = int(
        df["phone"].apply(lambda x: (not is_missing(x)) and str(x).strip().startswith("=")).sum()
    )
    if formula_phones > 0:
        issues["Phone numbers corrupted by Excel formula injection (leading '=')"] = formula_phones

    return issues


def run_ge_validation(df):
    """Run GE 1.x validation using the Fluent API with a pandas datasource,
    using exactly the expectations required by the assignment instructions."""
    context = gx.get_context(mode="ephemeral")

    # GE only recognizes real NaN/None as "null" - our raw load keeps blanks
    # as empty strings so analyze_data_quality() can count them precisely.
    # For the GE batch specifically, convert blanks to actual NaN so the
    # not-be-null expectations evaluate correctly.
    ge_df = df.replace("", pd.NA)

    datasource = context.data_sources.add_pandas("my_pandas_datasource")
    asset = datasource.add_dataframe_asset("customer_data_asset")

    batch_def = asset.add_batch_definition_whole_dataframe("full_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": ge_df})

    suite = gx.ExpectationSuite(name="customer_data_expectations")
    suite = context.suites.add(suite)

    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="customer_id"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=120),
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="email",
            regex=r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="salary", mostly=0.95),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="country",
            value_set=["USA", "Canada", "UK", "Australia"],
        ),
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="signup_date",
            regex=r"^\d{4}-\d{2}-\d{2}$",
        ),
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=500, max_value=1000),
    ]

    for exp in expectations:
        suite.add_expectation(exp)

    vd = context.validation_definitions.add(
        gx.ValidationDefinition(name="my_validation", data=batch_def, suite=suite)
    )

    results = vd.run(batch_parameters={"dataframe": ge_df})
    return results


def format_results(results):
    rows = []
    for r in results.results:
        rows.append({
            "column": r.expectation_config.kwargs.get("column", "table"),
            "expectation": r.expectation_config.type,
            "success": r.success,
        })
    return rows


def save_html_report(result_rows, issues, row_count, output_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = sum(1 for r in result_rows if r["success"])
    failed = len(result_rows) - passed

    rows_html = ""
    for r in result_rows:
        status = "PASSED" if r["success"] else "FAILED"
        cls = "pass" if r["success"] else "fail"
        rows_html += f'<tr class="{cls}"><td>{r["column"]}</td><td><code>{r["expectation"]}</code></td><td class="status">{status}</td></tr>'

    issues_html = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in issues.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>GE Validation Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5;color:#333}}
h1{{color:#1a3a5c;border-bottom:3px solid #2E75B6;padding-bottom:10px}}
h2{{color:#2E75B6;margin-top:40px}}
.summary{{display:flex;gap:20px;margin:20px 0}}
.card{{background:white;border-radius:8px;padding:20px 30px;box-shadow:0 2px 6px rgba(0,0,0,.1)}}
.card.green{{border-left:6px solid #27ae60}}.card.red{{border-left:6px solid #e74c3c}}
.card h3{{margin:0;font-size:2em}}.card p{{margin:4px 0 0;color:#666}}
table{{border-collapse:collapse;width:100%;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,.1)}}
th{{background:#1a3a5c;color:white;padding:12px 16px;text-align:left}}
td{{padding:10px 16px;border-bottom:1px solid #eee}}
tr.pass{{background:#f0fff4}}tr.fail{{background:#fff5f5}}
.status{{font-weight:bold}}.pass .status{{color:#27ae60}}.fail .status{{color:#e74c3c}}
code{{background:#eef;padding:2px 6px;border-radius:4px;font-size:.85em}}
.meta{{color:#888;font-size:.9em;margin-top:8px}}
</style></head><body>
<h1>Great Expectations - Data Quality Report</h1>
<p class="meta">Generated: {timestamp} | Suite: <strong>customer_data_expectations</strong> | Dataset: <strong>customer_data.csv</strong> ({row_count} rows)</p>
<h2>Summary</h2>
<div class="summary">
<div class="card green"><h3>{passed}</h3><p>Expectations Passed</p></div>
<div class="card red"><h3>{failed}</h3><p>Expectations Failed</p></div>
</div>
<h2>Expectation Results</h2>
<table><tr><th>Column</th><th>Expectation</th><th>Status</th></tr>{rows_html}</table>
<h2>Data Quality Issues Found</h2>
<table><tr><th>Issue</th><th>Count</th></tr>{issues_html}</table>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved -> {output_path}")


def print_summary(result_rows, issues):
    print("\n" + "=" * 65)
    print("  GREAT EXPECTATIONS VALIDATION RESULTS")
    print("=" * 65)
    passed = sum(1 for r in result_rows if r["success"])
    print(f"  Total: {len(result_rows)}  |  Passed: {passed}  |  Failed: {len(result_rows)-passed}")
    print("-" * 65)
    for r in result_rows:
        st = "PASS" if r["success"] else "FAIL"
        print(f"  [{st}] {r['column']}: {r['expectation']}")
    print("\n" + "=" * 65)
    print("  DATA QUALITY ISSUES")
    print("=" * 65)
    for k, v in issues.items():
        print(f"  - {k}: {v}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    print("Analyzing data quality issues...")
    issues = analyze_data_quality(df)

    print("Running Great Expectations validation...")
    ge_results = run_ge_validation(df)
    result_rows = format_results(ge_results)

    print_summary(result_rows, issues)

    save_html_report(result_rows, issues, len(df), os.path.join(OUTPUT_DIR, "ge_validation_report.html"))
    print("Done!")
