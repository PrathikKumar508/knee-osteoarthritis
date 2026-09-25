"""
OAI Dataset Inspector & Schema Validator
========================================
Stage 1 Tool: Automatically scans data/raw/ for Osteoarthritis Initiative (OAI)
clinical and radiographic files, inspects their structure, profiles missingness,
detects participant identifiers, visit timepoints, baseline clinical features,
follow-up outcome variables, and join keys.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_RAW_DIR = Path("data/raw")
REPORT_OUTPUT_PATH = Path("reports/results/oai_schema_report.md")


def detect_file_delimiter(filepath: Path) -> str:
    """Detect file delimiter for text-based datasets."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        if '\t' in first_line:
            return '\t'
        elif '|' in first_line:
            return '|'
        elif ';' in first_line:
            return ';'
        else:
            return ','


def inspect_single_file(filepath: Path) -> dict:
    """Inspect a single data file and return detailed column profiling."""
    ext = filepath.suffix.lower()
    info = {
        "filename": filepath.name,
        "path": str(filepath),
        "file_size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
        "num_rows": 0,
        "num_cols": 0,
        "columns": [],
        "id_candidates": [],
        "visit_candidates": [],
        "baseline_candidates": [],
        "followup_candidates": [],
        "leakage_candidates": [],
        "missing_summary": {},
        "error": None
    }

    try:
        if ext in ['.csv', '.txt', '.tsv', '.dat', '.asc']:
            sep = detect_file_delimiter(filepath)
            df = pd.read_csv(filepath, sep=sep, low_memory=False, nrows=5000)
            # Get total line count estimate if large
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                info["num_rows"] = sum(1 for _ in f) - 1
        elif ext == '.sas7bdat':
            df = pd.read_sas(filepath, format='sas7bdat', encoding='utf-8')
            info["num_rows"] = len(df)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, nrows=5000)
            info["num_rows"] = len(df)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
            info["num_rows"] = len(df)
        else:
            info["error"] = f"Unsupported file extension: {ext}"
            return info

        info["num_cols"] = len(df.columns)
        cols = list(df.columns)
        info["columns"] = cols

        # Calculate missingness percentages
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        info["missing_summary"] = missing_pct

        # Classification patterns (case-insensitive)
        for col in cols:
            col_lower = str(col).lower()
            
            # Identifier candidates
            if any(k in col_lower for k in ['id', 'patient', 'subject', 'read', 'side', 'knee']):
                info["id_candidates"].append(col)
                
            # Visit / timepoint candidates
            if any(k in col_lower for k in ['visit', 'v00', 'v01', 'v03', 'v06', 'month', 'year', 'timepoint']):
                info["visit_candidates"].append(col)
                
            # Baseline feature candidates
            if any(k in col_lower for k in ['age', 'sex', 'gender', 'bmi', 'height', 'weight', 'womac', 'koos', 'pase', 'kl', 'grade', 'jsn', 'pain', 'stiff', 'func', 'injur', 'surg', 'base', 'v00']):
                if not any(k in col_lower for k in ['v01', 'v03', 'v06', 'fu', 'followup', '48m', '24m', '36m', 'outcome', 'progression']):
                    info["baseline_candidates"].append(col)

            # Follow-up outcome candidates
            if any(k in col_lower for k in ['v01', 'v03', 'v06', 'v08', 'v10', '48m', '24m', '36m', '12m', 'fu', 'followup', 'progression', 'tkr', 'replacement']):
                info["followup_candidates"].append(col)
                
            # Data leakage risk candidates (follow-up features that shouldn't be baseline inputs)
            if any(k in col_lower for k in ['v01_', 'v03_', 'v06_', 'v08_', 'v10_', '_48m', '_24m', '_36m', 'fu_']) and any(k in col_lower for k in ['womac', 'pain', 'kl', 'jsn', 'score']):
                info["leakage_candidates"].append(col)

    except Exception as e:
        info["error"] = str(e)

    return info


def run_stage_1_inspection(data_dir: Path = DATA_RAW_DIR, report_path: Path = REPORT_OUTPUT_PATH) -> list:
    """Run Stage 1 dataset inspection over all files in data/raw/ and generate markdown report."""
    print("=" * 70)
    print("  STAGE 1: OAI DATASET INSPECTION & SCHEMA VALIDATOR")
    print("=" * 70)

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    data_files = []
    for ext in ['*.csv', '*.txt', '*.tsv', '*.sas7bdat', '*.xlsx', '*.parquet']:
        data_files.extend(list(data_dir.glob(ext)))

    if not data_files:
        print(f"\n[!] WARNING: No data files found in '{data_dir.resolve()}'.")
        print("    Please place your Osteoarthritis Initiative (OAI) clinical and radiographic data files")
        print("    (e.g., AllClinical00.txt, kxr_sq_bu00.txt, or processed .csv files) into 'data/raw/'.\n")
        
        # Generate initial status report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_content = f"""# OAI Dataset Schema Report (Stage 1)

**Status**: Pending Dataset Placement  
**Scan Directory**: `{data_dir.resolve()}`  
**Files Detected**: 0  

### Next Steps for User:
1. Download OAI clinical dataset files from [NIH NIMH Data Archive (NDA)](https://nda.nih.gov/oai/).
2. Place raw `.txt`, `.csv`, or `.sas7bdat` files into `data/raw/`.
3. Re-run dataset inspection:
   ```bash
   python -m src.inspect_dataset
   ```
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return []

    print(f"Detected {len(data_files)} raw data file(s) in '{data_dir}':")
    for f in data_files:
        print(f" - {f.name}")

    file_profiles = []
    report_md = [
        "# OAI Dataset Schema Report (Stage 1)",
        f"**Directory Inspected**: `{data_dir.resolve()}`",
        f"**Total Files Profiled**: {len(data_files)}\n",
        "---",
        "## File Profiling Summary\n"
    ]

    for fpath in data_files:
        print(f"\nProfiling '{fpath.name}'...")
        profile = inspect_single_file(fpath)
        file_profiles.append(profile)

        if profile["error"]:
            print(f"  [!] Error reading file: {profile['error']}")
            report_md.append(f"### File: `{profile['filename']}`")
            report_md.append(f"> **Error**: {profile['error']}\n")
            continue

        print(f"  Rows: {profile['num_rows']:,} | Cols: {profile['num_cols']} | Size: {profile['file_size_mb']} MB")
        print(f"  Participant ID Candidates: {profile['id_candidates'][:5]}")
        print(f"  Visit/Timepoint Candidates: {profile['visit_candidates'][:5]}")

        report_md.extend([
            f"### File: `{profile['filename']}`",
            f"- **File Size**: {profile['file_size_mb']} MB",
            f"- **Estimated Rows**: {profile['num_rows']:,}",
            f"- **Columns Count**: {profile['num_cols']}",
            f"- **Participant ID Candidates**: `{profile['id_candidates']}`",
            f"- **Visit/Timepoint Candidates**: `{profile['visit_candidates']}`",
            f"- **Baseline Clinical Candidates**: `{profile['baseline_candidates'][:10]}`",
            f"- **Follow-up Outcome Candidates**: `{profile['followup_candidates'][:10]}`",
            f"- **Data Leakage Risk Candidates**: `{profile['leakage_candidates'][:10]}`\n",
            "#### Column Details (Top Missingness):"
        ])

        # Top missing columns
        missing_sorted = sorted(profile["missing_summary"].items(), key=lambda x: x[1], reverse=True)[:10]
        missing_table = "| Column Name | Missing % |\n| --- | --- |\n"
        for col_name, pct in missing_sorted:
            missing_table += f"| `{col_name}` | {pct}% |\n"
        report_md.append(missing_table + "\n---\n")

    # Assess join potential across files
    report_md.append("## Cross-File Join & Integration Potential\n")
    if len(file_profiles) > 1:
        id_sets = {}
        for p in file_profiles:
            if not p["error"] and p["id_candidates"]:
                id_sets[p["filename"]] = p["id_candidates"]
        report_md.append("Detected participant identifier keys across files:\n")
        for fname, keys in id_sets.items():
            report_md.append(f"- `{fname}`: {keys}")
    else:
        report_md.append("Single dataset file detected. No multi-file join required.")

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_md))

    print("\n" + "=" * 70)
    print(f"Stage 1 Inspection Complete. Schema report saved to: {report_path}")
    print("=" * 70)

    return file_profiles


if __name__ == "__main__":
    run_stage_1_inspection()
