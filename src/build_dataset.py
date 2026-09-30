"""
Build the modeling dataset (one row per knee) from the raw OAI ASCII files.

Clinical baseline variables come from the wide AllClinical/Enrollees/SubjectChar
files (right and left knees in separate columns). KL grades come from the
X-ray semi-quantitative files KXR_SQ_BU00.txt (baseline) and KXR_SQ_BU06.txt
(follow-up), which have one row per knee with a SIDE column.

Output: data/processed/oai_processed.csv with the column names that
src/config.py expects (V00AGE, V00KL, V06KL, ...).

Run from the project root:
    python3 -m src.build_dataset

Check the column names and SIDE coding below against the OAI documentation
(CRFs/Workbooks and Image Assessments) before trusting the results.
"""

from functools import reduce

import pandas as pd

from src.config import config
from src.feature_engineering import save_processed_dataset

ID = config.PATIENT_ID_COL
SIDE = config.SIDE_COL

# Same value for both knees of a patient: model name -> OAI column
SHARED = {
    "V00AGE": "V00AGE",
    "V00SEX": "P02SEX",
    "V00BMI": "P01BMI",
    "V00PASE": "V00PASE",
}

# Different per knee: model name -> OAI column template ({s} = R or L)
PER_KNEE = {
    "V00WOMKP": "V00WOMKP{s}",
    "V00WOMAD": "V00WOMADL{s}",
    "V00WOMST": "V00WOMSTF{s}",
    "V00INJ": "P01INJ{s}",
    "V00SURG": "P01KSURG{s}",
}

SIDES = {"RIGHT": "R", "LEFT": "L"}

# When a knee has several readings, prefer this reading project. Confirm what
# each READPRJ means in the Image Assessments documentation.
PRIMARY_READPRJ = 15


def needed_columns():
    cols = list(SHARED.values())
    for tmpl in PER_KNEE.values():
        cols += [tmpl.format(s=s) for s in SIDES.values()]
    return cols


def load_needed_columns(needed):
    """Read only the needed columns, each from the first top-level file that has it."""
    remaining = set(needed)
    frames = []
    for path in sorted(config.RAW_DATA_DIR.glob("*.txt")):
        header = list(pd.read_csv(path, sep="|", nrows=0).columns)
        id_col = next((c for c in header if c.upper() == "ID"), None)
        if id_col is None:
            continue
        here = [c for c in header if c in remaining]
        if not here:
            continue
        df = pd.read_csv(path, sep="|", usecols=[id_col] + here, low_memory=False)
        df = df.rename(columns={id_col: ID})
        frames.append(df)
        remaining -= set(here)
        print(f"  {path.name}: {here}")

    if remaining:
        raise SystemExit(
            "\nThese clinical columns were not found in any file in data/raw/:\n  "
            + "\n  ".join(sorted(remaining))
            + "\n\nCheck the spelling against the OAI documentation."
        )

    wide = reduce(lambda a, b: a.merge(b, on=ID, how="outer"), frames)
    for c in wide.columns:
        if c != ID:
            # OAI missing-value codes (".", ".A", ...) become NaN
            wide[c] = to_number(wide[c])
    return wide


def to_number(series):
    """OAI values are plain numbers or 'code: label' text (e.g. '1: Male').
    Keep the leading number; missing codes like '.A: ...' become NaN."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def side_name(value):
    """Map the SIDE coding (1/2, '1: Right', 'R', ...) to RIGHT / LEFT."""
    s = str(value).strip().upper()
    if s.startswith("1") or s.startswith("R"):
        return "RIGHT"
    if s.startswith("2") or s.startswith("L"):
        return "LEFT"
    return None


def find_file(filename):
    """Find a file anywhere under data/raw/ (case-insensitive name match)."""
    matches = sorted(p for p in config.RAW_DATA_DIR.rglob("*.txt")
                     if p.name.lower() == filename.lower())
    if not matches:
        raise SystemExit(f"Could not find {filename} under {config.RAW_DATA_DIR}/")
    if len(matches) > 1:
        print(f"  Note: {len(matches)} copies of {filename} found; using {matches[0]}")
    return matches[0]


def load_kl(filename, kl_col):
    """Load one KL file: one row per knee (ID, SIDE, KL grade)."""
    path = find_file(filename)
    header = list(pd.read_csv(path, sep="|", nrows=0).columns)
    lower = {c.lower(): c for c in header}
    cols = [lower["id"], lower["side"], lower[kl_col.lower()]]
    names = [ID, "SIDE_RAW", kl_col]
    if "readprj" in lower:
        cols.append(lower["readprj"])
        names.append("READPRJ")
    df = pd.read_csv(path, sep="|", usecols=cols, low_memory=False)
    df = df[cols]
    df.columns = names

    print(f"\n{filename}: raw SIDE values -> {df['SIDE_RAW'].value_counts(dropna=False).to_dict()}")
    df[SIDE] = df["SIDE_RAW"].map(side_name)
    df[kl_col] = to_number(df[kl_col])
    df = df.dropna(subset=[SIDE]).drop(columns="SIDE_RAW")

    if "READPRJ" in df.columns:
        print(f"  READPRJ values: {df['READPRJ'].value_counts(dropna=False).to_dict()}")
    conflicts = (df.groupby([ID, SIDE])[kl_col].nunique() > 1).sum()
    print(f"  Knees whose duplicate rows disagree on KL: {conflicts}")

    dups = df.duplicated([ID, SIDE]).sum()
    if dups:
        print(f"  Dropping {dups} duplicate (ID, SIDE) rows (keeping READPRJ {PRIMARY_READPRJ} when it has a grade)")
        df = df.dropna(subset=[kl_col])
        pref = (to_number(df["READPRJ"]) != PRIMARY_READPRJ) if "READPRJ" in df.columns else 0
        df = (df.assign(_pref=pref).sort_values([ID, SIDE, "_pref"])
                .drop_duplicates([ID, SIDE], keep="first"))
    df = df[[ID, SIDE, kl_col]]
    print(f"  {kl_col} value counts: {df[kl_col].value_counts(dropna=False).sort_index().to_dict()}")
    return df


def check_followup_visit():
    """Sanity check: V06 should be about 4 years after baseline."""
    try:
        base = pd.read_csv(config.RAW_DATA_DIR / "AllClinical00.txt", sep="|",
                           usecols=["ID", "V00AGE"])
        fu = pd.read_csv(config.RAW_DATA_DIR / "AllClinical06.txt", sep="|",
                         usecols=["ID", "V06AGE"])
        m = base.merge(fu, on="ID")
        diff = (pd.to_numeric(m["V06AGE"], errors="coerce")
                - pd.to_numeric(m["V00AGE"], errors="coerce")).mean()
        print(f"\nMean age difference V06 - V00: {diff:.2f} years (expect about 4)")
    except Exception as e:
        print(f"\nVisit check skipped: {e}")


def to_one_row_per_knee(wide):
    parts = []
    for side_name_, s in SIDES.items():
        part = pd.DataFrame({ID: wide[ID], SIDE: side_name_})
        for model_name, oai_name in SHARED.items():
            part[model_name] = wide[oai_name]
        for model_name, tmpl in PER_KNEE.items():
            part[model_name] = wide[tmpl.format(s=s)]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    print("Locating clinical columns...")
    wide = load_needed_columns(needed_columns())
    print(f"\nParticipants with clinical data: {len(wide):,}")

    check_followup_visit()

    knees = to_one_row_per_knee(wide)
    print(f"Knee rows before adding KL: {len(knees):,}")

    kl0 = load_kl("KXR_SQ_BU00.txt", "V00XRKL")
    kl6 = load_kl("KXR_SQ_BU06.txt", "V06XRKL")
    knees = (knees.merge(kl0, on=[ID, SIDE], how="left")
                  .merge(kl6, on=[ID, SIDE], how="left")
                  .rename(columns={"V00XRKL": "V00KL", "V06XRKL": "V06KL"}))

    # Keep knees with both KL grades. Baseline KL 4 is removed later in
    # create_progression_target.
    knees = knees.dropna(subset=["V00KL", "V06KL"]).reset_index(drop=True)
    print(f"\nKnee rows with baseline and follow-up KL: {len(knees):,}")
    print("\nMissing values per column:")
    print(knees.isna().mean().round(3).to_string())

    save_processed_dataset(knees)
