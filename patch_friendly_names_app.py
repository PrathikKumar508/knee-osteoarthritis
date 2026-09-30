"""
Fixes feature-name display in app/app.py: the existing friendly-name
dictionaries use exact suffixes like "V00KL_0", but the real one-hot encoded
column names come out as "V00KL_0.0" (float-style category labels), so every
categorical feature was falling through to its raw column name whenever it
showed up as a top driver.

Adds one shared prefix-matching helper and points both the driver panel and
the feature-impact chart at it.

Run once from the project root (~/KOA):
    python3 patch_friendly_names_app.py
"""

from pathlib import Path

path = "app/app.py"
text = Path(path).read_text()


def rep(old, new, count=1):
    global text
    n = text.count(old)
    if n != count:
        raise SystemExit(
            f"Expected {count} match(es), found {n}. Nothing was changed. "
            f"Paste the relevant lines of app/app.py so I can adjust this."
        )
    text = text.replace(old, new)


old_chart_def = '''def create_feature_contribution_chart(feature_contributions: dict, top_n: int = 7):
    friendly_names = {
        "V00AGE": "Age (Baseline)", "V00BMI": "BMI (Joint Loading)",
        "V00WOMKP": "WOMAC Pain Score", "V00WOMAD": "WOMAC Disability",
        "V00WOMST": "WOMAC Stiffness", "V00PASE": "Physical Activity (PASE)",
        "V00SEX_1": "Sex: Male", "V00SEX_2": "Sex: Female",
        "V00KL_0": "KL Grade 0 (Normal)", "V00KL_1": "KL Grade 1 (Doubtful)",
        "V00KL_2": "KL Grade 2 (Minimal OA)", "V00KL_3": "KL Grade 3 (Moderate OA)",
        "V00INJ_0": "No Prior Injury", "V00INJ_1": "Prior Knee Injury",
        "V00SURG_0": "No Prior Surgery", "V00SURG_1": "Prior Knee Surgery"
    }
    sorted_feats = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_feats.reverse()
    names = [friendly_names.get(k, k) for k, _ in sorted_feats]'''

new_chart_def = '''# Base variable name -> readable label. Matched by prefix so it survives
# whatever suffix the one-hot encoder puts on categorical columns
# (e.g. "V00KL_0", "V00KL_0.0", "V00KL_2.0" all resolve correctly, and the
# category value itself, not just its presence, is shown for KL/injury/surgery).
FRIENDLY_BASE_NAMES = {
    "V00AGE": "Age (Baseline)", "V00BMI": "BMI (Joint Loading)",
    "V00WOMKP": "WOMAC Pain Score", "V00WOMAD": "WOMAC Disability",
    "V00WOMST": "WOMAC Stiffness", "V00PASE": "Physical Activity (PASE)",
}
FRIENDLY_CATEGORY_LABELS = {
    "V00SEX": {"1": "Sex: Male", "2": "Sex: Female"},
    "V00KL": {"0": "KL Grade 0 (Normal)", "1": "KL Grade 1 (Doubtful)",
              "2": "KL Grade 2 (Minimal OA)", "3": "KL Grade 3 (Moderate OA)"},
    "V00INJ": {"0": "No Prior Injury", "1": "Prior Knee Injury"},
    "V00SURG": {"0": "No Prior Surgery", "1": "Prior Knee Surgery"},
}


def friendly_feature_name(fname: str) -> str:
    if fname in FRIENDLY_BASE_NAMES:
        return FRIENDLY_BASE_NAMES[fname]
    for base, labels in FRIENDLY_CATEGORY_LABELS.items():
        if fname.startswith(base + "_"):
            suffix = fname[len(base) + 1:]
            code = suffix.split(".")[0]  # "0.0" -> "0", "1" -> "1"
            return labels.get(code, fname)
    return fname


def create_feature_contribution_chart(feature_contributions: dict, top_n: int = 7):
    sorted_feats = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_feats.reverse()
    names = [friendly_feature_name(k) for k, _ in sorted_feats]'''

rep(old_chart_def, new_chart_def)

old_panel = '''                        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
                        friendly = {
                            "V00AGE": "Age", "V00BMI": "BMI", "V00WOMKP": "WOMAC Pain",
                            "V00WOMAD": "WOMAC Disability", "V00WOMST": "WOMAC Stiffness", "V00PASE": "Physical Activity"
                        }
                        rows_html = ""
                        for k, v in sorted_feats:
                            name = friendly.get(k, k)'''

new_panel = '''                        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
                        rows_html = ""
                        for k, v in sorted_feats:
                            name = friendly_feature_name(k)'''

rep(old_panel, new_panel)

Path(path).write_text(text)
print("patched app/app.py")
