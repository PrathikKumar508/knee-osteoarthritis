from pathlib import Path

path = "app/app.py"
text = Path(path).read_text()

old = '''FRIENDLY_FEATURE_NAMES = {
    "V00AGE": "age", "V00BMI": "BMI", "V00WOMKP": "WOMAC pain score",
    "V00WOMAD": "WOMAC disability score", "V00WOMST": "WOMAC stiffness score",
    "V00PASE": "physical activity level", "V00SEX_1": "sex", "V00SEX_2": "sex",
    "V00KL_0": "baseline KL grade", "V00KL_1": "baseline KL grade",
    "V00KL_2": "baseline KL grade", "V00KL_3": "baseline KL grade",
    "V00INJ_0": "injury history", "V00INJ_1": "injury history",
    "V00SURG_0": "surgery history", "V00SURG_1": "surgery history",
}'''

new = '''# Base variable name -> readable label. Matched by prefix so it survives
# whatever suffix the one-hot encoder puts on categorical columns
# (e.g. "V00KL_0", "V00KL_0.0", "V00KL_2.0" all map to "baseline KL grade").
FRIENDLY_BASE_NAMES = {
    "V00AGE": "age", "V00BMI": "BMI", "V00WOMKP": "WOMAC pain score",
    "V00WOMAD": "WOMAC disability score", "V00WOMST": "WOMAC stiffness score",
    "V00PASE": "physical activity level", "V00SEX": "sex",
    "V00KL": "baseline KL grade", "V00INJ": "injury history",
    "V00SURG": "surgery history",
}


def friendly_feature_name(fname: str) -> str:
    for base, label in FRIENDLY_BASE_NAMES.items():
        if fname == base or fname.startswith(base + "_"):
            return label
    return fname'''

n = text.count(old)
if n != 1:
    raise SystemExit(f"Expected 1 match, found {n}. Paste the FRIENDLY_FEATURE_NAMES block from app.py.")
text = text.replace(old, new)

old2 = "        label = FRIENDLY_FEATURE_NAMES.get(fname, fname)"
n2 = text.count(old2)
if n2 != 1:
    raise SystemExit(f"Expected 1 match, found {n2}. Paste the generate_personalized_driver_summary function from app.py.")
text = text.replace(old2, "        label = friendly_feature_name(fname)")

Path(path).write_text(text)
print("patched app/app.py")
