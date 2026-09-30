from pathlib import Path
p = Path("src/build_dataset.py")
s = p.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

rep('''SIDES = {"RIGHT": "R", "LEFT": "L"}
''', '''SIDES = {"RIGHT": "R", "LEFT": "L"}

# When a knee has several readings, prefer this reading project. Confirm what
# each READPRJ means in the Image Assessments documentation.
PRIMARY_READPRJ = 15
''')

rep('''        print(f"  Dropping {dups} duplicate (ID, SIDE) rows (preferring rows that have a KL grade)")
        df = (df.sort_values([ID, SIDE, kl_col])
                .drop_duplicates([ID, SIDE], keep="first"))''',
'''        print(f"  Dropping {dups} duplicate (ID, SIDE) rows (keeping READPRJ {PRIMARY_READPRJ} when it has a grade)")
        df = df.dropna(subset=[kl_col])
        pref = (to_number(df["READPRJ"]) != PRIMARY_READPRJ) if "READPRJ" in df.columns else 0
        df = (df.assign(_pref=pref).sort_values([ID, SIDE, "_pref"])
                .drop_duplicates([ID, SIDE], keep="first"))''')
p.write_text(s)
print("patched")
