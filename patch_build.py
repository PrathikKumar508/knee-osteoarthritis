from pathlib import Path
p = Path("src/build_dataset.py")
s = p.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

rep('''def side_name(value):''', '''def to_number(series):
    """OAI values are plain numbers or 'code: label' text (e.g. '1: Male').
    Keep the leading number; missing codes like '.A: ...' become NaN."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"^\\s*(-?\\d+(?:\\.\\d+)?)", expand=False),
        errors="coerce",
    )


def side_name(value):''')

rep('''            wide[c] = pd.to_numeric(wide[c], errors="coerce")''',
    '''            wide[c] = to_number(wide[c])''')

rep('''    df = pd.read_csv(path, sep="|",
                     usecols=[lower["id"], lower["side"], lower[kl_col.lower()]],
                     low_memory=False)
    df.columns = [ID, "SIDE_RAW", kl_col]
''', '''    cols = [lower["id"], lower["side"], lower[kl_col.lower()]]
    names = [ID, "SIDE_RAW", kl_col]
    if "readprj" in lower:
        cols.append(lower["readprj"])
        names.append("READPRJ")
    df = pd.read_csv(path, sep="|", usecols=cols, low_memory=False)
    df = df[cols]
    df.columns = names
''')

rep('''    df[kl_col] = pd.to_numeric(df[kl_col], errors="coerce")
    df = df.dropna(subset=[SIDE]).drop(columns="SIDE_RAW")

    dups = df.duplicated([ID, SIDE]).sum()
    if dups:
        print(f"  Dropping {dups} duplicate (ID, SIDE) rows (keeping first)")
        df = df.drop_duplicates([ID, SIDE], keep="first")
''', '''    df[kl_col] = to_number(df[kl_col])
    df = df.dropna(subset=[SIDE]).drop(columns="SIDE_RAW")

    if "READPRJ" in df.columns:
        print(f"  READPRJ values: {df['READPRJ'].value_counts(dropna=False).to_dict()}")
    conflicts = (df.groupby([ID, SIDE])[kl_col].nunique() > 1).sum()
    print(f"  Knees whose duplicate rows disagree on KL: {conflicts}")

    dups = df.duplicated([ID, SIDE]).sum()
    if dups:
        print(f"  Dropping {dups} duplicate (ID, SIDE) rows (preferring rows that have a KL grade)")
        df = (df.sort_values([ID, SIDE, kl_col])
                .drop_duplicates([ID, SIDE], keep="first"))
    df = df[[ID, SIDE, kl_col]]
''')
p.write_text(s)
print("patched")
