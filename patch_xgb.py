from pathlib import Path
p = Path("src/compare_models.py")
s = p.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

rep('''from sklearn.pipeline import Pipeline
''', '''from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
''')

rep('''        random_state=config.RANDOM_STATE, n_jobs=-1),
}''', '''        random_state=config.RANDOM_STATE, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1),
}''')
p.write_text(s)
print("patched")
