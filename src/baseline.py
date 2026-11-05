"""Baseline: Morgan fingerprints + RandomForest for one target."""
import argparse
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits), dtype=np.uint8)


def featurize(smiles_list, n_bits=2048):
    feats, keep = [], []
    for i, s in enumerate(smiles_list):
        fp = morgan_fp(s, n_bits)
        if fp is not None:
            feats.append(fp)
            keep.append(i)
    return np.array(feats), np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ache_labeled.csv")
    ap.add_argument("--model-out", default="models/ache_rf.pkl")
    ap.add_argument("--pred-out", default="data/plant_preds_ache.csv")
    ap.add_argument("--plant-csv", default="data/plant_compounds.csv")
    ap.add_argument("--target-name", default="ache")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X, keep = featurize(df["canonical_smiles"].tolist())
    y = df["label"].values[keep]
    print(f"Training on {len(y)} compounds ({y.sum()} active)")

    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                 class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"CV ROC-AUC: {roc_auc_score(y, proba):.3f}")
    print(f"CV avg-precision: {average_precision_score(y, proba):.3f}")
    print("Confusion matrix @0.5:", confusion_matrix(y, proba >= 0.5).tolist())

    clf.fit(X, y)
    with open(args.model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved {args.model_out}")

    plants = pd.read_csv(args.plant_csv)
    col = f"{args.target_name}_active_prob"
    Xp, keepp = featurize(plants["smiles"].tolist())
    preds = clf.predict_proba(Xp)[:, 1]
    plants[col] = -1.0
    plants.iloc[keepp, plants.columns.get_loc(col)] = preds
    plants.to_csv(args.pred_out, index=False)
    print(plants[["name", col]].to_string(index=False))
    print(f"Wrote {args.pred_out}")


if __name__ == "__main__":
    main()
