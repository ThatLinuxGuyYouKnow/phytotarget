"""Fetch bioactivity data from ChEMBL for one target. Slice 1: AChE only."""
import argparse
import sys
import time
import requests
import pandas as pd

API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

STANDARD_TYPES = ["IC50", "Ki", "EC50", "Kd"]


def fetch_activities(chembl_id: str, max_records: int = 5000) -> pd.DataFrame:
    rows = []
    offset = 0
    limit = 1000
    session = requests.Session()
    while len(rows) < max_records:
        params = {
            "target_chembl_id": chembl_id,
            "limit": limit,
            "offset": offset,
        }
        r = session.get(API, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        acts = payload.get("activities", [])
        if not acts:
            break
        for a in acts:
            if a.get("standard_type") not in STANDARD_TYPES:
                continue
            if a.get("standard_value") is None or a.get("standard_units") != "nM":
                continue
            if not a.get("canonical_smiles"):
                continue
            rows.append({
                "target_chembl_id": chembl_id,
                "standard_type": a["standard_type"],
                "standard_value_nM": float(a["standard_value"]),
                "canonical_smiles": a["canonical_smiles"],
                "assay_chembl_id": a.get("assay_chembl_id"),
                "molecule_chembl_id": a.get("molecule_chembl_id"),
            })
            if len(rows) >= max_records:
                break
        print(f"  ... {len(rows)} usable records so far (offset {offset})", flush=True)
        # paginate via next link presence
        if not payload.get("page_meta", {}).get("next"):
            break
        offset += limit
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates(subset=["canonical_smiles", "standard_type"])
    return df


def label(df: pd.DataFrame, active_nm: float = 10000.0) -> pd.DataFrame:
    """Active = any measurement <= active_nm (default 10uM). Median-aggregate per compound."""
    df = df.copy()
    df["active_measurement"] = df["standard_value_nM"] <= active_nm
    agg = df.groupby("canonical_smiles").agg(
        n_measurements=("active_measurement", "size"),
        median_nM=("standard_value_nM", "median"),
        frac_active=("active_measurement", "mean"),
    ).reset_index()
    agg["label"] = (agg["median_nM"] <= active_nm).astype(int)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="CHEMBL220")
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--active-nm", type=float, default=10000.0)
    ap.add_argument("--out", default="data/ache_raw.csv")
    ap.add_argument("--labeled-out", default="data/ache_labeled.csv")
    args = ap.parse_args()

    print(f"Fetching {args.target} ...")
    df = fetch_activities(args.target, args.max_records)
    print(f"Got {len(df)} usable activity rows")
    if len(df) == 0:
        sys.exit("No data — check target ID or API status")
    df.to_csv(args.out, index=False)
    lab = label(df, args.active_nm)
    print(f"Labeled {len(lab)} unique compounds: "
          f"{lab['label'].sum()} active, {(1 - lab['label']).sum()} inactive")
    lab.to_csv(args.labeled_out, index=False)
    print(f"Wrote {args.out} + {args.labeled_out}")


if __name__ == "__main__":
    main()
