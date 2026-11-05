# 🌿 PhytoTarget ML — Plant Compound Screening via Graph Neural Networks

EthnoHack 2026 · Track 2 (Natural products & drug discovery)

Type a plant compound (SMILES) or pick an example — get ranked target-disease
predictions across 5 proteins: **AChE, BACE1, GSK3B** (Alzheimer's) and
**COX-2, TNF-α** (inflammation).

> ⚠️ Research prototype. Predictions are unvalidated computational estimates,
> not medical facts. Not for clinical use.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1. fetch ChEMBL bioactivities (one target, ~2 min)
python src/chembl_fetch.py --target CHEMBL220 --max-records 3000
# 2. train baseline (~1 min CPU)
python src/baseline.py --data data/ache_labeled.csv --model-out models/ache_rf.pkl \
    --pred-out data/plant_preds_ache.csv --target-name ache
# 3. launch app
streamlit run app.py
```

Repeat steps 1–2 for `CHEMBL4822/262/230/1825` (see `data/targets.yaml`).

## GNN training (Kaggle GPU)

Paste `kaggle_notebook.py` cells into a Kaggle notebook (GPU T4 + internet on),
run end to end, download `gnn_models/` + `gnn_plant_preds.csv` into the repo.
(`modal_notebook.py` is the legacy Modal variant — Modal sunset notebooks.)

## Data

- Bioactivities: ChEMBL (EBI), CC-BY-SA 3.0 — fetched live via API, no redistribution.
- Example plant compounds: curated demo set (`data/plant_compounds.csv`) with
  traditional-use annotations (Ayurveda/TCM/folk), structures validated with RDKit.

## Results (all measured, same ChEMBL sets)

| target | n | RF CV AUC | GCN test AUC |
|---|---|---|---|
| AChE | 2263 | 0.947 | 0.874 |
| BACE1 | 2068 | 0.954 | 0.861 |
| GSK3B | 2625 | 0.917 | 0.627 |
| COX-2 | 2043 | 0.893 | 0.635 |
| TNF-α | 2422 | 0.971 | 0.951 |

RF (Morgan fingerprints) beats the small 3-layer GCN on 4/5 targets, near-tie on
TNF. Likely causes: modest dataset sizes favor fingerprints; GSK3B is 89% active
(class skew hurts the GCN more); GCN val curves were still rising at epoch 60 on
3 targets (mild under-training). The app ranks diseases with RF and shows GNN
side-by-side — both are hypotheses, neither is a finding.

GCN training: Kaggle 2×T4 GPU, ~15 min total, early stopping (patience 12).
Per-compound comparison: `data/gnn_plant_preds.csv`; weights: `gnn_models/`.

## Method

- Features: Morgan fingerprints (r=2, 2048-bit); per-target RandomForest baselines
  (5-fold CV ROC-AUC 0.89–0.97); 3-layer GCN comparison on Modal.
- Disease score: mean of member-target probabilities (Alzheimer's = AChE/BACE1/GSK3B;
  inflammation = COX-2/TNF-α).

## Known limitations

- Huperzine A (a real AChE inhibitor) scores moderate — our SMILES drops its
  stereochemistry, which its activity depends on.
- GSK3B training data is 89% active (class imbalance); probabilities skew high.
- COX-2 model under-scores aspirin-like scaffolds — likely coxib-dominated training data.
- No wet-lab validation; treat all outputs as hypotheses for screening, not findings.
