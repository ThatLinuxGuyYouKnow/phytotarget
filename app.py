"""PhytoTarget ML v2 — RF baselines + GNN comparison, ranked disease predictions."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

st.set_page_config(page_title="PhytoTarget ML", page_icon="🌿", layout="wide")

TARGETS = {
    "ache":  ("AChE", "Alzheimer's"),
    "bace1": ("BACE1", "Alzheimer's"),
    "gsk3b": ("GSK3B", "Alzheimer's"),
    "ptgs2": ("COX-2", "Inflammation"),
    "tnf":   ("TNF-α", "Inflammation"),
}
ALZ = ["ache", "bace1", "gsk3b"]
INFLAM = ["ptgs2", "tnf"]
# 5-fold CV (RF) / held-out test (GNN), same ChEMBL labeled sets
RF_AUC = {"ache": 0.947, "bace1": 0.954, "gsk3b": 0.917, "ptgs2": 0.893, "tnf": 0.971}
GNN_AUC = {"ache": 0.874, "bace1": 0.861, "gsk3b": 0.627, "ptgs2": 0.635, "tnf": 0.951}

@st.cache_resource
def load_rf():
    models = {}
    for t in TARGETS:
        p = Path(f"models/{t}_rf.pkl")
        if p.exists():
            with open(p, "rb") as f:
                models[t] = pickle.load(f)
    return models

@st.cache_resource
def load_gnn():
    """Returns {target: net} or {} when torch/weights are unavailable."""
    try:
        import torch  # noqa: F401
        from src.gnn_model import load_gcn
    except ImportError:
        return {}
    nets = {}
    for t in TARGETS:
        p = Path(f"gnn_models/{t}_gcn.pt")
        if p.exists():
            try:
                nets[t] = load_gcn(p)
            except Exception:
                pass
    return nets

@st.cache_data
def load_table():
    return pd.read_csv("data/plant_preds_all.csv")

@st.cache_data
def load_gnn_csv():
    p = Path("data/gnn_plant_preds.csv")
    return pd.read_csv(p) if p.exists() else None

def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None
    return (np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.uint8), m)

st.title("🌿 PhytoTarget ML")
st.caption("Plant compound screening: fingerprints + RandomForest vs Graph Neural Network — "
           "EthnoHack 2026 · Track 2")
st.warning("⚠️ Research prototype. Predictions are unvalidated computational estimates, "
           "not medical facts. Not for clinical use.")

rf_models = load_rf()
gnn_nets = load_gnn()
plants = load_table()
gnn_csv = load_gnn_csv()

st.sidebar.success(f"{len(rf_models)}/5 RF models · {len(gnn_nets)}/5 GNN weights loaded")
with st.sidebar.expander("Model scorecard (AUC)"):
    score = pd.DataFrame({"target": list(TARGETS),
                          "RF (CV)": [RF_AUC[t] for t in TARGETS],
                          "GNN (test)": [GNN_AUC[t] for t in TARGETS]})
    st.dataframe(score, hide_index=True)
    st.caption("RF outperforms the small GCN on 4/5 targets; near-tie on TNF. "
               "Disease ranking uses RF (better-validated); GNN shown for comparison.")

mode = st.radio("Input", ["Pick an example plant compound", "Paste a SMILES"], horizontal=True)
gnn_lookup = None
if mode.startswith("Pick"):
    choice = st.selectbox("Compound", plants["name"].tolist(),
                          index=int(plants["name"].tolist().index("galantamine")))
    row = plants[plants["name"] == choice].iloc[0]
    smiles, plant, use = row["smiles"], row["plant_source"], row["traditional_use"]
    if gnn_csv is not None and choice in gnn_csv["name"].values:
        grow = gnn_csv[gnn_csv["name"] == choice].iloc[0]
        gnn_lookup = {t: float(grow[f"{t}_gnn_prob"]) for t in TARGETS}
else:
    smiles = st.text_input("SMILES", "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O")
    plant, use = "—", "—"

vec, mol = fp(smiles)
if mol is None:
    st.error("Invalid SMILES — could not parse structure.")
    st.stop()

rf_probs = {t: float(rf_models[t].predict_proba(vec.reshape(1, -1))[0, 1]) for t in rf_models}
if gnn_lookup is not None:
    gnn_probs = gnn_lookup
elif gnn_nets:
    from src.gnn_model import gnn_prob as _gp
    gnn_probs = {t: _gp(net, smiles) for t, net in gnn_nets.items()}
else:
    gnn_probs = {}

alz = float(np.mean([rf_probs[t] for t in ALZ if t in rf_probs]))
infl = float(np.mean([rf_probs[t] for t in INFLAM if t in rf_probs]))
ranked = sorted([("Alzheimer's relevance", alz), ("Inflammation relevance", infl)],
                key=lambda x: -x[1])

left, right = st.columns([1, 1.2])
with left:
    st.image(Draw.MolToImage(mol, size=(380, 380)), caption="Structure")
    st.write(f"**MW:** {Descriptors.MolWt(mol):.1f} · **LogP:** {Descriptors.MolLogP(mol):.2f} · "
             f"**H-donors:** {Descriptors.NumHDonors(mol)} · **H-acceptors:** {Descriptors.NumHAcceptors(mol)}")
    st.write(f"**Plant source:** {plant}")
    st.write(f"**Traditional use:** {use}")

with right:
    st.subheader("Ranked target diseases (RF)")
    for i, (disease, score) in enumerate(ranked, 1):
        medal = "🥇" if i == 1 else "🥈"
        st.metric(f"{medal} #{i} — {disease}", f"{score:.2f}")
    st.divider()
    st.subheader("Per-target: RF vs GNN")
    for t, (label, disease) in TARGETS.items():
        p = rf_probs.get(t, float("nan"))
        g = gnn_probs.get(t)
        line = f"**{label}** ({disease}) — RF {p:.2f}"
        if g is not None:
            line += f" · GNN {g:.2f}"
        st.write(line)
        st.progress(min(max(p, 0.0), 1.0))
    if not gnn_probs and mode.startswith("Paste"):
        st.caption("GNN inference needs torch + weights on this machine — showing RF only "
                   "for custom SMILES. Example compounds carry precomputed GNN scores.")

st.divider()
st.subheader("All example compounds (RF disease scores)")
view = plants[["name", "ache_active_prob", "bace1_active_prob", "gsk3b_active_prob",
               "ptgs2_active_prob", "tnf_active_prob", "alz_score", "inflam_score"]].round(2)
st.dataframe(view, use_container_width=True)
st.bar_chart(plants.set_index("name")[["alz_score", "inflam_score"]])
st.caption("RF: RandomForest on Morgan fingerprints, ChEMBL bioactivities "
           "(AChE n=2263 · BACE1 n=2068 · GSK3B n=2625 · COX-2 n=2043 · TNF n=2422). "
           "GNN: 3-layer GCN trained on Kaggle 2×T4 (~15 min), same splits. "
           "Full comparison in data/gnn_plant_preds.csv.")
