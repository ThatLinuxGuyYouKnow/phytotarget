# %% [markdown]
# PhytoTarget ML — GNN training on Kaggle GPU (T4)
# Setup: Settings -> Accelerator = GPU, Internet = ON.
# Run cells top to bottom. Models + predictions land in /kaggle/working (download them after).

# %% Cell 1 — install GNN lib only (torch + CUDA + rdkit are preinstalled; do NOT reinstall torch)
!pip install -q torch_geometric 2>&1 | tail -2

# %% Cell 2 — imports + GPU check
import torch
assert torch.cuda.is_available(), "Enable GPU: Settings -> Accelerator -> GPU T4"
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0))

# %% Cell 3 — config
TARGETS = {
    "ache":  "CHEMBL220",   # Alzheimer's
    "bace1": "CHEMBL4822",  # Alzheimer's
    "gsk3b": "CHEMBL262",   # Alzheimer's
    "ptgs2": "CHEMBL230",   # Inflammation (COX-2)
    "tnf":   "CHEMBL1825",  # Inflammation
}
MAX_RECORDS = 3000
ACTIVE_NM = 10000.0
EPOCHS = 60
BATCH = 64
LR = 1e-3
OUT_DIR = "/kaggle/working/gnn_models"

# Demo plant compounds (same set as the repo) — used for GNN-vs-RF comparison output
PLANTS = {
    "curcumin": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O",
    "quercetin": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
    "resveratrol": "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
    "EGCG": "C1C(C(OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O",
    "berberine": "COC1=C(C2=C(CN3CCC4=CC5=C(C=C4C3=C2)OCO5)C=C1)OC",
    "artemisinin": "CC1CCC2C(C(=O)OC3C2C1CCC3(C)OO)C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "galantamine": "CN1CCC23C=CC(CC2OC4=C(C=CC(=C34)C1)OC)O",
    "huperzine A": "CC1=CC2C(CC1)C3(CCN2C)C(=CC(=O)N3)C",
}

# %% Cell 4 — data: use uploaded labeled CSVs if present, else fetch ChEMBL live
import os, glob, time, requests, pandas as pd

def fetch_target(chembl_id, max_records=3000):
    rows, offset, sess = [], 0, requests.Session()
    while len(rows) < max_records:
        r = sess.get("https://www.ebi.ac.uk/chembl/api/data/activity.json",
                     params={"target_chembl_id": chembl_id, "limit": 1000, "offset": offset},
                     timeout=60)
        r.raise_for_status()
        p = r.json()
        acts = p.get("activities", [])
        if not acts:
            break
        for a in acts:
            if a.get("standard_type") not in ("IC50", "Ki", "EC50", "Kd"):
                continue
            if a.get("standard_units") != "nM" or a.get("standard_value") is None:
                continue
            if not a.get("canonical_smiles"):
                continue
            rows.append((a["canonical_smiles"], float(a["standard_value"])))
            if len(rows) >= max_records:
                break
        if not p.get("page_meta", {}).get("next"):
            break
        offset += 1000
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["smiles", "value_nM"]).drop_duplicates("smiles")
    df["label"] = (df["value_nM"] <= ACTIVE_NM).astype(int)
    print(f"{chembl_id}: {len(df)} compounds, {df['label'].sum()} active")
    return df

DATA = {}
uploads = {os.path.basename(p).split("_")[0]: p
           for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)}
for name, cid in TARGETS.items():
    if name in uploads:
        df = pd.read_csv(uploads[name])
        df = df.rename(columns={"canonical_smiles": "smiles"})
        df["label"] = (df["median_nM"] <= ACTIVE_NM).astype(int)
        print(f"{name}: {len(df)} compounds from upload, {df['label'].sum()} active")
        DATA[name] = df[["smiles", "label"]]
    else:
        print(f"{name}: no upload found, fetching {cid} ...")
        d = fetch_target(cid, MAX_RECORDS)
        DATA[name] = d[["smiles", "label"]]

# %% Cell 5 — molecules -> graphs
from rdkit import Chem
try:
    from torch_geometric.loader import DataLoader
except ImportError:  # older PyG
    from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import numpy as np

ATOM_FEATS = 9

def mol_to_graph(smiles, label):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    xs = []
    for a in m.GetAtoms():
        hyb = int(a.GetHybridization()) if a.GetHybridization() is not None else 0
        xs.append([a.GetAtomicNum() / 50.0, a.GetDegree(), a.GetFormalCharge(),
                   a.GetTotalNumHs(), hyb / 7.0, float(a.GetIsAromatic()),
                   float(a.IsInRing()), a.GetMass() / 200.0, a.GetTotalValence() / 8.0])
    x = torch.tensor(xs, dtype=torch.float)
    ei, eb = [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
              Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}.get(b.GetBondType(), 1.0)
        ei += [[i, j], [j, i]]
        eb += [[bt, float(b.GetIsConjugated())], [bt, float(b.GetIsConjugated())]]
    if not ei:  # single-atom molecules
        ei, eb = [[0, 0]], [[1.0, 0.0]]
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(eb, dtype=torch.float),
                y=torch.tensor([label], dtype=torch.float))

def make_loaders(df, seed=42):
    graphs = [g for g in (mol_to_graph(s, l) for s, l in zip(df["smiles"], df["label"])) if g is not None]
    print(f"  {len(graphs)}/{len(df)} valid graphs")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(graphs))
    n = len(graphs)
    tr, va, te = idx[:int(.8*n)], idx[int(.8*n):int(.9*n)], idx[int(.9*n):]
    mk = lambda ii, sh: DataLoader([graphs[i] for i in ii], batch_size=BATCH, shuffle=sh)
    return mk(tr, True), mk(va, False), mk(te, False)

# %% Cell 6 — GCN model + training
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score

class GCN(nn.Module):
    def __init__(self, in_dim=ATOM_FEATS, hid=128):
        super().__init__()
        self.c1, self.c2, self.c3 = GCNConv(in_dim, hid), GCNConv(hid, hid), GCNConv(hid, hid)
        self.lin = nn.Linear(hid, 1)
    def forward(self, d):
        x, e = d.x, d.edge_index
        x = F.relu(self.c1(x, e)); x = F.relu(self.c2(x, e)); x = self.c3(x, e)
        return self.lin(global_mean_pool(x, d.batch)).squeeze(-1)

def train_one(name, df):
    dev = torch.device("cuda")
    tr, va, te = make_loaders(df)
    pos = df["label"].mean()
    pw = torch.tensor([(1-pos)/max(pos, 1e-6)], device=dev)
    net = GCN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, patience = -1.0, None, 0
    for ep in range(1, EPOCHS+1):
        net.train()
        for b in tr:
            b = b.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(b), b.y)
            loss.backward(); opt.step()
        net.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in va:
                b = b.to(dev)
                ps += torch.sigmoid(net(b)).cpu().tolist()
                ys += b.y.cpu().tolist()
        try:
            auc = roc_auc_score(ys, ps)
        except ValueError:
            auc = 0.5
        if auc > best:
            best, best_state, patience = auc, {k: v.cpu() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 10 == 0 or patience == 0:
            print(f"  [{name}] ep {ep}: val AUC {auc:.3f} (best {best:.3f})", flush=True)
        if patience >= 12:
            print(f"  [{name}] early stop at ep {ep}")
            break
    net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    net.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in te:
            b = b.to(dev)
            ps += torch.sigmoid(net(b)).cpu().tolist()
            ys += b.y.cpu().tolist()
    test_auc = roc_auc_score(ys, ps)
    print(f"[{name}] TEST AUC: {test_auc:.3f}")
    return best_state, test_auc, net, dev

@torch.no_grad()
def gnn_predict(net, dev, smiles_list):
    loader = DataLoader([g for g in (mol_to_graph(s, 0) for s in smiles_list) if g is not None],
                        batch_size=BATCH)
    out = []
    for b in loader:
        out += torch.sigmoid(net(b.to(dev))).cpu().tolist()
    return out

# %% Cell 7 — train all, save models + GNN plant predictions
os.makedirs(OUT_DIR, exist_ok=True)
summary, plant_preds = {}, {}
for name, df in DATA.items():
    print(f"=== Training {name} ===")
    state, test_auc, net, dev = train_one(name, df)
    torch.save(state, f"{OUT_DIR}/{name}_gcn.pt")
    summary[name] = round(float(test_auc), 3)
    plant_preds[name] = gnn_predict(net, dev, list(PLANTS.values()))
    print(f"saved {OUT_DIR}/{name}_gcn.pt")

gnn_df = pd.DataFrame({"name": list(PLANTS.keys()),
                       **{f"{n}_gnn_prob": plant_preds[n] for n in DATA}})
gnn_df.to_csv("/kaggle/working/gnn_plant_preds.csv", index=False)
print("\nTest AUCs:", summary)
print(gnn_df.round(3).to_string(index=False))
print("\nDONE. Download from /kaggle/working: gnn_models/*.pt + gnn_plant_preds.csv")
