# %% [markdown]
# PhytoTarget ML — GNN training on Modal GPU (T4/A200)
# Run this in a Modal web notebook with GPU enabled (T4 is enough).
# Self-contained: installs deps, fetches ChEMBL data, trains a GCN per target, saves models.

# %% Cell 1 — install deps (run once, then restart kernel if prompted)
import sys
!pip install -q rdkit torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
!pip install -q torch_geometric scikit-learn pandas pyyaml 2>&1 | tail -2

# %% Cell 2 — imports + GPU check
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

# %% Cell 3 — config
TARGETS = {
    "ACHE":  "CHEMBL220",   # Alzheimer's
    "BACE1": "CHEMBL4822",  # Alzheimer's
    "GSK3B": "CHEMBL262",   # Alzheimer's
    "PTGS2": "CHEMBL230",   # Inflammation (COX-2)
    "TNF":   "CHEMBL1825",  # Inflammation
}
MAX_RECORDS = 3000
ACTIVE_NM = 10000.0
N_BITS_FALLBACK = True
EPOCHS = 60
BATCH = 64
LR = 1e-3
OUT_DIR = "gnn_models"

# %% Cell 4 — fetch ChEMBL data
import time, requests, pandas as pd

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

DATA = {name: fetch_target(cid, MAX_RECORDS) for name, cid in TARGETS.items()}

# %% Cell 5 — molecules -> graphs
from rdkit import Chem
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader
except ImportError:  # older PyG
    from torch_geometric.data import DataLoader
import numpy as np

ATOM_FEATS = 9  # atomic_num(1; scaled), degree, formal_charge, num_H, hybridization(1-hot-ish idx), aromatic, in_ring, mass_scaled, valence

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
    mk = lambda ii: DataLoader([graphs[i] for i in ii], batch_size=BATCH, shuffle=True)
    return mk(tr), DataLoader([graphs[i] for i in va], batch_size=BATCH), DataLoader([graphs[i] for i in te], batch_size=BATCH)

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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_loaders(df)
    # class balance via pos_weight
    pos = df["label"].mean()
    pw = torch.tensor([(1-pos)/max(pos, 1e-6)], device=dev)
    net = GCN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, patience = -1, None, 0
    for ep in range(1, EPOCHS+1):
        net.train()
        for b in tr:
            b = b.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(b), b.y)
            loss.backward(); opt.step()
        # validate
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
            print(f"  [{name}] ep {ep}: val AUC {auc:.3f} (best {best:.3f})")
        if patience >= 12:
            print(f"  [{name}] early stop at ep {ep}")
            break
    # test
    net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    net.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in te:
            b = b.to(dev)
            ps += torch.sigmoid(net(b)).cpu().tolist()
            ys += b.y.cpu().tolist()
    print(f"[{name}] TEST AUC: {roc_auc_score(ys, ps):.3f}")
    return best_state

# %% Cell 7 — train all + save
import os
os.makedirs(OUT_DIR, exist_ok=True)
for name, df in DATA.items():
    print(f"=== Training {name} ===")
    state = train_one(name, df)
    torch.save(state, f"{OUT_DIR}/{name}_gcn.pt")
    print(f"saved {OUT_DIR}/{name}_gcn.pt")
print("DONE. Download the gnn_models/ folder and place it in the repo root.")
