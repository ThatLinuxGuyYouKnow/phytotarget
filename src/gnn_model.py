"""Shared GCN definition (must match kaggle_notebook.py) + inference helpers.

Torch/PyG are optional imports: functions raise ImportError with a clear
message when the libs are missing, so the app can fall back to RF + CSV.
"""
import numpy as np

ATOM_FEATS = 9
HID = 128


def mol_to_graph(smiles, label=0):
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data
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
    if not ei:
        ei, eb = [[0, 0]], [[1.0, 0.0]]
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(eb, dtype=torch.float),
                y=torch.tensor([label], dtype=torch.float))


def build_gcn():
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    class GCN(nn.Module):
        def __init__(self, in_dim=ATOM_FEATS, hid=HID):
            super().__init__()
            self.c1, self.c2, self.c3 = (GCNConv(in_dim, hid), GCNConv(hid, hid),
                                         GCNConv(hid, hid))
            self.lin = nn.Linear(hid, 1)

        def forward(self, d):
            x, e = d.x, d.edge_index
            x = F.relu(self.c1(x, e))
            x = F.relu(self.c2(x, e))
            x = self.c3(x, e)
            return self.lin(global_mean_pool(x, d.batch)).squeeze(-1)

    return GCN()


def load_gcn(weight_path, device="cpu"):
    import torch
    net = build_gcn()
    state = torch.load(weight_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def gnn_prob(net, smiles, device="cpu"):
    import torch
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch_geometric.data import DataLoader
    g = mol_to_graph(smiles)
    if g is None:
        return None
    loader = DataLoader([g], batch_size=1)
    with torch.no_grad():
        for b in loader:
            return float(torch.sigmoid(net(b.to(device))).cpu().item())
    return None
