"""Render README gallery: molecule images + RF-vs-GNN chart."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

plants = pd.read_csv("data/plant_compounds.csv")
all_preds = pd.read_csv("data/plant_preds_all.csv").set_index("name")
gnn = pd.read_csv("data/gnn_plant_preds.csv").set_index("name")

# 1. molecule grid
mols = [Chem.MolFromSmiles(s) for s in plants["smiles"]]
legends = plants["name"].tolist()
img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(320, 320),
                           legends=legends, useSVG=False)
img.save("assets/molecules.png")
print("wrote assets/molecules.png")

# 2. RF vs GNN grouped bars per target (example compounds)
targets = ["ache", "bace1", "gsk3b", "ptgs2", "tnf"]
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
for ax, t in zip(axes, targets):
    rf = all_preds[f"{t}_active_prob"]
    gn = gnn[f"{t}_gnn_prob"]
    x = range(len(rf))
    ax.bar([i - 0.2 for i in x], rf.values, width=0.4, label="RF")
    ax.bar([i + 0.2 for i in x], gn.values, width=0.4, label="GNN")
    ax.set_title(t.upper())
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_preds.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("P(active)")
axes[0].legend()
fig.suptitle("RF vs GNN binding probabilities (9 plant compounds)")
fig.tight_layout()
fig.savefig("assets/rf_vs_gnn.png", dpi=100)
print("wrote assets/rf_vs_gnn.png")
