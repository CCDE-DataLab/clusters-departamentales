"""Chequeo idea Mauricio: 4 clusters -> 2 macro por semejanza."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

BASE = Path(__file__).resolve().parent.parent
P = BASE / "data" / "processed"
if __name__ == "__main__":
    clr = pd.read_excel(P / "clustering_paso1_clr.xlsx", index_col=0)
    X2 = PCA().fit(clr.values).transform(clr.values)[:, :2]
    idx = list(clr.index)
    l2 = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X2)
    l4 = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X2)
    cruce = pd.crosstab(pd.Series(l4, index=idx, name="K4"), pd.Series(l2, index=idx, name="K2"))
    print("Cruce K4 x K2 (anidamiento):\n", cruce.to_string())
    # distancia entre centroides K4 en PCA2
    cen = {k: X2[l4 == k].mean(0) for k in sorted(set(l4))}
    print("\nDist centroides K4:")
    ks = sorted(cen)
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            d = float(np.linalg.norm(cen[ks[i]]-cen[ks[j]]))
            print(f" d(K4={ks[i]},K4={ks[j]})={d:.2f}")
    # linkage: ultimas fusiones
    Z = linkage(X2, method="ward")
    print("\nUltimas 3 fusiones (dist, n):")
    print(Z[-3:].round(2))
    # potencia: obs por grupo
    df = pd.read_excel(P / "panel_econometrico_base.xlsx")
    df["K4tmp"] = df["departamento"].map(dict(zip(idx, l4)))
    df["K2tmp"] = df["departamento"].map(dict(zip(idx, l2)))
    print("\nObs K4:\n", df["K4tmp"].value_counts().to_string())
    print("Obs K2:\n", df["K2tmp"].value_counts().to_string())
