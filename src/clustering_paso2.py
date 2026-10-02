"""Paso 2: Ward vs KMeans vs PAM vs GMM sobre CLR+PCA2, PCA3 e ILR."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"

def pam(X, k, seed=0, max_iter=50):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) ** 0.5
    med = rng.choice(n, k, replace=False)
    for _ in range(max_iter):
        # asignar
        lab = np.argmin(D[:, med], axis=1)
        cost = D[np.arange(n), med[lab]].sum()
        best = None
        for i in range(k):
            for o in range(n):
                if o in med:
                    continue
                cand = med.copy()
                cand[i] = o
                l2 = np.argmin(D[:, cand], axis=1)
                c2 = D[np.arange(n), cand[l2]].sum()
                if c2 < cost and (best is None or c2 < best[0]):
                    best = (c2, cand)
        if best is None:
            break
        cost, med = best[1], best[1]
        # med ya actualizado
    lab = np.argmin(D[:, med], axis=1)
    return lab

if __name__ == "__main__":
    clr = pd.read_excel(PROCESSED / "clustering_paso1_clr.xlsx", index_col=0)
    ilr = pd.read_excel(PROCESSED / "clustering_paso1_ilr.xlsx", index_col=0)
    pca = PCA().fit(clr.values)
    sets = {
        "CLR+PCA2": pca.transform(clr.values)[:, :2],
        "CLR+PCA3": pca.transform(clr.values)[:, :3],
        "ILR-full": ilr.values,
    }
    rows = []
    for sname, X in sets.items():
        for k in [2, 3, 4, 5]:
            # Ward
            lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            rows.append([sname, k, "Ward",
                         silhouette_score(X, lab),
                         calinski_harabasz_score(X, lab),
                         davies_bouldin_score(X, lab), None])
            # KMeans
            lab = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(X)
            rows.append([sname, k, "KMeans",
                         silhouette_score(X, lab),
                         calinski_harabasz_score(X, lab),
                         davies_bouldin_score(X, lab), None])
            # PAM
            lab = pam(X, k)
            rows.append([sname, k, "PAM",
                         silhouette_score(X, lab),
                         calinski_harabasz_score(X, lab),
                         davies_bouldin_score(X, lab), None])
            # GMM
            g = GaussianMixture(n_components=k, n_init=5, random_state=0).fit(X)
            lab = g.predict(X)
            try:
                sil = silhouette_score(X, lab)
            except Exception:
                sil = float("nan")
            rows.append([sname, k, "GMM",
                         sil, float("nan"), float("nan"), g.bic(X)])
    out = pd.DataFrame(rows, columns=["features", "k", "metodo", "sil", "CH", "DB", "BIC"])
    out.to_excel(PROCESSED / "clustering_paso2_metricas.xlsx", index=False)
    print(out.round(3).to_string())
