"""Paso 3: gap statistic + estabilidad bootstrap (Jaccard)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"

def wcss(X, lab):
    s = 0.0
    for k in np.unique(lab):
        Xk = X[lab == k]
        c = Xk.mean(axis=0)
        s += ((Xk - c) ** 2).sum()
    return s

def gap_stat(X, kmax=5, B=30, seed=0):
    rng = np.random.default_rng(seed)
    mins, maxs = X.min(0), X.max(0)
    gaps, ses, lws = [], [], []
    for k in range(1, kmax + 1):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X) if k > 1 else np.zeros(len(X), int)
        wk = np.log(wcss(X, lab))
        wb = []
        for _ in range(B):
            Xu = rng.uniform(mins, maxs, size=X.shape)
            lb = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xu) if k > 1 else np.zeros(len(Xu), int)
            wb.append(np.log(wcss(Xu, lb)))
        gaps.append(float(np.mean(wb) - wk))
        ses.append(float(np.std(wb) * np.sqrt(1 + 1 / B)))
        lws.append(float(wk))
    return gaps, ses, lws

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0

def estabilidad(X, metodo="Ward", B=100, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    out = {}
    for k in [2, 3, 4, 5]:
        if metodo == "Ward":
            base = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        else:
            base = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        scores = []
        for _ in range(B):
            idx = rng.choice(n, int(n * 0.8), replace=False)
            if metodo == "Ward":
                lb = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
            else:
                lb = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X[idx])
            # mejor match por cluster base
            js = []
            for c in np.unique(base):
                A = set(np.where(base == c)[0])
                A_sub = A & set(idx)
                if not A_sub:
                    continue
                best = 0.0
                for d in np.unique(lb):
                    Bd = set(idx[lb == d])
                    best = max(best, jaccard(A_sub, Bd))
                js.append(best)
            scores.append(float(np.mean(js)) if js else 0.0)
        out[k] = float(np.mean(scores))
    return out

if __name__ == "__main__":
    clr = pd.read_excel(PROCESSED / "clustering_paso1_clr.xlsx", index_col=0)
    X2 = PCA().fit(clr.values).transform(clr.values)[:, :2]
    X3 = PCA().fit(clr.values).transform(clr.values)[:, :3]
    for name, X in [("PCA2", X2), ("PCA3", X3)]:
        g, se, lw = gap_stat(X, kmax=5, B=30)
        print(f"--- gap {name} ---")
        for k in range(1, 6):
            print(f" k={k} gap={g[k-1]:.3f} se={se[k-1]:.3f}")
    for m in ["Ward", "KMeans"]:
        est = estabilidad(X2, metodo=m, B=100)
        print(f"--- estabilidad {m} PCA2 ---", {k: round(v, 3) for k, v in est.items()})
    pd.DataFrame({"k": [1, 2, 3, 4, 5], "gap_PCA2": gap_stat(X2, kmax=5, B=30)[0]}).to_excel(PROCESSED / "clustering_paso3_gap.xlsx", index=False)
    print("Guardado gap.")
