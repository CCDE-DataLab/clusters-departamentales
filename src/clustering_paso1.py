"""Paso 1: reemplazo multiplicativo de ceros + CLR/ILR + PCA (2 vs 3 PCs)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"

def find_col(df, candidates):
    import unicodedata
    for c in df.columns:
        norm = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode().lower().strip()
        if norm in candidates or any(k == norm for k in candidates):
            return c
    # fallback: columna numerica con valores 2000-2030
    for c in df.columns:
        try:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.between(2000, 2030).mean() > 0.9:
                return c
        except Exception:
            continue
    raise KeyError(f"No se encontro columna {candidates} en {list(df.columns)}")

def multiplicative_replacement(X, delta=0.05, total=100.0):
    """Reemplazo multiplicativo: ceros -> delta, no-ceros se escalan."""
    X = X.copy().astype(float)
    n_zeros = int((X.values == 0).sum())
    if n_zeros == 0:
        print(f"[ceros] 0 ceros, sin reemplazo.")
        return X
    for idx in X.index:
        row = X.loc[idx].values
        z = int((row == 0).sum())
        if z == 0:
            continue
        row_new = row.copy()
        row_new[row == 0] = delta
        mask = row > 0
        factor = (total - z * delta) / total
        row_new[mask] = row[mask] * factor
        X.loc[idx] = row_new
    print(f"[ceros] {n_zeros} ceros reemplazados con delta={delta}.")
    return X

def clr(X):
    gm = np.exp(np.log(X.values).mean(axis=1))
    return pd.DataFrame(np.log(X.values / gm[:, None]), index=X.index, columns=X.columns)

def ilr(X):
    """ILR con base Helmert secuencial."""
    D = X.shape[1]
    logx = np.log(X.values)
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        v = np.zeros(D)
        v[: j + 1] = 1.0 / (j + 1)
        v[j + 1] = -1.0
        v = v / np.linalg.norm(v)
        V[:, j] = v
    return pd.DataFrame(logx @ V, index=X.index)

if __name__ == "__main__":
    df = pd.read_excel(PROCESSED / "pbi_departamental_consolidado.xlsx")
    col_anio = find_col(df, ["ano", "anio", "year"])
    print("col_anio detectada:", repr(col_anio))
    pre = df[(df[col_anio] >= 2015) & (df[col_anio] <= 2019)].copy()
    drop = [c for c in [col_anio, "PIB_total", "var_pib"] if c in pre.columns]
    pre = pre.drop(columns=drop)
    prom = pre.groupby("departamento").mean(numeric_only=True)
    # fusionar Otros como en notebook original
    if "Otros Servicios" in prom.columns:
        prom["Otros Actividades/1"] = prom["Otros Servicios"].fillna(0) + prom["Otros Actividades/1"].fillna(0)
        prom = prom.drop(columns=["Otros Servicios"])
    prom = prom.fillna(0)
    print("promedio:", prom.shape)
    print("ceros:", int((prom.values == 0).sum()))
    X_rep = multiplicative_replacement(prom, delta=0.05, total=100.0)
    df_clr = clr(X_rep)
    df_ilr = ilr(X_rep)
    print("CLR ok, ILR ok:", df_clr.shape, df_ilr.shape)
    for name, M in [("CLR", df_clr), ("ILR", df_ilr)]:
        p = PCA().fit(M.values)
        print(f"--- {name} varianza ---")
        for i, v in enumerate(p.explained_variance_ratio_[:5]):
            print(f"  PC{i+1}: {v:.4f} acum:{p.explained_variance_ratio_[:i+1].sum():.4f}")
    df_clr.to_excel(PROCESSED / "clustering_paso1_clr.xlsx")
    df_ilr.to_excel(PROCESSED / "clustering_paso1_ilr.xlsx")
    print("Guardado clr/ilr en data/processed/")
