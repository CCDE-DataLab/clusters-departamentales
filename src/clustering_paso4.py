"""Paso 4: dinamica 2015-19 + perfilado + panel v2."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"

if __name__ == "__main__":
    df = pd.read_excel(PROCESSED / "pbi_departamental_consolidado.xlsx")
    col_anio = [c for c in df.columns if str(c).strip() not in ("departamento",) and pd.to_numeric(df[c], errors="coerce").between(2000, 2030).mean() > 0.9][0]
    print("anio:", repr(col_anio))
    win = df[(df[col_anio] >= 2015) & (df[col_anio] <= 2019)].copy()
    sectors = [c for c in df.columns if c not in ("departamento", col_anio, "PIB_total", "var_pib")]
    # dinamica: std + pendiente por depto (solo 3 sectores clave para no saturar)
    claves = ["Extracción de Petróleo, Gas y Minerales", "Manufactura"]
    claves = [c for c in df.columns if any(k[:8] in str(c)[:12] for k in ["Extracci", "Manufact", "Agricult"])]
    print("claves:", claves)
    rows = []
    for dep, g in win.groupby("departamento"):
        g = g.sort_values(col_anio)
        x = g[col_anio].values.astype(float)
        r = {"departamento": dep, "var_pib_mean_1519": g["var_pib"].mean(), "var_pib_std_1519": g["var_pib"].std()}
        for c in claves:
            y = pd.to_numeric(g[c], errors="coerce").values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 else 0.0
            r[f"slope_{c[:12]}"] = slope
            r[f"std_{c[:12]}"] = float(np.std(y))
        rows.append(r)
    din = pd.DataFrame(rows).set_index("departamento")
    print(din.head(3).round(3).to_string())
    # test: PCA2 solo vs PCA2+dinamica
    clr = pd.read_excel(PROCESSED / "clustering_paso1_clr.xlsx", index_col=0)
    X2 = PCA().fit(clr.values).transform(clr.values)[:, :2]
    Z = StandardScaler().fit_transform(din.reindex(clr.index).fillna(0).values)
    Xext = np.hstack([StandardScaler().fit_transform(X2), Z * 0.5])
    for name, X in [("PCA2", X2), ("PCA2+din", Xext)]:
        lab = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
        print(f"{name} k=4 sil={silhouette_score(X, lab):.3f}")
    # perfilado con Ward k=2 y k=4 sobre PCA2
    prom = win.groupby("departamento").mean(numeric_only=True)
    for k in [2, 4]:
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X2)
        eti = pd.DataFrame({"departamento": list(clr.index), f"Cluster_WardK{k}": lab}).set_index("departamento")
        perfil = prom.join(eti).groupby(f"Cluster_WardK{k}").mean(numeric_only=True)
        print(f"--- perfil K={k} ---")
        print(perfil.round(2).to_string())
        perfil.to_excel(PROCESSED / f"clustering_paso4_perfil_K{k}.xlsx")
    # etiquetas finales: jerarquia 2 macro + 4 sub
    lab2 = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X2)
    lab4 = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X2)
    eti = pd.DataFrame({"departamento": list(clr.index), "Cluster_K2": lab2, "Cluster_K4": lab4})
    # nombres K4 segun medias (minero vs agro): mapeo por Extraccion media
    tmp = prom.join(eti.set_index("departamento")).groupby("Cluster_K4").mean(numeric_only=True)
    col_min = [c for c in tmp.columns if "Extracci" in str(c)][0]
    orden = tmp[col_min].sort_values(ascending=False).index.tolist()
    nombres = {orden[0]: "Minero/Enclave", orden[1]: "Minero/Diversificado", orden[2]: "Servicios/Manufactura", orden[3]: "Diversificado/Agro"} if len(orden) == 4 else {i: f"C{i}" for i in orden}
    eti["Nombre_Cluster_K4"] = eti["Cluster_K4"].map(nombres)
    eti.to_excel(PROCESSED / "clustering_paso4_etiquetas.xlsx", index=False)
    print("nombres K4:", nombres)
    # panel v2: merge + dummies tiempo extendidas
    panel = pd.read_excel(PROCESSED / "pbi_departamental_consolidado.xlsx")
    ca = [c for c in panel.columns if pd.to_numeric(panel[c], errors="coerce").between(2000, 2030).mean() > 0.9][0]
    panel = panel.merge(eti, left_on="departamento", right_on="departamento", how="left")
    panel["COVID"] = (panel[ca] == 2020).astype(int)
    panel["REBOTE"] = (panel[ca] == 2021).astype(int)
    panel["POST"] = (panel[ca] >= 2021).astype(int)
    panel.to_excel(PROCESSED / "panel_econometrico_v2.xlsx", index=False)
    print("panel v2:", panel.shape, list(panel.columns[-7:]))
