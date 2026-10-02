"""Paso2 Mauricio: 4 clusters -> 2 macro (minero vs diversificado) + panel."""
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from linearmodels.panel import PanelOLS

BASE = Path(__file__).resolve().parent.parent
P = BASE / "data" / "processed"
if __name__ == "__main__":
    clr = pd.read_excel(P / "clustering_paso1_clr.xlsx", index_col=0)
    X2 = PCA().fit(clr.values).transform(clr.values)[:, :2]
    idx = list(clr.index)
    l4 = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X2)
    # top2 extraccion -> mineros (robusto a orden de etiquetas)
    base = pd.read_excel(P / "pbi_departamental_consolidado.xlsx")
    ca = [c for c in base.columns if pd.to_numeric(base[c], errors="coerce").between(2000, 2030).mean() > 0.9][0]
    colmin = [c for c in base.columns if "Extracci" in str(c)][0]
    prom = base[(base[ca] >= 2015) & (base[ca] <= 2019)].groupby("departamento")[colmin].mean()
    med = pd.Series(l4, index=idx).groupby(level=0).mean()  # dummy
    kmean = pd.DataFrame({"k4": l4}, index=idx).join(prom).groupby("k4")[colmin].mean().sort_values(ascending=False)
    print("extraccion media por K4:\n", kmean.round(2).to_string())
    mineros = set(kmean.index[:2].tolist())
    print("K4 mineros:", mineros)
    macro = pd.DataFrame({"departamento": idx, "K4": l4})
    macro["Macro"] = macro["K4"].apply(lambda k: "Minero" if k in mineros else "Diversificado")
    macro.to_excel(P / "econ_paso2_macro.xlsx", index=False)
    df = base.merge(macro, on="departamento", how="left")
    print("obs macro:\n", df["Macro"].value_counts().to_string())
    df["COVID"] = (df[ca] == 2020).astype(int)
    d = df.set_index(["departamento", ca]).copy()
    d["Macro"] = pd.Categorical(d["Macro"], categories=["Diversificado", "Minero"], ordered=True)
    for cov, tag in [("robust", "robust"), ("clustered", "clustered")]:
        kw = {"cov_type": cov} if cov == "robust" else {"cov_type": cov, "cluster_entity": True}
        m = PanelOLS.from_formula("var_pib ~ 1 + COVID + COVID:Macro + EntityEffects", data=d).fit(**kw)
        print(f"--- Macro {tag} ---")
        print(m.params.round(3).to_string(), "\n", m.pvalues.round(4).to_string(), "\nR2w:", round(float(m.rsquared_within), 4))
    df.to_excel(P / "panel_econometrico_macro.xlsx", index=False)
    print("Guardado macro + panel macro.")
