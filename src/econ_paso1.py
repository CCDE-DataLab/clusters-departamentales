"""Econ paso1: reproduce modelo base + diagnostico de potencia."""
from pathlib import Path
import pandas as pd
from linearmodels.panel import PanelOLS

BASE = Path(__file__).resolve().parent.parent
P = BASE / "data" / "processed"

if __name__ == "__main__":
    df = pd.read_excel(P / "panel_econometrico_base.xlsx")
    ca = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").between(2000, 2030).mean() > 0.9][0]
    print("anio:", repr(ca), "n:", len(df))
    print("deptos:", df["departamento"].nunique())
    print("obs por cluster:\n", df["Cluster"].value_counts().to_string())
    print("deptos por cluster:\n", df.groupby("Cluster")["departamento"].nunique().to_string())
    print("COVID n:", int(df["COVID"].sum()), "de", len(df))
    print("var_pib std por cluster:\n", df.groupby("Cluster")["var_pib"].std().round(2).to_string())
    d = df.set_index(["departamento", ca]).copy()
    d["Nombre_Cluster"] = d["Cluster"].map({0: "Agro", 1: "Serv", 2: "Enclave", 3: "MinDiv"})
    d["Nombre_Cluster"] = pd.Categorical(d["Nombre_Cluster"], categories=["Agro", "Serv", "Enclave", "MinDiv"], ordered=True)
    m = PanelOLS.from_formula("var_pib ~ 1 + COVID + COVID:Nombre_Cluster + EntityEffects", data=d).fit(cov_type="robust")
    print(m.summary)
    print("\n[diag] SE interacciones grandes vs efecto:")
    for k in ["COVID", "COVID:Nombre_Cluster[T.Serv]", "COVID:Nombre_Cluster[T.Enclave]", "COVID:Nombre_Cluster[T.MinDiv]"]:
        if k in m.params.index:
            print(f" {k}: beta={m.params[k]:.2f} se={m.std_errors[k]:.2f} t={m.tstats[k]:.2f} p={m.pvalues[k]:.3f}")
    print("[diag] R2 within:", round(float(m.rsquared_within), 4), "| poolability p:", round(float(m.f_pooled.pval), 4) if hasattr(m, "f_pooled") else "NA")
