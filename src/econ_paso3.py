"""Paso3: REBOTE + pooled + sin outlier."""
from pathlib import Path
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels import PooledOLS

BASE = Path(__file__).resolve().parent.parent
P = BASE / "data" / "processed"
if __name__ == "__main__":
    df = pd.read_excel(P / "panel_econometrico_macro.xlsx")
    ca = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").between(2000, 2030).mean() > 0.9][0]
    df["REBOTE"] = (df[ca] == 2021).astype(int)
    df["POST"] = (df[ca] >= 2021).astype(int)
    df["Macro"] = pd.Categorical(df["Macro"], categories=["Diversificado", "Minero"], ordered=True)
    # M1 base macro FE
    d = df.set_index(["departamento", ca])
    m1 = PanelOLS.from_formula("var_pib ~ 1 + COVID + COVID:Macro + EntityEffects", data=d).fit(cov_type="clustered", cluster_entity=True)
    # M2 + rebote
    m2 = PanelOLS.from_formula("var_pib ~ 1 + COVID + REBOTE + COVID:Macro + REBOTE:Macro + EntityEffects", data=d).fit(cov_type="clustered", cluster_entity=True)
    # M3 pooled sin FE
    m3 = PooledOLS.from_formula("var_pib ~ 1 + COVID + COVID:Macro", data=d).fit(cov_type="clustered", cluster_entity=True)
    # M4 sin outlier Apurimac 2016
    d4 = d.reset_index()
    d4 = d4[~((d4["departamento"].str.contains("Apur", na=False)) & (d4[ca] == 2016))].set_index(["departamento", ca])
    m4 = PanelOLS.from_formula("var_pib ~ 1 + COVID + COVID:Macro + EntityEffects", data=d4).fit(cov_type="clustered", cluster_entity=True)
    for name, m in [("M1 base", m1), ("M2 rebote", m2), ("M3 pooled", m3), ("M4 sinApu2016", m4)]:
        print(f"--- {name} R2={float(m.rsquared):.3f} ---")
        for k in m.params.index:
            print(f" {k}: b={m.params[k]:.2f} p={m.pvalues[k]:.4f}")
