"""Fuente única de muestras, clasificaciones y estimaciones del estudio CCDE."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import tomllib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.mixture import GaussianMixture
from linearmodels.panel import PanelOLS
from composicional import reemplazar_ceros, clr, ilr, ward, pam, gap_stat, estabilidad

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["Diversificado/Agro", "Servicios/Manufactura", "Minero/Diversificado", "Minero/Enclave"]
MINING = "Extracción de Petróleo, Gas y Minerales"
AGRO = "Agricultura, Ganadería, Caza y Silvicultura"


@dataclass
class Estudio:
    config: dict
    base: pd.DataFrame
    shares: pd.DataFrame
    clr: pd.DataFrame
    ilr: pd.DataFrame
    pca: PCA
    coords: pd.DataFrame
    etiquetas: pd.DataFrame
    metricas: pd.DataFrame
    validacion: pd.DataFrame
    sensibilidad: pd.DataFrame
    panel: pd.DataFrame
    tasas: pd.DataFrame
    niveles: pd.DataFrame
    modelos: dict
    resumen: pd.DataFrame
    volatilidad: pd.DataFrame
    diagnostico: dict


def cargar_config(path=None):
    with open(path or ROOT / "config/analisis.toml", "rb") as f:
        config = tomllib.load(f)
    c, m = config["clustering"], config["muestra"]
    if c["k_tipologia"] != 4 or c["k_macro"] != 2:
        raise ValueError("La taxonomía editorial requiere cuatro perfiles y dos macrogrupos")
    if c["componentes"] != 2 or not {2, 4}.issubset(c["k_evaluados"]):
        raise ValueError("La tipología principal requiere PCA2 y evaluar al menos K=2 y K=4")
    if not m["pre_fin"] < m["choque"] or not m["base"] < m["choque"] < m["post_inicio"]:
        raise ValueError("La ventana estructural y la base deben preceder al choque")
    return config


def ejecutar(config):
    m, c, e = config["muestra"], config["clustering"], config["econometria"]
    base = pd.read_excel(ROOT / config["datos"]["entrada"]).rename(columns={"año": "anio", "PIB_total": "VAB_total", "var_pib": "crecimiento_vab"})
    if base.empty or base[["departamento", "anio"]].isna().any().any():
        raise ValueError("La base debe contener departamentos y años completos")
    if base.duplicated(["departamento", "anio"]).any():
        raise ValueError("Hay duplicados departamento-año")
    if "Otros Servicios" in base:
        base["Otros Actividades/1"] = base["Otros Actividades/1"].fillna(0) + base.pop("Otros Servicios").fillna(0)
    sectors = base.columns.difference(["departamento", "anio", "VAB_total", "crecimiento_vab"], sort=False).tolist()
    if not np.isfinite(base[sectors].to_numpy()).all() or base[sectors].lt(0).any().any() or not np.allclose(base[sectors].sum(axis=1), 100, atol=1e-6):
        raise ValueError("Las participaciones sectoriales deben ser finitas, no negativas y sumar 100")
    if not np.isfinite(base.VAB_total).all() or not base.VAB_total.gt(0).all():
        raise ValueError("El VAB debe ser finito y positivo")
    pre = base[base.anio.between(m["pre_inicio"], m["pre_fin"])].copy()
    departments = pd.Index(base.departamento.unique(), name="departamento")
    coverage = pre.groupby("departamento").size().reindex(departments, fill_value=0)
    if not coverage.eq(m["pre_fin"] - m["pre_inicio"] + 1).all():
        raise ValueError("Ventana pre-COVID incompleta")
    shares = pre.groupby("departamento")[sectors].mean()
    rep = reemplazar_ceros(shares, c["delta_ceros"])
    xclr, xilr = clr(rep), ilr(rep)
    pca = PCA(random_state=c["semilla"]).fit(xclr)
    x = pca.transform(xclr)[:, :c["componentes"]]
    coords = pd.DataFrame(x, index=shares.index, columns=[f"PC{i+1}" for i in range(x.shape[1])])
    l4, l2 = ward(x, c["k_tipologia"]), ward(x, c["k_macro"])
    labels = pd.DataFrame({"Cluster_K4": l4, "Cluster_K2": l2}, index=shares.index)
    profile_mining = shares.join(labels).groupby("Cluster_K4")[MINING].mean().sort_values()
    labels["Nombre_Cluster_K4"] = labels.Cluster_K4.map(dict(zip(profile_mining.index, ORDER)))
    # Agregación económica de los dos perfiles más extractivos; difiere del corte Ward K=2.
    extractive_clusters = set(profile_mining.index[-2:])
    labels["Macro"] = np.where(labels.Cluster_K4.isin(extractive_clusters), "Minero", "Diversificado")
    labels = labels.rename(columns={"Cluster_K2": "Cluster_Ward_K2"})
    if not labels.groupby("Cluster_K4").Cluster_Ward_K2.nunique().eq(1).all():
        raise ValueError("Las particiones Ward no están anidadas")
    sets = {"CLR+PCA2": pca.transform(xclr)[:, :2], "CLR+PCA3": pca.transform(xclr)[:, :3], "ILR-full": xilr.to_numpy()}
    rows, sensitivity = [], []
    for feature, values in sets.items():
        for k in c["k_evaluados"]:
            gmm = GaussianMixture(n_components=k, n_init=5, random_state=c["semilla"]).fit(values)
            algorithms = {"Ward": ward(values, k), "KMeans": KMeans(n_clusters=k, n_init=20, random_state=c["semilla"]).fit_predict(values), "PAM": pam(values, k, c["semilla"]), "GMM": gmm.predict(values)}
            for method, groups in algorithms.items():
                rows.append({"features": feature, "k": k, "metodo": method, "sil": silhouette_score(values, groups), "CH": calinski_harabasz_score(values, groups), "DB": davies_bouldin_score(values, groups), "BIC": gmm.bic(values) if method == "GMM" else np.nan, "ARI_vs_Ward": adjusted_rand_score(algorithms["Ward"], groups)})
        sensitivity.append({"escenario": feature, "ARI_vs_principal": adjusted_rand_score(l4, ward(values, c["k_tipologia"]))})
    for delta in c["sensibilidad_delta"]:
        z = clr(reemplazar_ceros(shares, delta))
        xd = PCA(n_components=c["componentes"]).fit_transform(z)
        sensitivity.append({"escenario": f"delta={delta:g}", "ARI_vs_principal": adjusted_rand_score(l4, ward(xd, c["k_tipologia"]))})
    ks = c["k_evaluados"]
    gaps, ses = gap_stat(x, max(ks), c["gap_replicas"], c["semilla"])
    jaccard = estabilidad(x, ks, c["estabilidad_replicas"], c["fraccion_submuestra"], c["semilla"])
    validation = pd.DataFrame({"k": range(1, len(gaps) + 1), "gap": gaps, "gap_ee": ses})
    validation["jaccard_ward"] = validation.k.map(jaccard)
    validation["jaccard_kmeans"] = validation.k.map(estabilidad(x, ks, c["estabilidad_replicas"], c["fraccion_submuestra"], c["semilla"], "KMeans"))
    gap_choices = np.flatnonzero(gaps[:-1] >= gaps[1:] - ses[1:]) + 1
    gap_k = int(gap_choices[0]) if len(gap_choices) else int(validation.loc[validation.gap.idxmax(), "k"])

    info = labels.join(coords)
    info["Share_Mineria_15_19"] = shares[MINING]
    info["HHI_15_19"] = ((shares / 100) ** 2).sum(axis=1)
    info["Mineria_z"] = (shares[MINING] - shares[MINING].mean()) / shares[MINING].std(ddof=1)
    panel = base.merge(info.reset_index(), on="departamento", how="left", validate="many_to_one")
    base_pib = panel[panel.anio.eq(m["base"])].set_index("departamento").VAB_total.reindex(departments)
    if base_pib.isna().any() or not base_pib.gt(0).all():
        raise ValueError("Falta un VAB base válido")
    panel["VAB_idx2019"] = panel.VAB_total / panel.departamento.map(base_pib) * 100
    panel["COVID"] = panel.anio.eq(m["choque"]).astype(int)
    panel["POST_21_23"] = panel.anio.between(m["post_inicio"], m["post_fin"]).astype(int)
    panel["Nombre_Cluster_K4"] = pd.Categorical(panel.Nombre_Cluster_K4, categories=ORDER, ordered=True)
    panel["Macro"] = pd.Categorical(panel.Macro, categories=["Diversificado", "Minero"], ordered=True)
    rates = panel[panel.anio.between(m["tasas_inicio"], m["tasas_fin"]) & panel.crecimiento_vab.notna()].copy()
    levels = panel[panel.anio.between(m["niveles_inicio"], m["niveles_fin"])].copy()
    for sample, start, end in [(rates, m["tasas_inicio"], m["tasas_fin"]), (levels, m["niveles_inicio"], m["niveles_fin"])]:
        if sample.VAB_total.isna().any():
            raise ValueError("VAB ausente")
        if len(sample) != len(info) * (end - start + 1):
            raise ValueError("La muestra de estimación debe estar balanceada")
    excluded = rates.departamento.eq(e["exclusion_departamento"]) & rates.anio.eq(e["exclusion_anio"])
    if excluded.sum() != 1:
        raise ValueError("M4 requiere exactamente una observación para la exclusión configurada")
    no_outlier = rates[~excluded]
    def fit(sample, y, interaction):
        formula = f"{y} ~ 1 + COVID + POST_21_23 + COVID:{interaction} + POST_21_23:{interaction} + EntityEffects"
        return PanelOLS.from_formula(formula, data=sample.set_index(["departamento", "anio"])).fit(cov_type="clustered", cluster_entity=True, debiased=True)
    models = {"M1": fit(rates, "crecimiento_vab", "Nombre_Cluster_K4"), "M2": fit(rates, "crecimiento_vab", "Macro"), "M3": fit(rates, "crecimiento_vab", "Mineria_z"), "M4": fit(no_outlier, "crecimiento_vab", "Nombre_Cluster_K4"), "M5": fit(levels, "VAB_idx2019", "Nombre_Cluster_K4"), "M6": fit(levels, "VAB_idx2019", "Mineria_z")}
    summary = []
    for code, result in models.items():
        ci = result.conf_int(level=e["confianza"])
        for param in result.params.index:
            summary.append({"Modelo": code, "Dependiente": "crecimiento_vab" if code in ["M1", "M2", "M3", "M4"] else "VAB_idx2019", "Parametro": param, "Coeficiente": result.params[param], "Error_Estandar": result.std_errors[param], "T_stat": result.tstats[param], "P_valor": result.pvalues[param], "IC_inferior": ci.loc[param].iloc[0], "IC_superior": ci.loc[param].iloc[1], "R2_within": result.rsquared_within, "N_obs": result.nobs, "Poolability_F": result.f_pooled.stat, "Poolability_p": result.f_pooled.pval})
    vol = rates.groupby("Nombre_Cluster_K4", observed=True).crecimiento_vab.agg(["std", "count"]).reindex(ORDER)
    vol["std_sin_exclusion"] = no_outlier.groupby("Nombre_Cluster_K4", observed=True).crecimiento_vab.std()
    groups = [rates.loc[rates.Nombre_Cluster_K4.eq(name), "crecimiento_vab"].to_numpy() for name in ORDER]
    levene, fligner = stats.levene(*groups, center="median"), stats.fligner(*groups)
    diag = {"gap_regla_1ee_k": gap_k, "levene": {"stat": float(levene.statistic), "p": float(levene.pvalue)}, "fligner": {"stat": float(fligner.statistic), "p": float(fligner.pvalue)}, "ratio_desviaciones": float(vol.loc[ORDER[-1], "std"] / vol.loc[ORDER[0], "std"]), "ceros": int((shares == 0).sum().sum()), "n_sectores": len(sectors), "n_departamentos": len(info), "n_base": len(base), "n_tasas": len(rates), "n_niveles": len(levels)}
    return Estudio(config, base, shares, xclr, xilr, pca, coords, info, pd.DataFrame(rows), validation, pd.DataFrame(sensitivity), panel, rates, levels, models, pd.DataFrame(summary), vol, diag)


def guardar(study):
    output = ROOT / study.config["datos"]["salida"]
    output.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for name, frame, index in [("composiciones_pre", study.shares, True), ("coordenadas_clr", study.clr, True), ("coordenadas_ilr", study.ilr, True), ("etiquetas_clusters", study.etiquetas, True), ("metricas_clustering", study.metricas, False), ("validacion_clustering", study.validacion, False), ("sensibilidad_clustering", study.sensibilidad, False), ("panel_econometrico_final", study.panel, False), ("resumen_modelos_econometricos", study.resumen, False), ("volatilidad", study.volatilidad, True)]:
        path = output / f"{name}.xlsx"
        frame.to_excel(path, index=index)
        fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = study.base.groupby("anio").departamento.nunique()
    manifest = {"config": study.config, "diagnostico": study.diagnostico, "cobertura": {str(k): int(v) for k, v in coverage.items()}, "sha256_entrada": hashlib.sha256((ROOT / study.config["datos"]["entrada"]).read_bytes()).hexdigest(), "sha256_fuentes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "data/raw").glob("*.xlsx"))}, "versiones": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "linearmodels", "matplotlib", "adjustText"]}}
    manifest["sha256_salidas"] = fingerprints
    (output / "resultados.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
