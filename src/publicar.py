"""Exporta cifras y tablas editoriales desde los mismos objetos estimados."""
import numpy as np
from analisis import ROOT, ORDER, MINING, AGRO


def tex(text):
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def numero(value, digits=2):
    return f"{float(value):.{digits}f}"


def prob(value):
    return r"$<0.0001$" if value < .0001 else f"{value:.4f}"


def parametro_label(param):
    main = {"Intercept": "Constante", "COVID": "Choque 2020", "POST_21_23": "Periodo 2021--2023"}
    if param in main:
        return main[param]
    period, predictor = param.split(":", 1)
    labels = {"Mineria_z": "Minería estandarizada", "Macro[T.Minero]": "Macro minero"}
    predictor = labels.get(predictor, predictor.replace("Nombre_Cluster_K4[T.", "").removesuffix("]"))
    return ("2020" if period == "COVID" else "2021--2023") + r" $\times$ " + tex(predictor)


def tabla(path, header, rows, spec):
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generar_resultados(study):
    output = ROOT / "resultados"
    output.mkdir(exist_ok=True)
    m, c, diag = study.config["muestra"], study.config["clustering"], study.diagnostico
    macros = {"BaseObs": str(len(study.base)), "RatesObs": str(len(study.tasas)), "LevelsObs": str(len(study.niveles)),
        "Departments": str(len(study.shares)), "Sectors": str(study.shares.shape[1]), "Zeros": str(diag["ceros"]),
        "PreStart": str(m["pre_inicio"]), "PreEnd": str(m["pre_fin"]), "RateStart": str(m["tasas_inicio"]), "RateEnd": str(m["tasas_fin"]),
        "LevelStart": str(m["niveles_inicio"]), "LevelEnd": str(m["niveles_fin"]), "BaseYear": str(m["base"]), "ShockYear": str(m["choque"]),
        "PostStart": str(m["post_inicio"]), "PostEnd": str(m["post_fin"]), "DeltaZero": numero(c["delta_ceros"]),
        "BaseStart": str(int(study.base.anio.min())), "BaseEnd": str(int(study.base.anio.max())),
        "GapReplicas": str(c["gap_replicas"]), "StabilityReplicas": str(c["estabilidad_replicas"]), "SubsamplePct": numero(c["fraccion_submuestra"]*100, 0),
        "PcaOne": numero(study.pca.explained_variance_ratio_[0]*100), "PcaTwo": numero(study.pca.explained_variance_ratio_[1]*100),
        "PcaCumulative": numero(study.pca.explained_variance_ratio_[:2].sum()*100), "PcaThree": numero(study.pca.explained_variance_ratio_[:3].sum()*100),
        "SdRatio": numero(diag["ratio_desviaciones"], 1), "LeveneP": prob(diag["levene"]["p"]), "FlignerP": prob(diag["fligner"]["p"]),
        "MacroMiningCount": str(int(study.etiquetas.Macro.eq("Minero").sum())),
        "MacroOtherCount": str(int(study.etiquetas.Macro.eq("Diversificado").sum())),
        "GapChoice": str(diag["gap_regla_1ee_k"]), "SdEnclave": numero(study.volatilidad.loc[ORDER[-1], "std"], 1),
        "SdAgro": numero(study.volatilidad.loc[ORDER[0], "std"], 1),
        "SpecCount": str(len(study.metricas)), "ConfidencePct": numero(study.config["econometria"]["confianza"] * 100, 0)}
    ward_metrics = study.metricas.query("features == 'CLR+PCA2' and metodo == 'Ward'").set_index("k")
    validation = study.validacion.set_index("k")
    macros.update({"SilMacro": numero(ward_metrics.loc[2, "sil"], 3), "SilTypology": numero(ward_metrics.loc[4, "sil"], 3),
        "JaccardWard": numero(validation.loc[4, "jaccard_ward"] * 100, 1), "JaccardKmeans": numero(validation.loc[4, "jaccard_kmeans"] * 100, 1),
        "GapTypology": numero(validation.loc[4, "gap"], 3)})
    sizes = sorted(study.etiquetas.Cluster_Ward_K2.value_counts())
    macros.update({"WardSmallCount": str(sizes[0]), "WardLargeCount": str(sizes[-1])})
    for code, word in [("M1", "One"), ("M2", "Two"), ("M3", "Three"), ("M4", "Four"), ("M5", "Five"), ("M6", "Six")]:
        result = study.modelos[code]
        macros["R" + word] = numero(result.rsquared_within, 3)
        macros["N" + word] = str(result.nobs)
    enclave_param = "POST_21_23:Nombre_Cluster_K4[T.Minero/Enclave]"
    p5, p6 = study.modelos["M5"], study.modelos["M6"]
    low, high = p5.conf_int(level=study.config["econometria"]["confianza"]).loc[enclave_param]
    macros.update({"ScarBeta": numero(p5.params[enclave_param]), "ScarP": prob(p5.pvalues[enclave_param]), "ScarLow": numero(low), "ScarHigh": numero(high),
        "CovidEnclaveBeta": numero(p5.params["COVID:Nombre_Cluster_K4[T.Minero/Enclave]"]), "CovidEnclaveP": prob(p5.pvalues["COVID:Nombre_Cluster_K4[T.Minero/Enclave]"]),
        "ContinuousBeta": numero(p6.params["POST_21_23:Mineria_z"]), "ContinuousP": prob(p6.pvalues["POST_21_23:Mineria_z"]),
        "MacroCovidBeta": numero(study.modelos["M2"].params["COVID"]), "MacroCovidP": prob(study.modelos["M2"].pvalues["COVID"])})
    trajectory = study.niveles.groupby(["anio", "Nombre_Cluster_K4"], observed=True).VAB_idx2019.mean().unstack()
    for name, word in zip(ORDER, ["Agro", "Services", "Mining", "Enclave"]):
        macros[word + "Final"] = numero(trajectory.loc[m["niveles_fin"], name], 1)
        macros[word + "Count"] = str(int(study.etiquetas.Nombre_Cluster_K4.eq(name).sum()))
    for year, word in [(2021, "First"), (2022, "Second"), (2023, "Third")]:
        macros["Enclave" + word] = numero(trajectory.loc[year, ORDER[-1]], 1)
    maximum = study.tasas.loc[study.tasas.crecimiento_vab.idxmax()]
    macros["ExtremeGrowth"] = numero(maximum.crecimiento_vab, 1)
    (output / "cifras.tex").write_text("% Generado por src/publicar.py; no editar a mano.\n" + "\n".join("\\newcommand{\\" + name + "}{" + value + "}" for name, value in macros.items()) + "\n", encoding="utf-8")

    model_rows = []
    specs = ["Tasas, cuatro perfiles", "Tasas, dos macrogrupos", "Tasas, minería continua", "Tasas, sin Apurímac 2016", "Índice, cuatro perfiles", "Índice, minería continua"]
    for (code, result), spec in zip(study.modelos.items(), specs):
        sample = study.tasas if code in ["M1", "M2", "M3", "M4"] else study.niveles
        model_rows.append([code, spec, f"{sample.anio.min()}--{sample.anio.max()}", result.nobs, numero(result.rsquared_within, 3)])
    tabla(output / "modelos.tex", ["Modelo", "Especificación", "Periodo", "Obs.", r"$R^2_w$"], model_rows, "llrrr")
    for codes, filename in [(["M1", "M2", "M3", "M4"], "tasas.tex"), (["M5", "M6"], "niveles.tex")]:
        params = list(dict.fromkeys(p for code in codes for p in study.modelos[code].params.index))
        rows = []
        for param in params:
            row = [parametro_label(param)]
            for code in codes:
                result = study.modelos[code]
                if param in result.params:
                    row.append(r"\shortstack{" + numero(result.params[param]) + r" \\ (" + numero(result.std_errors[param]) + ")}")
                else:
                    row.append("---")
            rows.append(row)
        rows += [["Observaciones"] + [str(study.modelos[code].nobs) for code in codes], [r"$R^2$ within"] + [numero(study.modelos[code].rsquared_within, 3) for code in codes]]
        tabla(output / filename, ["Regresor"] + codes, rows, "l" + "r"*len(codes))
    profile = study.shares.join(study.etiquetas.Nombre_Cluster_K4).groupby("Nombre_Cluster_K4", observed=True).mean().reindex(ORDER)
    tabla(output / "perfiles.tex", ["Perfil", "n", "Extracción", "Manufactura", "Agro"], [[tex(n), int(study.etiquetas.Nombre_Cluster_K4.eq(n).sum()), numero(profile.loc[n, MINING], 1), numero(profile.loc[n, "Manufactura"], 1), numero(profile.loc[n, AGRO], 1)] for n in ORDER], "lrrrr")
    tabla(output / "sensibilidad.tex", ["Representación / reemplazo", "ARI frente a PCA2"], [[tex(row.escenario), numero(row.ARI_vs_principal, 3)] for row in study.sensibilidad.itertuples()], "lr")
    tabla(output / "validacion.tex", ["K", "Silueta Ward", "Gap", "EE Gap", "Jaccard Ward"], [[int(row.k), numero(ward_metrics.loc[row.k, "sil"], 3) if row.k in ward_metrics.index else "---", numero(row.gap, 3), numero(row.gap_ee, 3), numero(row.jaccard_ward, 3) if np.isfinite(row.jaccard_ward) else "---"] for row in study.validacion.itertuples()], "rrrrr")
    rows = []
    for name in ORDER:
        members = study.etiquetas.index[study.etiquetas.Nombre_Cluster_K4.eq(name)]
        rows.append([tex(name), tex(", ".join(members))])
    tabla(output / "departamentos.tex", ["Perfil", "Departamentos"], rows, r"p{.27\linewidth}p{.65\linewidth}")
    loads = study.pca.components_[:3].T
    table_rows = [[tex(name), *[numero(v, 3) for v in row]] for name, row in zip(study.clr.columns, loads)]
    tabla(output / "cargas_pca.tex", ["Actividad", "PC1", "PC2", "PC3"], table_rows, r"p{.58\linewidth}rrr")
    print("Resultados LaTeX generados desde las estimaciones.", flush=True)
