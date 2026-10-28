"""Figuras compartidas por artículo, presentación y cuadernos."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram
from adjustText import adjust_text
from analisis import ROOT, ORDER, MINING, AGRO

NAVY = "#0A1E40"
COLORS = dict(zip(ORDER, ["#16805F", "#0077B6", "#784EA0", "#CA5D36"]))
SHORT = dict(zip(ORDER, ["Diversificado / agro", "Servicios / manufactura", "Minero diversificado", "Minero enclave"]))


def preparar():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.titlesize": 18,
        "axes.labelsize": 16, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1", "text.color": NAVY, "axes.labelcolor": NAVY,
        "xtick.color": NAVY, "ytick.color": NAVY, "axes.axisbelow": True,
        "grid.color": "#E2E8F0", "grid.alpha": .8, "figure.facecolor": "white"})


def guardar(fig, name, study):
    folder = ROOT / study.config["datos"]["figuras"]
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(folder / f"{name}.{suffix}", dpi=study.config["visual"]["dpi"], bbox_inches="tight", pad_inches=.12,
                    metadata={"Creator": "CCDE"} if suffix == "pdf" else None)
    plt.close(fig)


def pca(study):
    fig, ax = plt.subplots(figsize=(13, 6.8), layout="constrained")
    texts = []
    for name in ORDER:
        rows = study.etiquetas[study.etiquetas.Nombre_Cluster_K4.eq(name)]
        ax.scatter(rows.PC1, rows.PC2, color=COLORS[name], s=90,
                   marker="D" if "Minero" in name else "o", edgecolor="white", zorder=4,
                   label=f"{SHORT[name]} (n={len(rows)})")
        for dep, row in rows.iterrows():
            texts.append(ax.text(row.PC1, row.PC2, dep, fontsize=16, color=COLORS[name], weight="medium"))
    ax.margins(x=.16, y=.18)
    ax.axhline(0, color="#94A3B8", lw=.7, ls=":")
    ax.axvline(0, color="#94A3B8", lw=.7, ls=":")
    explained = study.pca.explained_variance_ratio_ * 100
    ax.set(xlabel=f"PC1 · {explained[0]:.1f}% de la varianza CLR", ylabel=f"PC2 · {explained[1]:.1f}% de la varianza CLR")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=2, frameon=False, fontsize=14)
    ax.grid(alpha=.5)
    fig.canvas.draw()
    np.random.seed(study.config["clustering"]["semilla"])
    adjust_text(texts, x=study.coords.PC1.to_numpy(), y=study.coords.PC2.to_numpy(),
                ax=ax, expand=(1.12, 1.35), force_text=(.6, .9), force_static=(.4, .6),
                force_pull=(.004, .004), iter_lim=700, ensure_inside_axes=True,
                arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": .7})
    guardar(fig, "fig1_pca_clusters", study)


def validacion(study):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), layout="constrained")
    ax, gap_ax, stab_ax = axes
    styles = [("Ward", NAVY, "o", "-"), ("KMeans", "#0077B6", "s", "--"), ("PAM", "#16805F", "^", "-."), ("GMM", "#CA5D36", "D", ":")]
    metrics = study.metricas[study.metricas.features.eq("CLR+PCA2")]
    for method, color, marker, ls in styles:
        data = metrics[metrics.metodo.eq(method)].sort_values("k")
        ax.plot(data.k, data.sil, label=method, color=color, ls=ls, marker=marker,
                markerfacecolor="none" if method == "KMeans" else color, markersize=8, lw=2)
    ax.set(title="Separación entre grupos", ylabel="Silueta · mayor es mejor", ylim=(0, max(.55, metrics.sil.max() * 1.1)))
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=14)
    validation = study.validacion
    gap_ax.errorbar(validation.k, validation.gap, yerr=validation.gap_ee, color=NAVY, marker="o", capsize=4, lw=2)
    gap_ax.axhline(0, color="#94A3B8", lw=.8, ls=":")
    gap_ax.set(title="Gap y error de Monte Carlo", ylabel="Gap ± 1 error estándar")
    for col, label, color in [("jaccard_ward", "Ward", NAVY), ("jaccard_kmeans", "K-Means", "#0077B6")]:
        stab_ax.plot(validation.k, validation[col], marker="o", color=color, label=label, lw=2)
    stab_ax.set(title="Estabilidad por submuestreo", ylabel="Jaccard medio · escala 0–1", ylim=(0, 1.04))
    stab_ax.legend(loc="lower left", frameon=False)
    for axis in axes:
        axis.set_xlabel("Número de grupos (K)")
        axis.set_xticks(validation.k if axis is gap_ax else study.config["clustering"]["k_evaluados"])
        axis.grid(axis="y")
        axis.axvspan(study.config["clustering"]["k_tipologia"] - .15, study.config["clustering"]["k_tipologia"] + .15, color="#0077B6", alpha=.07)
    guardar(fig, "fig2_validacion_clustering", study)


def perfiles(study):
    profile = study.shares.join(study.etiquetas.Nombre_Cluster_K4).groupby("Nombre_Cluster_K4", observed=True).mean().reindex(ORDER[::-1])
    selected = [MINING, "Manufactura", AGRO]
    values = profile[selected].copy()
    values["resto"] = profile.drop(columns=selected).sum(axis=1)
    fig, ax = plt.subplots(figsize=(13, 5.4), layout="constrained")
    left = np.zeros(len(profile))
    segment_labels = []
    for col, label, color in [(MINING, "Extracción minera e hidrocarburos", "#CA5D36"), ("Manufactura", "Manufactura", NAVY), (AGRO, "Agropecuario", "#16805F"), ("resto", "Otras actividades", "#B4C4D3")]:
        v = values[col].to_numpy()
        ax.barh(range(len(v)), v, left=left, color=color, height=.58, edgecolor="white", label=label)
        for i, value in enumerate(v):
            if value >= 5:
                label = ax.text(left[i] + value / 2, i, f"{value:.1f}%", ha="center", va="center", color=NAVY if col == "resto" else "white", weight="bold", fontsize=17)
                segment_labels.append((label, value))
        left += v
    ax.set(yticks=range(len(profile)), yticklabels=[f"{SHORT[n]}  (n={int(study.etiquetas.Nombre_Cluster_K4.eq(n).sum())})" for n in profile.index], xlim=(0, 100), xlabel="Participación media en el VAB departamental (%)")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False, fontsize=14)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, width in segment_labels:
        available = ax.transData.transform((width, 0))[0] - ax.transData.transform((0, 0))[0]
        if label.get_window_extent(renderer).width > available * .85:
            label.remove()
    guardar(fig, "fig3_perfiles_sectoriales", study)


def jerarquia(study):
    linkage_matrix = linkage(study.coords.to_numpy(), method="ward")
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    def cut_height(k):
        return float((linkage_matrix[-k, 2] + linkage_matrix[-k+1, 2]) / 2)
    cut4, cut2 = cut_height(4), cut_height(2)
    dendrogram(linkage_matrix, labels=study.coords.index.tolist(), orientation="right", color_threshold=0, above_threshold_color="#94A3B8", leaf_font_size=16, ax=ax)
    for label in ax.get_yticklabels():
        label.set_color(COLORS[study.etiquetas.loc[label.get_text(), "Nombre_Cluster_K4"]])
    ax.axvline(cut4, color="#0077B6", ls="--", lw=1.5, label="Corte K=4")
    ax.axvline(cut2, color=NAVY, ls=":", lw=1.5, label="Corte K=2")
    ax.set_xlabel("Altura de fusión Ward · espacio CLR + PCA2")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x")
    guardar(fig, "fig3b_jerarquia_ward", study)


def volatilidad(study):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    ax, std_ax = axes
    rng = np.random.default_rng(study.config["clustering"]["semilla"])
    for i, name in enumerate(ORDER):
        rows = study.tasas[study.tasas.Nombre_Cluster_K4.eq(name)]
        ax.scatter(rows.crecimiento_vab, i + rng.uniform(-.16, .16, len(rows)), s=22, color=COLORS[name], alpha=.6, edgecolors="white", linewidth=.3)
        ax.boxplot([rows.crecimiento_vab], positions=[i], vert=False, showfliers=False, widths=.45, manage_ticks=False,
                   medianprops={"color": NAVY, "lw": 2}, boxprops={"color": NAVY}, whiskerprops={"color": NAVY}, capprops={"color": NAVY})
    ax.axvline(0, color="#94A3B8", lw=1)
    extreme = study.tasas.loc[study.tasas.crecimiento_vab.idxmax()]
    ax.annotate(f"{extreme.departamento} {int(extreme.anio)}\n{extreme.crecimiento_vab:+.1f}%", xy=(extreme.crecimiento_vab, ORDER.index(extreme.Nombre_Cluster_K4)), xytext=(-4, -55), textcoords="offset points", ha="center", fontsize=16, color=COLORS[extreme.Nombre_Cluster_K4], arrowprops={"arrowstyle": "->", "color": "#CA5D36"})
    lo, hi = study.tasas.crecimiento_vab.min(), study.tasas.crecimiento_vab.max()
    ax.set(xlim=(min(lo, 0) - .06 * (hi-lo), hi + .15 * (hi-lo)), yticks=range(len(ORDER)), yticklabels=[SHORT[n] for n in ORDER], xlabel="Crecimiento anual del VAB real (%)", title="Todas las observaciones · eje lineal")
    vol = study.volatilidad
    for i, name in enumerate(ORDER):
        full, reduced = vol.loc[name, ["std", "std_sin_exclusion"]]
        std_ax.barh(i + .16, full, height=.29, color=COLORS[name])
        std_ax.barh(i - .16, reduced, height=.29, color=COLORS[name], alpha=.35, edgecolor=NAVY, hatch="//")
        std_ax.text(full + .35, i + .16, f"{full:.1f}", va="center", fontsize=16, weight="bold")
        std_ax.text(reduced + .35, i - .16, f"{reduced:.1f}", va="center", fontsize=16)
    std_ax.set(yticks=range(len(ORDER)), yticklabels=[], xlim=(0, vol["std"].max()*1.25), xlabel="Desviación estándar del crecimiento (pp)", title="El extremo influye en la volatilidad")
    std_ax.legend(handles=[Line2D([], [], color=NAVY, lw=8, label="Muestra completa"), Line2D([], [], color="#A8B7CB", lw=8, label="Sin la observación excluida en M4")], loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=14)
    for axis in axes:
        axis.grid(axis="x")
    guardar(fig, "fig4_volatilidad_estructural", study)


def trayectoria(study):
    m = study.config["muestra"]
    trajectory = study.niveles.groupby(["anio", "Nombre_Cluster_K4"], observed=True).VAB_idx2019.mean().unstack().reindex(columns=ORDER)
    fig, ax = plt.subplots(figsize=(13, 6.2), layout="constrained")
    for name in ORDER:
        ax.plot(trajectory.index, trajectory[name], color=COLORS[name], marker="o", lw=3 if name == ORDER[-1] else 2.2, ls="-" if name == ORDER[-1] else "--", ms=6)
    end = trajectory.iloc[-1].sort_values()
    label_y = []
    for value in end:
        label_y.append(max(value, label_y[-1] + 2.2) if label_y else value)
    for (name, value), y in zip(end.items(), label_y):
        ax.annotate(f"{SHORT[name]}  {value:.1f}", xy=(trajectory.index[-1], value), xytext=(trajectory.index[-1] + .22, y), ha="left", va="center", fontsize=17, color=COLORS[name], weight="bold" if name == ORDER[-1] else "normal", arrowprops={"arrowstyle": "-", "color": COLORS[name], "lw": .9})
    ax.axhline(100, color=NAVY, lw=1.2, ls=":")
    ax.text(trajectory.index[0], 100.7, f"Nivel de {m['base']}", fontsize=16)
    ax.axvspan(m["choque"]-.15, m["choque"]+.15, color="#CA5D36", alpha=.09)
    ax.text(m["choque"], trajectory.min().min() - 1.8, str(m["choque"]), ha="center", fontsize=14, color="#CA5D36")
    values = trajectory.to_numpy()
    span = values.max() - values.min()
    ax.set(xlim=(trajectory.index.min()-.2, trajectory.index.max()+3.1), ylim=(values.min()-.15*span, max(values.max(), max(label_y))+.12*span), xticks=trajectory.index, ylabel=f"Índice de VAB real · {m['base']} = 100", xlabel="Año")
    ax.grid(axis="y")
    guardar(fig, "fig5_trayectoria_vab_indexado", study)


def forest(study):
    confidence = study.config["econometria"]["confianza"]
    parameters = [(f"{period}:Nombre_Cluster_K4[T.{name}]", f"{'2020' if period == 'COVID' else '2021–2023'} · {SHORT[name]}", name) for period in ["COVID", "POST_21_23"] for name in ORDER[:0:-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, layout="constrained")
    for axis, codes, title, unit in [(axes[0], ["M1", "M4"], "Tasas anuales", "Diferencial de crecimiento (pp)"), (axes[1], ["M5"], "Niveles indexados", "Diferencial del índice (puntos)")]:
        bounds = [0.]
        for code in codes:
            model = study.modelos[code]
            ci = model.conf_int(level=confidence)
            for y, (param, label, name) in enumerate(parameters):
                b = model.params[param]
                low, high = ci.loc[param]
                bounds.extend([low, high])
                color = COLORS[name] if code == "M5" else ("#94A3B8" if code == "M1" else "#0077B6")
                offset = -.14 if code == "M1" else (.14 if code == "M4" else 0)
                axis.errorbar(b, y+offset, xerr=np.array([[b-low], [high-b]]), fmt="D" if code == "M4" else "o", color=color, capsize=4, ms=7, lw=2)
        lower, upper = min(bounds), max(bounds)
        pad = (upper-lower)*.10
        axis.set(xlim=(lower-pad, upper+pad), xlabel=f"{unit} · IC {confidence:.0%}", title=title)
        axis.axvline(0, color=NAVY, ls=":", lw=1.2)
        axis.axhline(2.5, color="#CBD5E1", lw=.8)
        axis.grid(axis="x")
    axes[0].set(yticks=range(len(parameters)), yticklabels=[p[1] for p in parameters], ylim=(len(parameters)-.45, -.65))
    axes[0].tick_params(axis="y", labelsize=15)
    axes[0].legend(handles=[Line2D([], [], marker="o", color="#94A3B8", label="M1 · muestra completa"), Line2D([], [], marker="D", color="#0077B6", label="M4 · sin Apurímac 2016")], loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False, fontsize=14)
    axes[1].text(.5, -.30, "M5 · referencia: diversificado / agro", transform=axes[1].transAxes, ha="center", fontsize=14)
    guardar(fig, "fig6_forest_plot_modelos", study)


def generar(study):
    preparar()
    for function in [pca, validacion, perfiles, jerarquia, volatilidad, trayectoria, forest]:
        function(study)
        print(f"Figura: {function.__name__}", flush=True)
