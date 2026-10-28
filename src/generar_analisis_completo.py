"""
Pipeline integral de análisis composicional, clustering jerárquico robusto,
econometría de panel (tasas vs. niveles indexados) y generación de figuras CCDE
con jerarquía visual optimizada y cero colisiones de texto.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from linearmodels.panel import PanelOLS

# Asegurar importación local desde src/
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustering_paso3 import estabilidad, gap_stat

# ---------- Constantes nombradas (Regla CCDE) ----------
RANDOM_STATE = 0
DELTA_CEROS = 0.05
TOTAL_COMP = 100.0
YEAR_MIN_PRE = 2015
YEAR_MAX_PRE = 2019
YEAR_MIN_PANEL = 2008
YEAR_MAX_PANEL = 2023
YEAR_BASE_INDEX = 2019
BOOT_B_GAP = 30
BOOT_B_JACCARD = 100
SUBSAMPLE_RATIO = 0.80

BASE_DIR = SRC_DIR.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"
PRES_FIG_DIR = BASE_DIR / "presentacion" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PRES_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Paleta Institucional CCDE
CCDE_NAVY = "#0A1E40"
CCDE_BLUE = "#22355E"
CCDE_MID = "#2B5B9B"
CCDE_ACCENT = "#0077B6"
CCDE_SKY = "#3D96D2"
CCDE_LIGHT = "#73C4DF"
CCDE_ICE = "#EBF7FC"
CCDE_WARN = "#CA5D36"
CCDE_GREEN = "#16805F"
CCDE_PURPLE = "#784EA0"
CCDE_GRAY = "#585757"

CLUSTER_COLORS = {
    "Diversificado/Agro": CCDE_GREEN,
    "Servicios/Manufactura": CCDE_ACCENT,
    "Minero/Diversificado": CCDE_PURPLE,
    "Minero/Enclave": CCDE_WARN,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.85,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.65,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": CCDE_NAVY,
    "axes.labelcolor": CCDE_NAVY,
    "xtick.color": CCDE_NAVY,
    "ytick.color": CCDE_NAVY,
})


def save_fig(fig, name):
    """Guarda cada figura en PNG (300 dpi) y PDF vectorial tanto en figures/ como en presentacion/figures/."""
    for folder in (FIGURES_DIR, PRES_FIG_DIR):
        fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[fig] Guardada: {name} (.png y .pdf)")


def find_year_col(df):
    for c in df.columns:
        if str(c).strip() != "departamento":
            v = pd.to_numeric(df[c], errors="coerce")
            if v.between(2000, 2030).mean() > 0.9:
                return c
    raise KeyError("No se encontró columna de año")


def main():
    # 1. Cargar datos base y resultados de clustering
    df = pd.read_excel(PROCESSED_DIR / "pbi_departamental_consolidado.xlsx")
    col_anio = find_year_col(df)
    clr = pd.read_excel(PROCESSED_DIR / "clustering_paso1_clr.xlsx", index_col=0)
    ilr = pd.read_excel(PROCESSED_DIR / "clustering_paso1_ilr.xlsx", index_col=0)
    metricas = pd.read_excel(PROCESSED_DIR / "clustering_paso2_metricas.xlsx")
    eti = pd.read_excel(PROCESSED_DIR / "clustering_paso4_etiquetas.xlsx")
    macro = pd.read_excel(PROCESSED_DIR / "econ_paso2_macro.xlsx")

    # 2. PCA sobre CLR
    pca = PCA(random_state=RANDOM_STATE).fit(clr.values)
    X_pca2 = pca.transform(clr.values)[:, :2]
    var_exp = pca.explained_variance_ratio_
    coords_df = pd.DataFrame(X_pca2, index=clr.index, columns=["PC1", "PC2"])
    loadings_df = pd.DataFrame(pca.components_[:3].T, index=clr.columns, columns=["PC1", "PC2", "PC3"])

    # Unir etiquetas y coordenadas
    dept_info = eti.merge(macro[["departamento", "Macro"]], on="departamento").set_index("departamento")
    dept_info = dept_info.join(coords_df)

    # Calcular HHI (escala 0-1) y Share Minería 2015-2019
    win_pre = df[(df[col_anio] >= YEAR_MIN_PRE) & (df[col_anio] <= YEAR_MAX_PRE)].copy()
    if "Otros Servicios" in win_pre.columns:
        win_pre["Otros Actividades/1"] = win_pre["Otros Servicios"].fillna(0) + win_pre["Otros Actividades/1"].fillna(0)
    sectores_cols = list(clr.columns)
    prom_sec = win_pre.groupby("departamento")[sectores_cols].mean()
    col_mineria = [c for c in sectores_cols if "Extracci" in c][0]
    dept_info["Share_Mineria_15_19"] = prom_sec[col_mineria]
    dept_info["HHI_15_19"] = ((prom_sec / 100.0) ** 2).sum(axis=1)

    # =========================================================================
    # FIGURA 1: Biplot PCA + Clusters (Jerarquía Visual sin Superposiciones)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11.2, 6.5))

    # Posiciones calibradas (dx, dy en puntos) y flag de línea guía (leader line)
    # para descongestionar el núcleo central sin ninguna superposición
    dept_label_cfg = {
        "Lima":          (18, 6, False),
        "Lambayeque":    (0, 11, False),
        "San Martín":    (0, 11, False),
        "Amazonas":      (0, 11, False),
        "Ucayali":       (-34, -8, True),
        "La Libertad":   (-48, 8, True),
        "Tumbes":        (4, 24, True),
        "Piura":         (20, 18, True),
        "Ica":           (4, 15, False),
        "Loreto":        (-38, -17, True),
        "Puno":          (-26, -11, False),
        "Huánuco":       (-34, -8, True),
        "Ayacucho":      (-34, -18, True),
        "Cajamarca":     (38, -16, True),
        "Junín":         (32, 4, True),
        "Madre de Dios": (-28, 12, True),
        "Tacna":         (24, -12, False),
        "Cusco":         (28, -4, False),
        "Áncash":        (24, 8, False),
        "Arequipa":      (0, 11, False),
        "Moquegua":      (0, -15, False),
        "Apurímac":      (-32, 9, False),
        "Pasco":         (24, -10, False),
        "Huancavelica":  (0, 11, False),
    }

    # 1. Dibujar primero los vectores sectoriales (capa intermedia zorder=3..4)
    # con posiciones de etiqueta explícitas para no chocar con ningún departamento
    vector_cfg = [
        (col_mineria, "Minería e Hidrocarburos", (2.68, 0.42)),
        ("Manufactura", "Manufactura", (-0.52, 1.98)),
        ("Agricultura, Ganadería, Caza y Silvicultura", "Agropecuario", (-0.42, -1.75)),
        ("Comercio", "Comercio", (-0.92, 0.03)),
        ("Telecom. y Otros Serv. de Información", "Telecom. y Servicios", (-1.05, 1.18)),
        ("Electricidad, Gas y Agua", "Electricidad y Agua", (0.58, -1.02)),
    ]
    scale_vec = 2.75
    for orig_col, short_name, lbl_xy in vector_cfg:
        if orig_col in loadings_df.index:
            lx = loadings_df.loc[orig_col, "PC1"] * scale_vec
            ly = loadings_df.loc[orig_col, "PC2"] * scale_vec
            ax.annotate(
                "", xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.35, alpha=0.72, mutation_scale=11),
                zorder=3
            )
            ax.text(
                lbl_xy[0], lbl_xy[1], short_name,
                fontsize=7.8, style="italic", fontweight="semibold", color=CCDE_MID,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.20", fc="#F1F5F9", ec="#94A3B8", lw=0.6, alpha=0.92)
            )

    # 2. Dibujar los puntos de los 24 departamentos y sus etiquetas (capa superior zorder=5..7)
    for cname, color in CLUSTER_COLORS.items():
        sub = dept_info[dept_info["Nombre_Cluster_K4"] == cname]
        marker = "D" if "Minero" in cname else "o"
        ax.scatter(
            sub["PC1"], sub["PC2"],
            c=color, s=105, marker=marker, alpha=0.94,
            edgecolors="white", linewidth=1.3,
            label=f"{cname} (n={len(sub)})", zorder=5
        )
        for dep, row in sub.iterrows():
            dx, dy, use_leader = dept_label_cfg.get(dep, (0, 9, False))
            arrow_kw = dict(arrowstyle="-", color="#94A3B8", lw=0.75, shrinkA=0, shrinkB=4) if use_leader else None
            ax.annotate(
                dep, (row["PC1"], row["PC2"]),
                xytext=(dx, dy), textcoords="offset points",
                ha="center", va="center", fontsize=8.2, fontweight="bold",
                color=CCDE_NAVY, zorder=7,
                arrowprops=arrow_kw,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78) if use_leader else None
            )

    ax.set_xlim(-4.35, 3.45)
    ax.set_ylim(-2.85, 2.35)
    ax.axhline(0, color="#94A3B8", lw=0.9, ls=":", zorder=1)
    ax.axvline(0, color="#94A3B8", lw=0.9, ls=":", zorder=1)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var. explicada) — Gradiente Extractivo Minero (+)", fontsize=10, fontweight="bold", labelpad=7)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var. explicada) — Manufactura (+) vs. Agropecuario (-)", fontsize=9.8, fontweight="bold", labelpad=7)
    ax.set_title("Espacio Composicional CLR + PCA: Tipología Productiva Departamental (2015–2019)", fontsize=11.8, fontweight="bold", pad=12)
    ax.legend(
        loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1",
        fontsize=8.8, title="Clusters Jerárquicos Ward (K=4)", title_fontsize=9.0
    )
    fig.tight_layout()
    save_fig(fig, "fig1_pca_biplot_clusters")

    # =========================================================================
    # FIGURA 2: Validación de Clustering (Visibilidad Ward vs KMeans + Pico Jaccard)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.9))
    m_pca2 = metricas[metricas["features"] == "CLR+PCA2"]

    style_map = {
        "Ward":   dict(color=CCDE_NAVY, marker="o", ls="-",  lw=2.5, ms=6.5, mfc=CCDE_NAVY, mew=1.2, zorder=5, label="Ward (Jerárquico)"),
        "KMeans": dict(color=CCDE_ACCENT, marker="s", ls="--", lw=2.0, ms=9.5, mfc="none", mew=2.0, zorder=6, label="K-Means (coincide en K=3,4)"),
        "PAM":    dict(color=CCDE_GREEN, marker="^", ls="-.", lw=1.9, ms=7.0, mfc=CCDE_GREEN, mew=1.2, zorder=4, label="PAM (K-Medoids)"),
        "GMM":    dict(color=CCDE_WARN, marker="D", ls=":",  lw=2.0, ms=6.5, mfc=CCDE_WARN, mew=1.2, zorder=3, label="GMM (Mezclas Gaussianas)"),
    }
    for mname in ["GMM", "PAM", "Ward", "KMeans"]:
        grp = m_pca2[m_pca2["metodo"] == mname].sort_values("k")
        if not grp.empty:
            ax1.plot(grp["k"], grp["sil"], **style_map[mname])

    ax1.axvline(2, color=CCDE_NAVY, ls="--", lw=1.0, alpha=0.55)
    ax1.axvline(4, color=CCDE_ACCENT, ls="--", lw=1.0, alpha=0.55)
    ax1.annotate(
        "Óptimo Macro\nK=2 (Sil = 0.498)", xy=(2, 0.498), xytext=(2.28, 0.482),
        arrowprops=dict(arrowstyle="->", color=CCDE_NAVY, lw=1.2),
        fontsize=8.3, fontweight="bold", color=CCDE_NAVY,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=CCDE_NAVY, lw=0.7, alpha=0.94)
    )
    ax1.annotate(
        "Óptimo Tipológico\nK=4 (Sil = 0.407)\nWard ≡ K-Means", xy=(4, 0.407), xytext=(3.18, 0.448),
        arrowprops=dict(arrowstyle="->", color=CCDE_ACCENT, lw=1.2),
        fontsize=8.2, fontweight="bold", color=CCDE_ACCENT,
        bbox=dict(boxstyle="round,pad=0.22", fc=CCDE_ICE, ec=CCDE_ACCENT, lw=0.7, alpha=0.94)
    )
    ax1.set_xticks([2, 3, 4, 5])
    ax1.set_ylim(0.265, 0.520)
    ax1.set_xlabel("Número de Clusters (k)", fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Coeficiente de Silueta (Silhouette)", fontsize=9.8, fontweight="bold")
    ax1.set_title("A. Benchmark de Algoritmos sobre CLR + PCA2", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0)

    # Panel B: Gap Statistic & Estabilidad Bootstrap (Jaccard) con ligero desplazamiento horizontal
    gaps, ses, _ = gap_stat(X_pca2, kmax=5, B=BOOT_B_GAP, seed=RANDOM_STATE)
    est_ward = estabilidad(X_pca2, metodo="Ward", B=BOOT_B_JACCARD, seed=RANDOM_STATE)
    ks_all = np.array([1, 2, 3, 4, 5], dtype=float)
    ks_jac = np.array(list(est_ward.keys()), dtype=float)
    vals_jac = np.array(list(est_ward.values()), dtype=float)

    ax2_twin = ax2.twinx()
    ax2.errorbar(
        ks_all - 0.05, gaps, yerr=ses, fmt="-o", color=CCDE_NAVY,
        lw=2.1, capsize=3.5, ms=6.2, zorder=4, label="Gap Statistic (±1 EE)"
    )
    ax2_twin.plot(
        ks_jac + 0.05, vals_jac, "--s", color=CCDE_WARN,
        lw=2.2, ms=7.2, zorder=5, label="Estabilidad Jaccard (Bootstrap)"
    )
    # Etiqueta directa en el pico de K=4
    jac_k4 = est_ward[4]
    ax2_twin.annotate(
        f"Máxima Estabilidad K=4\nJaccard = {jac_k4:.3f} (85.0%)\nPico local Gap = {gaps[3]:.3f}",
        xy=(4.05, jac_k4), xytext=(2.18, 0.885),
        arrowprops=dict(arrowstyle="->", color=CCDE_WARN, lw=1.25),
        fontsize=8.1, fontweight="bold", color=CCDE_WARN,
        bbox=dict(boxstyle="round,pad=0.24", fc="#FFF7ED", ec=CCDE_WARN, lw=0.8, alpha=0.95)
    )
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xlim(0.7, 5.3)
    ax2.set_ylim(-0.14, 0.54)
    ax2_twin.set_ylim(0.55, 0.96)
    ax2.set_xlabel("Número de Clusters (k)", fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Gap Statistic (Tibshirani, B=30)", fontsize=9.8, fontweight="bold", color=CCDE_NAVY)
    ax2_twin.set_ylabel("Estabilidad Bootstrap Jaccard (B=100)", fontsize=9.8, fontweight="bold", color=CCDE_WARN)
    ax2_twin.grid(False)
    ax2.set_title("B. Gap Statistic y Estabilidad de Remuestreo", fontsize=10.5, fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0)
    fig.tight_layout()
    save_fig(fig, "fig2_validacion_clustering")

    # =========================================================================
    # FIGURA 3: Dendrograma Ward + Perfil Sectorial (Grilla al fondo y Callouts)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw={"width_ratios": [1.02, 1.18]})
    Z_link = linkage(X_pca2, method="ward")
    ax1.set_axisbelow(True)
    ax1.grid(True, linestyle="--", alpha=0.45, zorder=0)
    dendrogram(
        Z_link, labels=list(clr.index), orientation="right",
        leaf_font_size=8.3, color_threshold=3.2, above_threshold_color=CCDE_GRAY, ax=ax1
    )
    ax1.axvline(3.2, color=CCDE_WARN, ls="--", lw=1.3, label="Corte K=4 (Tipología)")
    ax1.axvline(5.5, color=CCDE_NAVY, ls="-.", lw=1.3, label="Corte K=2 (Macro)")
    ax1.set_xlabel("Distancia Euclidiana de Ward (sobre CLR + PCA2)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Jerarquía Anidada: K=4 dentro de K=2", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="white", edgecolor="#CBD5E1")

    # Panel B: Composición sectorial apilada por cluster K=4 con grilla estrictamente detrás
    prof = prom_sec.join(dept_info["Nombre_Cluster_K4"]).groupby("Nombre_Cluster_K4").mean()
    order_k4 = ["Minero/Enclave", "Minero/Diversificado", "Servicios/Manufactura", "Diversificado/Agro"]
    prof = prof.reindex(order_k4)
    prof_agg = pd.DataFrame(index=prof.index)
    prof_agg["Minería e Hidrocarburos"] = prof[col_mineria]
    prof_agg["Manufactura"] = prof["Manufactura"]
    prof_agg["Agropecuario"] = prof["Agricultura, Ganadería, Caza y Silvicultura"]
    prof_agg["Construcción y Elec."] = prof["Construcción"] + prof["Electricidad, Gas y Agua"]
    prof_agg["Comercio y Transp."] = prof["Comercio"] + prof["Transporte, Almacen., Correo y Mensajería"]
    rest_cols = [
        c for c in prof.columns
        if c not in [
            col_mineria, "Manufactura", "Agricultura, Ganadería, Caza y Silvicultura",
            "Construcción", "Electricidad, Gas y Agua", "Comercio", "Transporte, Almacen., Correo y Mensajería"
        ]
    ]
    prof_agg["Servicios y Adm. Pública"] = prof[rest_cols].sum(axis=1)

    ax2.set_axisbelow(True)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax2.grid(False, axis="y")

    sec_colors = [CCDE_WARN, CCDE_NAVY, CCDE_GREEN, CCDE_SKY, CCDE_PURPLE, "#94A3B8"]
    left_cum = np.zeros(len(prof_agg))
    for col_s, scolor in zip(prof_agg.columns, sec_colors):
        vals = prof_agg[col_s].values
        ax2.barh(
            prof_agg.index, vals, left=left_cum, color=scolor,
            edgecolor="white", linewidth=1.1, height=0.56, label=col_s, zorder=3
        )
        for idx_b, v in enumerate(vals):
            if v >= 7.8:
                txt_col = "white" if scolor in (CCDE_WARN, CCDE_NAVY, CCDE_GREEN, CCDE_PURPLE) else CCDE_NAVY
                ax2.text(
                    left_cum[idx_b] + v / 2.0, idx_b, f"{v:.1f}%",
                    va="center", ha="center", color=txt_col,
                    fontsize=8.2, fontweight="bold", zorder=5
                )
        left_cum += vals

    # Callouts externos para los segmentos pequeños clave
    # Minería en Diversificado/Agro (idx=3, x~0.45) y Manufactura en Minero/Enclave (idx=0, x~43.0)
    ax2.annotate(
        "Minería: 0.9%", xy=(0.5, 3.28), xytext=(4.5, 3.43),
        arrowprops=dict(arrowstyle="->", color=CCDE_WARN, lw=1.0),
        fontsize=7.6, fontweight="bold", color=CCDE_WARN, va="center", zorder=6
    )
    ax2.annotate(
        "Manuf.: 2.1%", xy=(42.9, -0.28), xytext=(33.0, -0.44),
        arrowprops=dict(arrowstyle="->", color=CCDE_NAVY, lw=1.0),
        fontsize=7.6, fontweight="bold", color=CCDE_NAVY, va="center", zorder=6
    )

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.55, 3.60)
    ax2.set_xlabel("Participación Media en el VAB Departamental 2015–2019 (%)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Estructura Productiva por Cluster (%)", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.48, -0.15), ncol=3, frameon=False, fontsize=8.2)
    fig.tight_layout()
    save_fig(fig, "fig3_dendrograma_perfiles")

    # =========================================================================
    # PREPARACIÓN DEL PANEL ECONOMÉTRICO FINAL (VENTANA EFECTIVA 2008–2023)
    # =========================================================================
    panel_all = df.merge(
        dept_info.reset_index()[[
            "departamento", "Cluster_K2", "Cluster_K4", "Nombre_Cluster_K4",
            "Macro", "PC1", "PC2", "Share_Mineria_15_19", "HHI_15_19"
        ]],
        on="departamento", how="left"
    )
    panel_all["COVID"] = (panel_all[col_anio] == 2020).astype(int)
    panel_all["REBOTE"] = (panel_all[col_anio] == 2021).astype(int)
    panel_all["POST_21_23"] = (panel_all[col_anio].between(2021, 2023)).astype(int)
    panel_all["Enclave"] = (panel_all["Nombre_Cluster_K4"] == "Minero/Enclave").astype(int)

    pib_2019_map = panel_all[panel_all[col_anio] == YEAR_BASE_INDEX].set_index("departamento")["PIB_total"]
    panel_all["PIB_idx2019"] = (panel_all["PIB_total"] / panel_all["departamento"].map(pib_2019_map)) * 100.0
    panel_all["dln_PIB_2019"] = (np.log(panel_all["PIB_total"]) - np.log(panel_all["departamento"].map(pib_2019_map))) * 100.0

    min_mean = dept_info["Share_Mineria_15_19"].mean()
    min_std = dept_info["Share_Mineria_15_19"].std()
    pc1_mean = dept_info["PC1"].mean()
    pc1_std = dept_info["PC1"].std()
    panel_all["Mineria_z"] = (panel_all["Share_Mineria_15_19"] - min_mean) / min_std
    panel_all["PC1_z"] = (panel_all["PC1"] - pc1_mean) / pc1_std

    panel_all.to_excel(PROCESSED_DIR / "panel_econometrico_final.xlsx", index=False)
    print("[data] Guardado: panel_econometrico_final.xlsx", panel_all.shape)

    # Ventana efectiva 2008-2023 con var_pib válido (380 observaciones)
    panel = panel_all[
        (panel_all[col_anio] >= YEAR_MIN_PANEL)
        & (panel_all[col_anio] <= YEAR_MAX_PANEL)
        & (panel_all["var_pib"].notna())
    ].copy()

    # =========================================================================
    # FIGURA 4: Régimen de Volatilidad Estructural con EJE X PARTIDO (Broken Axis)
    # =========================================================================
    fig = plt.figure(figsize=(12.4, 5.1))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 0.28, 1.05], wspace=0.06)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1], sharey=ax1a)
    ax2 = fig.add_subplot(gs[0, 2])
    # Añadir espacio extra entre ax1b y ax2 ajustando posición
    pos_b = ax1b.get_position()
    pos_2 = ax2.get_position()
    ax2.set_position([pos_2.x0 + 0.035, pos_2.y0, pos_2.width - 0.02, pos_2.height])

    order_plot = ["Diversificado/Agro", "Servicios/Manufactura", "Minero/Diversificado", "Minero/Enclave"]
    data_box = [panel[panel["Nombre_Cluster_K4"] == c]["var_pib"].dropna().values for c in order_plot]

    bp = ax1a.boxplot(
        data_box, vert=False, patch_artist=True, tick_labels=order_plot,
        widths=0.48, showfliers=False, zorder=2
    )
    for patch, cname in zip(bp["boxes"], order_plot):
        patch.set_facecolor(CLUSTER_COLORS[cname])
        patch.set_alpha(0.32)
        patch.set_edgecolor(CLUSTER_COLORS[cname])
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color(CCDE_NAVY)
        median.set_linewidth(2.1)

    rng = np.random.default_rng(RANDOM_STATE)
    for idx_c, cname in enumerate(order_plot, start=1):
        sub = panel[panel["Nombre_Cluster_K4"] == cname]
        sub_norm = sub[sub[col_anio] != 2020]
        sub_2020 = sub[sub[col_anio] == 2020]
        jitter = rng.uniform(-0.15, 0.15, size=len(sub_norm))
        # Graficar en subpanel izquierdo (-24 a +33) y subpanel derecho (130 a 148)
        for ax_sub in (ax1a, ax1b):
            ax_sub.scatter(
                sub_norm["var_pib"], np.full(len(sub_norm), idx_c) + jitter,
                color=CLUSTER_COLORS[cname], alpha=0.60, s=24, edgecolors="white", linewidth=0.3, zorder=3
            )
        ax1a.scatter(
            sub_2020["var_pib"], np.full(len(sub_2020), idx_c),
            color=CCDE_NAVY, marker="X", s=54, edgecolors="white", linewidth=0.5,
            zorder=5, label="Choque COVID (2020)" if idx_c == 1 else ""
        )

    ax1a.axvline(0, color=CCDE_GRAY, ls="--", lw=1.0, zorder=1)
    ax1a.set_xlim(-23.5, 32.5)
    ax1b.set_xlim(133.0, 148.0)
    ax1a.set_xticks([-20, -10, 0, 10, 20, 30])
    ax1b.set_xticks([141.2])
    ax1b.set_xticklabels(["+141.2%"], fontweight="bold", color=CCDE_WARN)

    # Ocultar bordes internos entre ax1a y ax1b para el eje partido
    ax1a.spines["right"].set_visible(False)
    ax1b.spines["left"].set_visible(False)
    ax1b.tick_params(axis="y", left=False, labelleft=False)

    # Dibujar marcas diagonales de corte de eje (//)
    d = 0.015
    kwargs_break = dict(transform=ax1a.transAxes, color="#64748B", clip_on=False, lw=1.2)
    ax1a.plot((1 - d, 1 + d), (-d, +d), **kwargs_break)
    ax1a.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_break)
    kwargs_break.update(transform=ax1b.transAxes)
    ax1b.plot((-d * 3.5, +d * 3.5), (-d, +d), **kwargs_break)
    ax1b.plot((-d * 3.5, +d * 3.5), (1 - d, 1 + d), **kwargs_break)

    # Anotar el outlier Apurímac 2016 con nitidez en el subpanel derecho / puente
    apu16_val = panel[(panel["departamento"].str.contains("Apur")) & (panel[col_anio] == 2016)]["var_pib"].values[0]
    ax1b.annotate(
        f"Apurímac 2016\n(+{apu16_val:.1f}%)\nHito Las Bambas",
        xy=(apu16_val, 4.0), xytext=(140.5, 2.55),
        arrowprops=dict(arrowstyle="->", color=CCDE_WARN, lw=1.6),
        ha="center", va="center", fontsize=8.0, fontweight="bold", color=CCDE_WARN,
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF7ED", ec=CCDE_WARN, lw=0.9), zorder=6
    )

    ax1a.set_xlabel("Tasa de Variación Anual del PBI Real Departamental (%, Eje Partido)", fontsize=9.5, fontweight="bold")
    ax1a.set_title("A. Distribución de Crecimiento Anual y Shock Las Bambas", fontsize=10.5, fontweight="bold", loc="left")
    ax1a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.2)
    ax1a.tick_params(axis="y", labelsize=9.2)
    for lbl in ax1a.get_yticklabels():
        lbl.set_fontweight("bold")

    # Panel B: Desviación estándar por cluster sin tapar barras con la leyenda
    stds_pooled = [panel[panel["Nombre_Cluster_K4"] == c]["var_pib"].std() for c in order_plot]
    stds_within = [panel[panel["Nombre_Cluster_K4"] == c].groupby("departamento")["var_pib"].std().mean() for c in order_plot]
    y_pos = np.arange(len(order_plot))

    ax2.barh(
        y_pos + 0.17, stds_pooled, height=0.33,
        color=[CLUSTER_COLORS[c] for c in order_plot], alpha=0.92, zorder=3
    )
    ax2.barh(
        y_pos - 0.17, stds_within, height=0.33,
        color=[CLUSTER_COLORS[c] for c in order_plot], alpha=0.38,
        hatch="//", edgecolor=CCDE_NAVY, linewidth=0.8, zorder=3
    )
    for i, (v_p, v_w) in enumerate(zip(stds_pooled, stds_within)):
        ax2.text(v_p + 0.45, i + 0.17, f"{v_p:.1f}%", va="center", fontsize=8.3, fontweight="bold", color=CCDE_NAVY, zorder=5)
        ax2.text(v_w + 0.45, i - 0.17, f"{v_w:.1f}%", va="center", fontsize=8.0, color=CCDE_GRAY, zorder=5)

    lev_stat, lev_p = stats.levene(*data_box, center="median")
    flig_stat, flig_p = stats.fligner(*data_box)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 28.0)
    ax2.set_xlabel("Volatilidad del Crecimiento (Desv. Estándar σ, pp)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Régimen de Volatilidad Estructural (2008–2023)", fontsize=10.5, fontweight="bold")

    # Leyenda limpia ubicada en la zona derecha libre entre y=1 y y=2 (x > 11.5, lejos de 8.5%)
    patch_total = mpatches.Patch(facecolor=CCDE_BLUE, alpha=0.90, label="Desv. Estándar Total (σ)")
    patch_intra = mpatches.Patch(facecolor="#CBD5E1", edgecolor=CCDE_NAVY, hatch="//", label="Desv. Estándar Intra-Dpto.")
    ax2.legend(
        handles=[patch_total, patch_intra], loc="center right",
        bbox_to_anchor=(0.98, 0.46), frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=7.6
    )

    ax2.text(
        0.98, 0.06,
        f"Pruebas de Homocedasticidad:\n• Levene F = {lev_stat:.2f} (p = {lev_p:.4f})\n• Fligner-Killeen χ² = {flig_stat:.2f} (p = {flig_p:.4f})\n→ Varianza 4.4× mayor en Enclave",
        transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.28", fc=CCDE_ICE, ec=CCDE_ACCENT, lw=0.9), zorder=5
    )
    save_fig(fig, "fig4_volatilidad_estructural")

    # =========================================================================
    # FIGURA 5: Trayectoria del PBI Real Indexado (2019 = 100) con Valle 2020 y Final 2023
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    win_idx = panel_all[panel_all[col_anio].between(2017, 2023)].copy()

    # Offsets verticales para las etiquetas finales de 2023 y del valle de 2020
    end_offsets = {
        "Diversificado/Agro": 0.35,
        "Minero/Diversificado": -0.25,
        "Servicios/Manufactura": -0.35,
        "Minero/Enclave": 0.0,
    }
    trough_cfg = {
        "Diversificado/Agro":    (-0.38, 93.6),
        "Minero/Diversificado":  (0.38, 91.0),
        "Servicios/Manufactura": (-0.38, 88.8),
        "Minero/Enclave":        (0.38, 86.2),
    }

    for cname in order_plot:
        sub = win_idx[win_idx["Nombre_Cluster_K4"] == cname]
        stats_yr = sub.groupby(col_anio)["PIB_idx2019"].agg(["mean", "sem"]).reset_index()
        lw = 3.1 if cname == "Minero/Enclave" else 2.2
        ls = "-" if cname in ("Minero/Enclave", "Diversificado/Agro") else "--"
        ax.plot(
            stats_yr[col_anio], stats_yr["mean"],
            marker="o", lw=lw, ls=ls, ms=6.5,
            color=CLUSTER_COLORS[cname], label=cname, zorder=4
        )
        if cname in ("Minero/Enclave", "Diversificado/Agro"):
            ax.fill_between(
                stats_yr[col_anio],
                stats_yr["mean"] - 1.0 * stats_yr["sem"],
                stats_yr["mean"] + 1.0 * stats_yr["sem"],
                color=CLUSTER_COLORS[cname], alpha=0.13, zorder=2
            )
        # Etiqueta al final (2023)
        val_2023 = stats_yr.loc[stats_yr[col_anio] == 2023, "mean"].values[0]
        dy = end_offsets.get(cname, 0.0)
        ax.text(
            2023.13, val_2023 + dy, f"{val_2023:.1f}",
            fontsize=8.9, fontweight="bold", color=CLUSTER_COLORS[cname], va="center", zorder=6
        )
        # Etiqueta en el valle de 2020
        val_2020 = stats_yr.loc[stats_yr[col_anio] == 2020, "mean"].values[0]
        dx_t, y_t = trough_cfg[cname]
        ax.annotate(
            f"{val_2020:.1f}", xy=(2020, val_2020), xytext=(2020 + dx_t, y_t),
            arrowprops=dict(arrowstyle="-", color=CLUSTER_COLORS[cname], lw=0.75, alpha=0.7),
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=CLUSTER_COLORS[cname],
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=CLUSTER_COLORS[cname], lw=0.5, alpha=0.9), zorder=6
        )

    ax.axhline(100, color=CCDE_NAVY, lw=1.2, ls=":", label="Nivel Pre-Pandemia (2019 = 100)", zorder=2)
    ax.axvspan(2019.55, 2020.45, color=CCDE_WARN, alpha=0.07, label="Choque COVID-19 (2020)", zorder=1)
    ax.annotate(
        "Efecto Cicatriz (Scarring Effect):\nMinero/Enclave permanece bajo su\nnivel de 2019 durante 2021–2023\n(97.6 en 2023; β = -10.92 pts, p = 0.0029)",
        xy=(2022, 94.6), xytext=(2020.62, 85.2),
        arrowprops=dict(arrowstyle="->", color=CCDE_WARN, lw=1.5),
        fontsize=8.4, fontweight="bold", color=CCDE_WARN,
        bbox=dict(boxstyle="round,pad=0.30", fc="#FFF7ED", ec=CCDE_WARN, lw=0.9), zorder=6
    )
    ax.set_xlim(2016.75, 2023.75)
    ax.set_ylim(82.5, 109.5)
    ax.set_xticks(range(2017, 2024))
    ax.set_xlabel("Año", fontsize=10, fontweight="bold")
    ax.set_ylabel("Índice de PBI Real Departamental (Base 2019 = 100)", fontsize=10, fontweight="bold")
    ax.set_title("Trayectoria Acumulada del PBI Real (2017–2023): Choque 2020 vs. Cicatriz Post-Pandemia", fontsize=11.4, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.6)
    fig.tight_layout()
    save_fig(fig, "fig5_trayectoria_pib_indexado")

    # =========================================================================
    # ESTIMACIÓN DE LA BATERÍA DE MODELOS ECONOMÉTRICOS (M1 A M6)
    # =========================================================================
    d_full = panel.set_index(["departamento", col_anio]).copy()
    d_full["Nombre_Cluster_K4"] = pd.Categorical(
        d_full["Nombre_Cluster_K4"],
        categories=["Diversificado/Agro", "Servicios/Manufactura", "Minero/Diversificado", "Minero/Enclave"],
        ordered=True
    )
    d_full["Macro"] = pd.Categorical(d_full["Macro"], categories=["Diversificado", "Minero"], ordered=True)

    # M1: Modelo Base K=4 sobre var_pib (2008-2023, 380 obs)
    m1 = PanelOLS.from_formula(
        "var_pib ~ 1 + COVID + POST_21_23 + COVID:Nombre_Cluster_K4 + POST_21_23:Nombre_Cluster_K4 + EntityEffects",
        data=d_full
    ).fit(cov_type="clustered", cluster_entity=True)

    # M2: Modelo Macro K=2 sobre var_pib (380 obs)
    m2 = PanelOLS.from_formula(
        "var_pib ~ 1 + COVID + POST_21_23 + COVID:Macro + POST_21_23:Macro + EntityEffects",
        data=d_full
    ).fit(cov_type="clustered", cluster_entity=True)

    # M3: Modelo Continuo Mineria_z sobre var_pib (380 obs)
    m3 = PanelOLS.from_formula(
        "var_pib ~ 1 + COVID + POST_21_23 + COVID:Mineria_z + POST_21_23:Mineria_z + EntityEffects",
        data=d_full
    ).fit(cov_type="clustered", cluster_entity=True)

    # M4: Modelo K=4 sin Outlier Apurímac 2016 (379 obs)
    d_no_apu = d_full.reset_index()
    d_no_apu = d_no_apu[~((d_no_apu["departamento"].str.contains("Apur")) & (d_no_apu[col_anio] == 2016))].set_index(["departamento", col_anio])
    m4 = PanelOLS.from_formula(
        "var_pib ~ 1 + COVID + POST_21_23 + COVID:Nombre_Cluster_K4 + POST_21_23:Nombre_Cluster_K4 + EntityEffects",
        data=d_no_apu
    ).fit(cov_type="clustered", cluster_entity=True)

    # M5: Modelo en Niveles Indexados (PIB_idx2019, 2017-2023, 168 obs) con Tipología K=4
    d_idx = panel_all[panel_all[col_anio].between(2017, 2023)].copy()
    d_idx["COVID_20"] = (d_idx[col_anio] == 2020).astype(int)
    d_idx["POST_21_23"] = (d_idx[col_anio] >= 2021).astype(int)
    d_idx["Nombre_Cluster_K4"] = pd.Categorical(
        d_idx["Nombre_Cluster_K4"],
        categories=["Diversificado/Agro", "Servicios/Manufactura", "Minero/Diversificado", "Minero/Enclave"],
        ordered=True
    )
    d_idx = d_idx.set_index(["departamento", col_anio])
    m5 = PanelOLS.from_formula(
        "PIB_idx2019 ~ 1 + COVID_20 + POST_21_23 + COVID_20:Nombre_Cluster_K4 + POST_21_23:Nombre_Cluster_K4 + EntityEffects",
        data=d_idx
    ).fit(cov_type="clustered", cluster_entity=True)

    # M6: Modelo en Niveles Indexados con Intensidad Minera Continua (Mineria_z)
    m6 = PanelOLS.from_formula(
        "PIB_idx2019 ~ 1 + COVID_20 + POST_21_23 + COVID_20:Mineria_z + POST_21_23:Mineria_z + EntityEffects",
        data=d_idx
    ).fit(cov_type="clustered", cluster_entity=True)

    # Guardar tabla resumen de modelos
    summary_rows = []
    for mcode, mlabel, dep_var, mod in [
        ("M1", "K=4 en Tasas (2008-2023)", "var_pib", m1),
        ("M2", "Macro K=2 en Tasas (2008-2023)", "var_pib", m2),
        ("M3", "Continuo Minería_z en Tasas", "var_pib", m3),
        ("M4", "K=4 en Tasas sin Apurímac 2016", "var_pib", m4),
        ("M5", "K=4 en Nivel Indexado (2019=100)", "PIB_idx2019", m5),
        ("M6", "Continuo Minería_z en Nivel Indexado", "PIB_idx2019", m6),
    ]:
        for param_name in mod.params.index:
            summary_rows.append({
                "Modelo": mcode,
                "Especificacion": mlabel,
                "Dependiente": dep_var,
                "Parametro": param_name,
                "Coeficiente": float(mod.params[param_name]),
                "Error_Estandar": float(mod.std_errors[param_name]),
                "T_stat": float(mod.tstats[param_name]),
                "P_valor": float(mod.pvalues[param_name]),
                "R2_within": float(mod.rsquared_within),
                "N_obs": int(mod.nobs),
                "Poolability_F": float(mod.f_pooled.stat),
                "Poolability_p": float(mod.f_pooled.pval),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(PROCESSED_DIR / "resumen_modelos_econometricos.xlsx", index=False)
    print("[data] Guardado: resumen_modelos_econometricos.xlsx")

    # =========================================================================
    # FIGURA 6: Forest Plot Comparativo (Pastillas Blancas sin cruce en x=0)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)

    params_order = [
        ("COVID:Nombre_Cluster_K4[T.Minero/Enclave]",          "COVID_20:Nombre_Cluster_K4[T.Minero/Enclave]",          "2020 × Minero/Enclave"),
        ("COVID:Nombre_Cluster_K4[T.Minero/Diversificado]",    "COVID_20:Nombre_Cluster_K4[T.Minero/Diversificado]",    "2020 × Minero/Diversif."),
        ("COVID:Nombre_Cluster_K4[T.Servicios/Manufactura]",   "COVID_20:Nombre_Cluster_K4[T.Servicios/Manufactura]",   "2020 × Serv./Manuf."),
        ("POST_21_23:Nombre_Cluster_K4[T.Minero/Enclave]",     "POST_21_23:Nombre_Cluster_K4[T.Minero/Enclave]",        "Recup 21–23 × Minero/Enclave"),
        ("POST_21_23:Nombre_Cluster_K4[T.Minero/Diversificado]","POST_21_23:Nombre_Cluster_K4[T.Minero/Diversificado]",  "Recup 21–23 × Minero/Diversif."),
        ("POST_21_23:Nombre_Cluster_K4[T.Servicios/Manufactura]","POST_21_23:Nombre_Cluster_K4[T.Servicios/Manufactura]","Recup 21–23 × Serv./Manuf."),
    ]
    y_idx = np.arange(len(params_order))[::-1]

    # Panel A: M1 (Base, gris) vs M4 (Sin Apurímac 2016, azul)
    ax1.axvline(0, color=CCDE_NAVY, ls="--", lw=1.2, zorder=1)
    for y, (p_rate, _, lbl) in zip(y_idx, params_order):
        b1, se1, pv1 = m1.params[p_rate], m1.std_errors[p_rate], m1.pvalues[p_rate]
        b4, se4, pv4 = m4.params[p_rate], m4.std_errors[p_rate], m4.pvalues[p_rate]

        ax1.errorbar(
            b1, y + 0.15, xerr=1.96 * se1, fmt="o", color="#64748B", ecolor="#94A3B8",
            elinewidth=1.8, capsize=3.5, ms=5.8, zorder=3,
            label="M1: Panel completo (R²w = 0.138)" if y == y_idx[0] else ""
        )
        ax1.errorbar(
            b4, y - 0.15, xerr=1.96 * se4, fmt="D", color=CCDE_ACCENT, ecolor=CCDE_ACCENT,
            elinewidth=2.0, capsize=3.5, ms=5.8, zorder=4,
            label="M4: Sin Apurímac 2016 (R²w = 0.262)" if y == y_idx[0] else ""
        )
        # Anotar solo los parámetros donde M4 difiere de M1 (Minero/Enclave) con pastilla blanca
        if "Minero/Enclave" in lbl:
            ax1.text(
                b4 - 0.6, y - 0.37, f"M4: β={b4:+.2f} (p={pv4:.2f})",
                ha="center", va="center", fontsize=7.6, fontweight="bold", color=CCDE_ACCENT,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=CCDE_ACCENT, lw=0.5, alpha=0.95), zorder=6
            )
        else:
            ax1.text(
                b1, y + 0.35, f"β={b1:+.2f} (p={pv1:.2f})",
                ha="center", va="center", fontsize=7.5, color="#475569",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.90), zorder=5
            )

    ax1.set_yticks(y_idx)
    ax1.set_yticklabels([x[2] for x in params_order], fontsize=9.0, fontweight="bold")
    ax1.set_xlim(-21.5, 13.5)
    ax1.set_ylim(-1.65, len(params_order) - 0.20)
    ax1.set_xlabel("Diferencial en Crecimiento Anual Δ% PBI (pp, IC 95%)", fontsize=9.3, fontweight="bold")
    ax1.set_title("A. Tasas Anuales (M1 vs. M4): Alta Varianza e Ilusión de Base\n(Aislar Las Bambas en M4 reduce el EE de Enclave en 73%)", fontsize=9.8, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=7.9)

    # Panel B: M5 (Nivel Indexado PBI 2019=100, K=4) + recuadro de M6 (Continuo)
    ax2.axvline(0, color=CCDE_NAVY, ls="--", lw=1.2, zorder=1)
    for y, (_, p_idx, lbl) in zip(y_idx, params_order):
        b5, se5, pv5 = m5.params[p_idx], m5.std_errors[p_idx], m5.pvalues[p_idx]
        is_sig = pv5 < 0.05
        is_marg = 0.05 <= pv5 < 0.10
        c_use = CCDE_WARN if is_sig else (CCDE_PURPLE if is_marg else "#64748B")
        fmt_m = "D" if (is_sig or is_marg) else "o"

        ax2.errorbar(
            b5, y, xerr=1.96 * se5, fmt=fmt_m, color=c_use, ecolor=c_use,
            elinewidth=2.3 if is_sig else 1.8, capsize=4.0, ms=6.8 if is_sig else 5.8, zorder=4
        )
        # Colocar etiqueta con pastilla blanca ligeramente desplazada para nunca tocar x=0
        stars = "***" if pv5 < 0.01 else ("**" if pv5 < 0.05 else ("*" if pv5 < 0.10 else ""))
        ax2.text(
            b5, y + 0.30, f"β = {b5:+.2f}{stars} (p = {pv5:.3f})",
            ha="center", va="center", fontsize=8.0,
            fontweight="bold" if (is_sig or is_marg) else "normal", color=c_use,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c_use if is_sig else "none", lw=0.6 if is_sig else 0, alpha=0.95),
            zorder=6
        )

    # Añadir caja compacta de M6 en la banda inferior dedicada (y < 0) sin rebasar el ancho de ax2
    b6_cov, p6_cov = m6.params["COVID_20:Mineria_z"], m6.pvalues["COVID_20:Mineria_z"]
    b6_post, p6_post = m6.params["POST_21_23:Mineria_z"], m6.pvalues["POST_21_23:Mineria_z"]
    ax2.text(
        0.02, 0.02,
        f"Consistencia Continua (M6, Minería_z):\n• 2020 × Minería_z: β = {b6_cov:+.2f}* (p = {p6_cov:.3f})\n• Recup 21–23 × Minería_z: β = {b6_post:+.2f}* (p = {p6_post:.3f})",
        transform=ax2.transAxes, ha="left", va="bottom", fontsize=7.6, fontweight="bold", color=CCDE_BLUE,
        bbox=dict(boxstyle="round,pad=0.22", fc=CCDE_ICE, ec=CCDE_ACCENT, lw=0.8, alpha=0.96), zorder=6
    )

    ax2.set_xlim(-20.5, 9.5)
    ax2.set_xlabel("Diferencial en Nivel de PBI Real Indexado 2019 = 100 (pts, IC 95%)", fontsize=9.3, fontweight="bold")
    ax2.set_title("B. Nivel Indexado (M5 y M6): Evidencia del Efecto Cicatriz\n(R² within = 0.355 | Test F Poolability = 3.14, p < 0.0001)", fontsize=9.8, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig6_forest_plot_modelos")
    print("Pipeline completo finalizado con éxito.")


if __name__ == "__main__":
    main()
