# Clusters Departamentales: Estructura Productiva Regional, Volatilidad Macroeconómica y el Efecto Cicatriz del COVID-19 en el Perú

Investigación empírica sobre la especialización productiva regional del Perú (2007–2024) combinando **Análisis de Datos Composicionales (CoDa)** en la geometría de Aitchison (2015–2019) y **Modelos de Datos de Panel con Efectos Fijos** (2008–2023) en tasas de crecimiento anuales ($\Delta\%\text{ PBI}$) y en niveles reales indexados ($\text{PBI}_{2019=100}$).

---

## Principales Hallazgos Empíricos

1. **Taxonomía Productiva Composicional ($K=4$ y $K=2$, ventana pre-COVID 2015–2019)**:
   - Las participaciones de los 12 sectores VAB del INEI se proyectan del simplex $\mathcal{S}^{12}$ al espacio real mediante reemplazo multiplicativo ($\delta = 0.05\%$) y coordenadas **CLR** (*Centered Log-Ratio*) / **ILR**.
   - Los dos primeros componentes principales explican el **71.68%** de la varianza (**87.39%** en 3D): **PC1 (51.65%)** captura intensidad extractiva minera ($+0.906$), mientras que **PC2 (20.02%)** opone manufactura ($+0.630$) frente a agropecuaria ($-0.542$).
   - Los algoritmos de Ward y K-Means convergen exactamente en la misma partición en **$K=4$** ($\text{Silhouette} = 0.407$, $\text{Gap} = 0.231$, estabilidad Bootstrap Jaccard de **85.0%–87.7%**), la cual anida estrictamente en dos macro-bloques (**$K=2$**, $\text{Silhouette} = 0.498$):
     - **Macro Minero ($N=10$)**: *Minero/Enclave* ($N=3$: Apurímac, Huancavelica, Pasco; $41.9\%$ minería) y *Minero/Diversificado* ($N=7$: Arequipa, Cusco, Junín, Madre de Dios, Moquegua, Tacna, Áncash; $37.4\%$ minería y $12.2\%$ manufactura).
     - **Macro Diversificado ($N=14$)**: *Servicios/Manufactura* ($N=11$) y *Diversificado/Agro* ($N=3$: Lima, Lambayeque, San Martín; $0.9\%$ minería).

2. **Resolución del Puzzle de las Tasas Anuales ($\Delta\%\text{ PBI}$) y Régimen de Volatilidad**:
   - En los modelos de efectos fijos sobre tasas de variación anual (**M1–M3**, 2008–2023, $N=380$), el shock COVID-19 de 2020 es altamente significativo ($\approx -14.41\text{ pp}, p < 0.0001$), pero las interacciones $\text{COVID} \times \text{Cluster}$ resultan estadísticamente no significativas ($p \in [0.20, 0.91]$).
   - Esto obedece a tres mecanismos: (i) el confinamiento estricto de Q2-2020 paralizó simultáneamente todos los sectores; (ii) el régimen de **heterocedasticidad estructural** entre clusters —la volatilidad de *Minero/Enclave* ($\sigma = 20.46\%$) es **$4.4\times$ mayor** que la de *Diversificado/Agro* ($\sigma = 4.65\%$, Levene $p = 0.0083$, Fligner-Killeen $p = 0.0005$)— impulsado por el shock de oferta de **Las Bambas en Apurímac (2016: $+141.2\%$)**: al excluir esa única celda (**M4**), el $R^2_{\text{within}}$ casi se duplica de $0.1381$ a **$0.2623$**; y (iii) la ilusión aritmética de rebote por efecto base en tasas anuales.

3. **El Efecto Cicatriz (*Scarring Effect*) en Niveles Reales Indexados ($\text{PBI}_{2019=100}$, M5–M6)**:
   - Al modelar el índice real acumulado base $2019=100$ en un panel balanceado (**2017–2023**, $N=24, T=7, \text{Obs}=168$), el $R^2_{\text{within}}$ asciende a **$0.3554$** y el test $F$ de efectos fijos departamentales resulta altamente significativo ($F = 3.14, p < 0.0001$).
   - El cluster **Minero/Enclave** sufrió una caída adicional estadísticamente significativa en 2020 ($\beta = -6.57\text{ pts}, p = 0.0495$) y un **rezago estructural de recuperación (Efecto Cicatriz) en 2021–2023 de $\beta = -10.92\text{ pts}$ ($t = -3.03, p = 0.0029$)**, permaneciendo por debajo de su nivel pre-pandemia durante todo el trienio 2021–2023 ($95.9 \rightarrow 94.6 \rightarrow 97.6$), mientras los demás clusters superaron el umbral de 100 ($103.1\text{–}106.6$).
   - La especificación con predictor continuo estandarizado (**M6**, `Mineria_z`) confirma que $+1\text{ SD}$ en participación minera pre-COVID reduce el nivel de recuperación 2021–2023 en $-2.76\text{ pts}$ ($p = 0.0747$).

---

## Estructura del Repositorio

```text
.
├── assets/
│   ├── ccde-beamer.sty                          # Hoja de estilo visual CCDE para Beamer
│   ├── ccde-logo-positivo.png                   # Logotipo institucional positivo recortado (fondo transparente)
│   ├── ccde-logo-negativo.png                   # Logotipo institucional negativo recortado (fondo transparente)
│   ├── ccde-logo-negativo-fondo.png             # Logotipo institucional negativo recortado (con fondo azul)
│   ├── Logo positivo CCDE.jpg                   # Logotipo institucional positivo (formato original)
│   └── Logo final negativo.png                  # Logotipo institucional negativo (formato original)
├── data/
│   ├── processed/                               # Bases consolidadas y resultados econométricos
│   │   ├── pbi_departamental_consolidado.xlsx   # Base consolidada INEI (2007-2024)
│   │   ├── Resultados_Clusters_Departamentales.xlsx
│   │   ├── Metricas_Validacion_Clustering.xlsx
│   │   ├── panel_econometrico_base.xlsx
│   │   ├── panel_econometrico_final.xlsx        # Panel enriquecido con PIB_idx2019 y predictores continuos
│   │   └── resumen_modelos_econometricos.xlsx   # Tabla comparativa de modelos M1 a M6
│   └── raw/                                     # 24 libros Excel originales INEI (excluidos en git)
├── figures/                                     # 6 figuras de calidad publicación (PNG 300 DPI + PDF vectorial)
├── logs/                                        # Bitácoras metodológicas del proyecto
├── presentacion/                                # Presentación empírica en diseño visual CCDE
│   ├── presentacion-clusters-departamentales.tex
│   ├── presentacion-clusters-departamentales.pdf
│   ├── presentacion-clusters-departamentales.pptx
│   ├── figures/                                 # Copia local de figuras para compilación autónoma
│   └── sections/                                # Secciones modulares LaTeX Beamer (01 a 05)
├── src/
│   ├── generador-df.py                          # ETL de consolidación de archivos Excel INEI
│   ├── clustering_paso1.py                      # Pipeline CoDa CLR/ILR, PCA y validación de clustering
│   ├── generar_analisis_completo.py             # Pipeline maestro: CoDa + Panel FE (M1-M6) + 6 figuras
│   ├── modelo-clusters.ipynb                    # Notebook ejecutado: Análisis Composicional y Clustering
│   └── modelo-econometrico.ipynb                # Notebook ejecutado: Econometría de Panel y Efecto Cicatriz
└── requirements.txt
```

---

## Reproducibilidad y Ejecución

1. **Crear y activar el entorno virtual**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Ejecutar el pipeline analítico y regenerar bases y figuras**:

   ```bash
   python src/clustering_paso1.py
   python src/generar_analisis_completo.py
   ```

3. **Explorar los Notebooks interactivos (ya ejecutados con salidas y gráficos integrados)**:
   - `src/modelo-clusters.ipynb`: Transformación CLR/ILR, PCA, validación multi-algoritmo ($K=2..6$), Bootstrap Jaccard y perfiles sectoriales.
   - `src/modelo-econometrico.ipynb`: Pruebas de volatilidad estructural (Levene / Fligner-Killeen) y progresión econométrica de Efectos Fijos (`M1` a `M6`).

4. **Compilar la presentación Beamer PDF**:

   ```bash
   cd presentacion
   pdflatex -interaction=nonstopmode presentacion-clusters-departamentales.tex
   pdflatex -interaction=nonstopmode presentacion-clusters-departamentales.tex
   ```
