# Estructura productiva y recuperación departamental en el Perú

Investigación del **Círculo de Ciencia de Datos y Econometría (CCDE)** sobre composición productiva, dispersión y recuperación posterior a 2020. Los libros INEI utilizados miden **VAB real a precios constantes de 2007**. La denominación PBI de las variables antiguas se corrige en las salidas finales.

## Entregables

- `articulo/main.tex` y `articulo/main.pdf`: documento de trabajo con metodología, resultados, discusión, referencias y anexos.
- `presentacion/presentacion-clusters-departamentales.tex`, `.pdf` y `.pptx`: presentación final con la portada institucional original.
- `figures/`: siete figuras compartidas, en PDF vectorial y PNG.
- `src/modelo-clusters.ipynb` y `src/modelo-econometrico.ipynb`: cuadernos ejecutados que consultan las mismas salidas.
- `docs/revision-final.md`: revisión visual y correcciones metodológicas.

PowerPoint reproduce las páginas finales del PDF mediante imágenes y conserva texto en notas y descripciones. **El contenido y la composición se editan en LaTeX**; los elementos de PowerPoint no son objetos editables individuales.

## Alcance y resultados

La base fuente contiene 428 filas de 2008–2025. Los modelos de crecimiento usan 2008–2023 (384 observaciones; 383 en M4). Los modelos de índices usan 2017–2023 (168 observaciones), base 2019=100. El promedio productivo previo corresponde a 2015–2019 y contiene **11 actividades armonizadas**, sin ceros observados.

Ward K=4 identifica cuatro perfiles descriptivos. Los criterios no coinciden: K=2 tiene mayor silueta y la regla Gap de un error estándar selecciona K=1. El corte Ward K=2 separa 3 y 21 departamentos; **no equivale** a la agregación económica de 10 mineros y 14 diversificados.

El índice medio de enclave termina 2023 en 97.6. M5 estima un cambio relativo de −10.92 puntos para 2021–2023 frente a diversificado/agro y 2017–2019 (IC 95%: [−18.05, −3.79]; p=0.0029). Es asociación y persistencia observada; no pérdida causal exclusiva del COVID ni demostración de daño permanente. El artículo desarrolla los límites de clasificación e inferencia.

## Reproducción

Requiere Python 3.11 o posterior (verificado con 3.12) y LaTeX con `pdflatex` en PATH (MiKTeX o TeX Live). Paquetes TeX: Beamer, babel español, Latin Modern, microtype, graphicx, booktabs, amsmath, amssymb, array, geometry, hyperref, fancyhdr, placeins y tcolorbox/TikZ.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/consolidar.py --comprobar
python src/generar_analisis_completo.py
python src/compilar.py
python src/verificar.py
```

`consolidar.py --comprobar` contrasta fuentes y base sin escribirla. Sin ese argumento reconstruye la entrada con nombres armonizados y los mismos datos numéricos. El pipeline acepta ambos esquemas.

`python src/generar_analisis_completo.py --config ruta/al/analisis.toml` permite una configuración alternativa. Los parámetros son supuestos; las cifras se calculan. Los textos editoriales describen el diseño actual: cambiar ventanas o modelos exige revisar también su interpretación.

Para auditar una configuración alternativa, utiliza `python src/verificar.py --config ruta/al/analisis.toml`. La tipología editorial requiere PCA2 y evaluar K=2 y K=4. La auditoría rechaza configuraciones, fuentes o salidas procesadas que hayan cambiado desde el cálculo registrado.

Las pruebas de regresión se ejecutan con `python -m unittest discover -s tests -v`. Comprueban cobertura departamental, entradas inválidas, conservación de las estimaciones y detección de archivos desactualizados, sin escribir los entregables.

Los cuadernos pueden ejecutarse en Jupyter usando este entorno. Leen panel, estimaciones, métricas y figuras; no mantienen otra implementación de los modelos.

## Inspección visual

```powershell
python src/inspeccionar.py presentacion/presentacion-clusters-departamentales.pdf --salida .qa/presentacion
python src/inspeccionar.py articulo/main.pdf --salida .qa/articulo
```

Se generan páginas y hojas de contacto para inspección humana. La compilación rechaza referencias indefinidas, glifos ausentes y desbordamientos. La auditoría comprueba fuentes, crecimiento, cierre composicional, isometría CLR/ILR, muestras, intervalos, referencias y concordancia PDF/PowerPoint. Estas comprobaciones complementan la revisión visual.

## Organización

```text
assets/                 Estilo Beamer y logotipo positivo CCDE
articulo/               Fuente LaTeX modular y PDF del artículo
config/analisis.toml    Supuestos y ventanas
 data/raw/              24 libros INEI originales conservados en Git
 data/processed/        Base original y salidas finales
figures/                Única copia de figuras compartidas
presentacion/           Beamer, PDF y PowerPoint
resultados/             Cifras y tablas LaTeX generadas
src/                    ETL, análisis, gráficos y verificaciones
```

`data/processed/resultados.json` registra SHA-256 de entrada, fuentes y salidas procesadas, cobertura, configuración y versiones. Los perfiles son etiquetas relativas; diversificado/agro incluye Lima sin implicar predominio agrario en todos sus miembros. Los originales se conservan para auditoría.
