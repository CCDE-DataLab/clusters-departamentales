# Clusters departamentales

Análisis de la estructura productiva de los departamentos y su relación con la variación del PIB.

> **Estado:** proyecto en fase inicial. La metodología, las variables y los resultados todavía pueden cambiar.

## Descripción

El proyecto reúne un flujo de análisis en tres etapas:

1. Consolidación de archivos Excel departamentales en una base procesada.
2. Identificación de grupos de departamentos mediante transformación CLR, PCA y clustering jerárquico.
3. Estimación de un modelo de datos de panel con efectos fijos para estudiar la variación del PIB y el periodo COVID-19 según el cluster productivo.

## Estructura

```text
.
├── data/
│   ├── processed/       # Bases procesadas y resultados intermedios
│   └── raw/             # Datos originales; no se publica
├── logs/                # Registros de trabajo
├── src/
│   ├── generador-df.py
│   ├── modelo-clusters.ipynb
│   └── modelo-econometrico.ipynb
├── .gitignore
└── requirements.txt
```

## Instalación

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Colocar los archivos fuente Excel en `data/raw/`.
2. Generar la base consolidada:

   ```bash
   python src/generador-df.py
   ```

3. Abrir `src/modelo-clusters.ipynb` para ejecutar el análisis de componentes principales y clustering.
4. Abrir `src/modelo-econometrico.ipynb` para estimar el modelo de panel.

Los notebooks utilizan las bases ubicadas en `data/processed/`, entre ellas `pbi_departamental_consolidado.xlsx` y `panel_econometrico_base.xlsx`.

## Datos

Los archivos de `data/raw/` están excluidos mediante `.gitignore` porque contienen los datos originales. Las bases procesadas que ya forman parte del proyecto sí pueden versionarse, siempre que su distribución sea compatible con las fuentes originales.
