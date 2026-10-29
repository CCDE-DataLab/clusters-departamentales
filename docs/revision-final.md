# Revisión de la versión final CCDE

La inspección incluyó el PDF original, los gráficos y todas las páginas renderizadas de la presentación y el artículo final. La portada institucional original se conserva por solicitud del usuario.

## Composición y narrativa

- La presentación abre con el resultado de recuperación y sigue con datos, clasificación, volatilidad, distinción entre tasas y niveles, estimaciones y límites.
- Los gráficos ocupan una lámina cada uno. Perfiles sectoriales y dendrograma se separan; la jerarquía pasa al anexo.
- Se aumentan fuentes y se mantienen colores por perfil. Las etiquetas del PCA se distribuyen automáticamente; las trayectorias se identifican directamente al final.
- Se elimina el doble eje Gap/Jaccard. Cada medida tiene su propio panel; Jaccard usa 0–1 y las barras de dispersión parten de cero.
- Se reemplaza el eje partido por un eje lineal con todas las observaciones.
- Los intervalos del forest plot proceden de `conf_int`, con la distribución utilizada por los valores p, y se muestran completos. Las unidades y escalas independientes son explícitas.
- Se retiran anotaciones superpuestas y bandas de error estándar aplicadas selectivamente a dos perfiles. Las medias por grupo son promedios departamentales sin ponderar.

## Correcciones empíricas

| Aspecto | Versión reproducida |
| --- | --- |
| Variable de actividad | VAB real según los cuadros INEI; `PIB_total` era el nombre heredado |
| Base conservada | 428 filas, 2008–2025; dos años finales con 22 departamentos |
| M1–M3 | 384 observaciones, 2008–2023 |
| M4 | 383 observaciones; excluye solamente Apurímac 2016 |
| M5–M6 | 168 observaciones, 2017–2023 |
| Actividades | 11 al armonizar dos nombres del residual |
| Ceros del promedio | 0; el reemplazo no altera los datos actuales |
| Ward K=2 | 3 y 21 departamentos |
| Agregación económica K=4 | 10 mineros y 14 diversificados; distinta de Ward K=2 |
| Gap | Regla de un error estándar selecciona K=1 |
| Estabilidad | Submuestreo sin reemplazo, no bootstrap |
| Dispersión | Razón de desviaciones estándar, no de varianzas |
| Recuperación enclave M5 | Cambio relativo frente a diversificado/agro y 2017–2019; no pérdida causal absoluta desde 2019 |

El artículo documenta sensibilidad a PCA3 e ILR, pocos departamentos enclave, inferencia agrupada con 24 unidades, dependencia en los diagnósticos de varianza y ausencia de identificación causal. No presenta los modelos de niveles como superiores por comparar R² entre dependientes y muestras distintas.

## Depuración y trazabilidad

Se retiran scripts exploratorios reemplazados, paneles intermedios, antiguas secciones Beamer, figuras obsoletas, copias de figuras en la presentación y variantes de logotipo sin referencias. Se conservan todos los Excel fuente y la base consolidada original.

`config/analisis.toml` centraliza supuestos; `analisis.py` estima; `figuras.py` dibuja; `publicar.py` genera cifras y tablas compartidas. Los cuadernos consultan salidas finales. `resultados.json` registra cobertura, versiones y huellas. Las comprobaciones, compilación y renderizado pueden repetirse con los comandos del README.

La exportación PowerPoint contiene una imagen por página del PDF y texto en notas y descripciones. Conserva la composición final; la fuente editable es LaTeX.

## Revisión de ramas · 29 de octubre de 2026

Se revisaron los dos commits de `avance-benjamin` pendientes de integrar en `master`. La comparación de fuentes conserva las 428 filas originales y las estimaciones de los seis modelos.

- Se impide perder silenciosamente un departamento que no tenga ninguna observación en la ventana previa. También se validan participaciones no negativas, VAB finito y positivo, cobertura del año base y presencia de la observación excluida en M4.
- La auditoría compara la configuración vigente, el conjunto completo de fuentes y las huellas de los diez archivos procesados con el manifiesto. Admite `--config` para auditar el TOML utilizado en el análisis.
- Se añaden 15 pruebas de regresión, ejecutables con `python -m unittest discover -s tests -v`, que incluyen alteraciones y ausencias de archivos y conservación de coeficientes, errores estándar, valores p e intervalos.
