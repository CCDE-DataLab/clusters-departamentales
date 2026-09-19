import pandas as pd 
# pyrefly: ignore [missing-import]
import numpy as np
import os
from pathlib import Path
import openpyxl
import re


DIRECTORIO_BASE = Path(__file__).resolve().parent.parent
PATH_RAW = DIRECTORIO_BASE / "data/raw/"
PATH_PROCESSED = DIRECTORIO_BASE / "data/processed/"

lista_dfs = []

#Lectura cuadro2 para obtener los shares del PIB de cada departamento,sector y año

for file_path in PATH_RAW.glob("*.xlsx"):
    file = file_path.name
    if file.endswith(".xlsx"):
        df = pd.read_excel(file_path, skiprows=6,sheet_name="cuadro2")
        df = df[:12]

        df.dropna(subset = [df.columns[0]],inplace = True) #Excluimos columnas de nulos del df
        df.reset_index(drop=True, inplace=True)


        #Transponemos el df para que los años sean columnas y las actividades filas
        col_actividades = df.columns[0]
        df_t = df.set_index(col_actividades).T

        df_t = df_t.reset_index().rename(columns={"index":"año"})
        # Unifica todas las variaciones raras a un solo nombre correcto
        df_t.columns = df_t.columns.str.replace(r'Administración Públ.* y Defensa', 'Administración Pública y Defensa', regex=True)
        celda_departamento = pd.read_excel(file_path,sheet_name = "cuadro2",header=None,nrows=2,usecols="A")
        valor_departamento = celda_departamento.iloc[1, 0]
        df_t["departamento"] = str(valor_departamento).split(":")[0].strip()
        lista_dfs.append(df_t)

df_final = pd.concat(lista_dfs,ignore_index=True)
df_final["año"] = df_final["año"].astype(str).str.replace(r'\D', '', regex=True)
df_final["año"] = df_final["año"].astype(int)

columnas_ordenadas = ["departamento","año"] + [col for col in df_final.columns if col not in ["departamento","año"]]
df_final = df_final[columnas_ordenadas]

#Lectura cuadro1 para obtener los valores del PIB total de cada departamento y año
lista_dfs_cuadro1 = []

for file_path in PATH_RAW.glob("*.xlsx"):
    file = file_path.name
    if file.endswith(".xlsx"):
        df_cuadro1 = pd.read_excel(file_path, skiprows=6,sheet_name= "cuadro1")
        df_cuadro1.dropna(subset = [df_cuadro1.columns[0]],inplace = True) #Excluimos columnas de nulos del df
        df_cuadro1.reset_index(drop=True, inplace=True)

        indice_limite = df_cuadro1[df_cuadro1[df_cuadro1.columns[0]].str.contains("Fuente",na=False)].index[0]
        df_cuadro1 = df_cuadro1.iloc[:indice_limite]
        col_actividades_cuadro1 = df_cuadro1.columns[0]
        df_cuadro1_t = df_cuadro1.set_index(col_actividades_cuadro1).T

        df_cuadro1_t = df_cuadro1_t.reset_index().rename(columns={"index":"año"})

        columna_pib = df_cuadro1_t.columns[-1]
        df_cuadro1_t = df_cuadro1_t.rename(columns={columna_pib: "PIB_total"})
        df_cuadro1_t["PIB_total"] = pd.to_numeric(df_cuadro1_t["PIB_total"], errors = "coerce")

        df_cuadro1_t["var_pib"] = df_cuadro1_t["PIB_total"].pct_change() * 100
        df_cuadro1_t["var_pib"] = df_cuadro1_t["var_pib"].round(2)
        df_cuadro1_t = df_cuadro1_t.dropna(subset=["var_pib"])

        celda_departamento_cuadro1 = pd.read_excel(file_path,sheet_name = "cuadro1",header=None,nrows=2,usecols="A")
        valor_departamento_cuadro1 = celda_departamento_cuadro1.iloc[1, 0]
        df_cuadro1_t["departamento"] = str(valor_departamento_cuadro1).split(":")[0].strip()

        df_cuadro1_t = df_cuadro1_t[["departamento", "año", "PIB_total", "var_pib"]]

        lista_dfs_cuadro1.append(df_cuadro1_t)

df_final_cuadro1 = pd.concat(lista_dfs_cuadro1,ignore_index=True)
df_final_cuadro1["año"] = df_final_cuadro1["año"].astype(str).str.replace(r'\D', '', regex=True)
df_final_cuadro1["año"] = df_final_cuadro1["año"].astype(int)


df_final = pd.merge(df_final, df_final_cuadro1, on = ["departamento", "año"], how = "inner")


print("DataFrame final generado con éxito.")
print(f"Dimensiones del DataFrame final: {df_final.shape}")
print(f"Columnas del DataFrame final: {df_final.columns.tolist()}")



archivo_salida = PATH_PROCESSED / "pbi_departamental_consolidado.xlsx"
df_final.to_excel(archivo_salida, index=False)
print(f"Archivo de salida guardado en: {archivo_salida}")

