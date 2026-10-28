"""Lee cuadros INEI por encabezados; --comprobar verifica la base sin escribirla."""
import argparse
import re
import numpy as np
import pandas as pd
from analisis import ROOT, cargar_config


def cuadro(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    headings = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Actividades")]
    if len(headings) != 1:
        raise ValueError(f"Encabezado ambiguo: {path.name}/{sheet}")
    header = headings[0]
    department = next(str(v).split(":")[0].strip() for v in raw.iloc[:header, 0] if ": Valor Agregado Bruto" in str(v))
    # Algunas hojas traen errores de rótulo (p. ej. «VaMad Agregado Bruto»).
    total = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.fullmatch(r"Va.* Agregado Bruto")]
    if len(total) != 1:
        raise ValueError(f"Total VAB ambiguo: {path.name}/{sheet}")
    years = {}
    for col, value in raw.loc[header].items():
        match = re.fullmatch(r"\s*((?:19|20)\d{2})(?:\.0|\s*[PEpe]?/?)*\s*", str(value))
        if match:
            years[col] = int(match[1])
    if not years:
        raise ValueError(f"No hay años: {path.name}/{sheet}")
    data = raw.loc[header+1:total[0], [0, *years]].dropna(subset=[0]).copy()
    data[0] = data[0].astype(str).str.strip()
    data[0] = data[0].str.replace(r"Administración Públ.* y Defensa", "Administración Pública y Defensa", regex=True)
    data[0] = data[0].replace({"Otros Servicios": "Otros Actividades/1"})
    data[0] = data[0].str.replace(r"Va.* Agregado Bruto", "Valor Agregado Bruto", regex=True)
    data = data.set_index(0).rename(columns=years).T.apply(pd.to_numeric, errors="raise")
    data.index.name = "anio"
    data["departamento"] = department
    return data.reset_index()


def extraer():
    files = sorted((ROOT / "data/raw").glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No hay libros INEI en data/raw")
    frames = []
    for path in files:
        shares, levels = cuadro(path, "cuadro2"), cuadro(path, "cuadro1")
        shares = shares.drop(columns="Valor Agregado Bruto")
        levels = levels[["departamento", "anio", "Valor Agregado Bruto"]].rename(columns={"Valor Agregado Bruto": "VAB_total"}).sort_values("anio")
        levels["crecimiento_vab"] = levels.VAB_total.pct_change(fill_method=None).mul(100).round(2)
        frames.append(shares.merge(levels, on=["departamento", "anio"], validate="one_to_one").dropna(subset=["crecimiento_vab"]))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["departamento", "anio"]).any():
        raise ValueError("Fuentes con unidades duplicadas")
    return result


def comprobar(result, target):
    base = pd.read_excel(target).rename(columns={"año": "anio", "PIB_total": "VAB_total", "var_pib": "crecimiento_vab"})
    if "Otros Servicios" in base:
        base["Otros Actividades/1"] = base["Otros Actividades/1"].fillna(0) + base.pop("Otros Servicios").fillna(0)
    keys = ["departamento", "anio"]
    left = result.set_index(keys).sort_index().sort_index(axis=1)
    right = base.set_index(keys).sort_index().sort_index(axis=1)
    pd.testing.assert_index_equal(left.index, right.index)
    pd.testing.assert_index_equal(left.columns, right.columns, check_names=False)
    np.testing.assert_allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), atol=1e-8, rtol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comprobar", action="store_true")
    args = parser.parse_args()
    result = extraer()
    target = ROOT / cargar_config()["datos"]["entrada"]
    if args.comprobar:
        comprobar(result, target)
        print(f"Fuentes y base coinciden: {len(result)} filas.")
    else:
        result.to_excel(target, index=False)
        print(f"Base armonizada guardada: {target}")


if __name__ == "__main__":
    main()
