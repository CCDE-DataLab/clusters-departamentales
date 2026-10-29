"""Regresiones de cobertura, estimaciones y trazabilidad, sin modificar entregables."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analisis import ROOT, cargar_config, ejecutar
from composicional import reemplazar_ceros
from verificar import verificar_trazabilidad


class ValidacionDatos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cargar_config()
        # Solo se reducen las réplicas diagnósticas; las estimaciones son las mismas.
        cls.config["clustering"].update(gap_replicas=2, estabilidad_replicas=2)
        cls.base = pd.read_excel(ROOT / cls.config["datos"]["entrada"])

    def ejecutar_con(self, base, config=None):
        with patch("analisis.pd.read_excel", return_value=base.copy()), threadpool_limits(limits=1):
            return ejecutar(config or self.config)

    def test_departamento_sin_toda_la_ventana_previa(self):
        missing = self.base.departamento.eq("Lima") & self.base["año"].between(2015, 2019)
        with self.assertRaisesRegex(ValueError, "Ventana pre-COVID incompleta"):
            self.ejecutar_con(self.base.loc[~missing])

    def test_participacion_negativa_aunque_la_fila_sume_cien(self):
        base = self.base.copy()
        difference = base.loc[0, "Manufactura"] + 1
        base.loc[0, "Manufactura"] = -1
        base.loc[0, "Comercio"] += difference
        with self.assertRaisesRegex(ValueError, "no negativas"):
            self.ejecutar_con(base)

    def test_vab_invalido(self):
        for value in [0, -1, np.inf, np.nan]:
            with self.subTest(value=value):
                base = self.base.copy()
                base["PIB_total"] = base["PIB_total"].astype(float)
                base.loc[0, "PIB_total"] = value
                with self.assertRaisesRegex(ValueError, "VAB debe ser finito y positivo"):
                    self.ejecutar_con(base)

    def test_base_del_indice_ausente(self):
        config = copy.deepcopy(self.config)
        config["muestra"]["base"] = 2007
        with self.assertRaisesRegex(ValueError, "Falta un VAB base válido"):
            self.ejecutar_con(self.base, config)

    def test_exclusion_fuera_de_la_muestra(self):
        config = copy.deepcopy(self.config)
        config["econometria"]["exclusion_anio"] = 2007
        with self.assertRaisesRegex(ValueError, "M4 requiere exactamente una observación"):
            self.ejecutar_con(self.base, config)

    def test_se_conservan_muestras_y_estimaciones(self):
        expected = pd.read_excel(ROOT / self.config["datos"]["salida"] / "resumen_modelos_econometricos.xlsx")
        study = self.ejecutar_con(self.base)
        self.assertEqual((len(study.shares), len(study.tasas), len(study.niveles)), (24, 384, 168))
        self.assertEqual(study.modelos["M4"].nobs, 383)
        keys = ["Modelo", "Parametro"]
        columns = ["Coeficiente", "Error_Estandar", "P_valor", "IC_inferior", "IC_superior"]
        actual = study.resumen.set_index(keys).sort_index()[columns]
        expected = expected.set_index(keys).sort_index()[columns]
        pd.testing.assert_frame_equal(actual, expected, atol=1e-10, rtol=1e-10)


class ValidacionTrazabilidad(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ["data/raw", "data/processed"]:
            (self.root / name).mkdir(parents=True)
        self.config = {"datos": {"entrada": "data/processed/base.xlsx", "salida": "data/processed"}}
        for name in ["data/raw/fuente.xlsx", "data/processed/base.xlsx", "data/processed/panel.xlsx"]:
            (self.root / name).write_bytes(name.encode())
        def fingerprint(name):
            return hashlib.sha256((self.root / name).read_bytes()).hexdigest()
        self.manifest = {"config": copy.deepcopy(self.config),
                         "sha256_entrada": fingerprint("data/processed/base.xlsx"),
                         "sha256_fuentes": {"fuente.xlsx": fingerprint("data/raw/fuente.xlsx")},
                         "sha256_salidas": {"panel.xlsx": fingerprint("data/processed/panel.xlsx")}}
        root_patch = patch("verificar.ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_archivos_y_configuracion_coinciden(self):
        verificar_trazabilidad(self.config, self.manifest)

    def test_configuracion_modificada(self):
        self.config["clustering"] = {"semilla": 99}
        with self.assertRaisesRegex(ValueError, "Configuración modificada"):
            verificar_trazabilidad(self.config, self.manifest)

    def test_entrada_modificada(self):
        (self.root / self.config["datos"]["entrada"]).write_bytes(b"modificada")
        with self.assertRaisesRegex(ValueError, "Entrada modificada"):
            verificar_trazabilidad(self.config, self.manifest)

    def test_fuente_agregada(self):
        (self.root / "data/raw/nueva.xlsx").write_bytes(b"nueva")
        with self.assertRaisesRegex(ValueError, "conjunto de fuentes"):
            verificar_trazabilidad(self.config, self.manifest)

    def test_fuente_eliminada(self):
        (self.root / "data/raw/fuente.xlsx").unlink()
        with self.assertRaisesRegex(ValueError, "conjunto de fuentes"):
            verificar_trazabilidad(self.config, self.manifest)

    def test_salida_modificada(self):
        (self.root / "data/processed/panel.xlsx").write_bytes(b"modificada")
        with self.assertRaisesRegex(ValueError, "Salida modificada o ausente"):
            verificar_trazabilidad(self.config, self.manifest)

    def test_salida_eliminada(self):
        (self.root / "data/processed/panel.xlsx").unlink()
        with self.assertRaisesRegex(ValueError, "Salida modificada o ausente"):
            verificar_trazabilidad(self.config, self.manifest)

    def test_manifiesto_sin_huellas_de_salidas(self):
        del self.manifest["sha256_salidas"]
        with self.assertRaisesRegex(ValueError, "Faltan huellas"):
            verificar_trazabilidad(self.config, self.manifest)


class ValidacionCeros(unittest.TestCase):
    def test_delta_no_finito(self):
        for delta in [np.nan, np.inf]:
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "delta incompatible"):
                reemplazar_ceros(pd.DataFrame([[0, 20, 80]]), delta)


if __name__ == "__main__":
    unittest.main()
