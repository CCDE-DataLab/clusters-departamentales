"""Regenera análisis, figuras, tablas y resultados LaTeX desde una configuración."""
import argparse
from threadpoolctl import threadpool_limits
from analisis import cargar_config, ejecutar, guardar
from figuras import generar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Ruta a un TOML alternativo")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        study = ejecutar(cargar_config(args.config))
        guardar(study)
        generar(study)
        from publicar import generar_resultados
        generar_resultados(study)
    print("Análisis, figuras y resultados editoriales actualizados.")


if __name__ == "__main__":
    main()
