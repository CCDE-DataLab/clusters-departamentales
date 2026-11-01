"""Compila el artículo y la presentación Beamer en PDF verificando cero desbordamientos."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "articulo/main.tex", ROOT / "presentacion/presentacion-clusters-departamentales.tex"]


def compilar(path):
    engine = shutil.which("pdflatex")
    if not engine:
        raise RuntimeError("Instala una distribución LaTeX con pdflatex (MiKTeX o TeX Live).")
    for _ in range(3):
        result = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", path.name], cwd=path.parent, capture_output=True)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode("utf-8", errors="replace")[-7000:])
    log = path.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    problems = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line.lower() or "Missing character:" in line]
    if problems:
        raise RuntimeError(f"Revisar {path}:\n" + "\n".join(problems))
    print(f"Compilación correcta: {path.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo", choices=["articulo", "presentacion"])
    args = parser.parse_args()
    for path in TARGETS:
        if not args.solo or path.parent.name == args.solo:
            compilar(path)


if __name__ == "__main__":
    main()
