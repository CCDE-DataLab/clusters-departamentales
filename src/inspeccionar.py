"""Genera páginas y hojas de contacto locales para inspección visual humana."""
import argparse
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw


def render(path, output, scale=1.6):
    output.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(path)
    tiles = []
    for index, page in enumerate(doc):
        image_path = output / f"pagina-{index+1:02}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(image_path)
        with Image.open(image_path) as image:
            image.thumbnail((600, 800))
            tile = Image.new("RGB", (620, image.height + 35), "#E2E8F0")
            tile.paste(image, (10, 25))
        ImageDraw.Draw(tile).text((10, 6), f"Página {index+1}", fill="#0A1E40")
        tiles.append(tile)
    # Hojas pequeñas facilitan revisar todas las páginas sin reducirlas en exceso.
    for start in range(0, len(tiles), 6):
        group = tiles[start:start+6]
        height = max(t.height for t in group)
        contact = Image.new("RGB", (620*3, height*((len(group)+2)//3)), "white")
        for i, tile in enumerate(group):
            contact.paste(tile, ((i%3)*620, (i//3)*height))
        contact.save(output / f"contacto-{start//6+1:02}.png")
    print(f"{path.name}: {len(doc)} páginas renderizadas en {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    render(args.pdf, args.salida)
