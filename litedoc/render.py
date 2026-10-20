"""
Rendering helpers used while preparing the benchmarks (paper Sec. 4.3 "Inference Pipelines"):

  * PDF -> PNG with PyMuPDF (DocILE @ 150 DPI, Kleister Charity @ 300 DPI);
  * WikiTableQuestions HTML tables -> PNG with headless Chrome via Selenium (Arial, cell padding,
    alternating row colors), with a pure-PIL fallback renderer for machines without Chrome.

Rendered files are cached: an existing output file is never re-rendered.
"""

import csv
import html
import os
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


def load_image(path: str) -> Image.Image:
    """Open an image as RGB; multi-page files (TIFF) use their first page."""
    image = Image.open(path)
    if getattr(image, "n_frames", 1) > 1:
        image.seek(0)
    return image.convert("RGB")


def render_pdf(pdf_path: str, out_dir: str, dpi: int = 150, max_pages: Optional[int] = None) -> List[str]:
    """Render the pages of a PDF to PNG files `<out_dir>/<stem>_p<k>.png`."""
    import fitz  # PyMuPDF

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    paths = []
    with fitz.open(pdf_path) as doc:
        num_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        zoom = dpi / 72.0
        for page_idx in range(num_pages):
            out_path = os.path.join(out_dir, f"{stem}_p{page_idx}.png")
            if not os.path.isfile(out_path):
                pix = doc.load_page(page_idx).get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                pix.save(out_path)
            paths.append(out_path)
    return paths


TABLE_CSS = """
body { margin: 0; padding: 16px; background: #ffffff; font-family: Arial, Helvetica, sans-serif; }
table { border-collapse: collapse; font-size: 15px; color: #111111; }
th, td { border: 1px solid #9a9a9a; padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: #dde4ee; font-weight: bold; }
tr:nth-child(even) td { background: #f3f5f8; }
"""


def table_rows_from_csv(csv_path: str) -> List[List[str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f)]


def table_rows_to_html(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "<table></table>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows[1:]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_table_selenium(table_html: str, out_path: str, window_width: int = 1280) -> str:
    """Render an HTML table with headless Chrome and save a full-size screenshot of the table."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    options = webdriver.ChromeOptions()
    for arg in ("--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                f"--window-size={window_width},800"):
        options.add_argument(arg)
    driver = webdriver.Chrome(options=options)
    try:
        page = f"<html><head><meta charset='utf-8'><style>{TABLE_CSS}</style></head><body>{table_html}</body></html>"
        driver.get("data:text/html;charset=utf-8," + _quote(page))
        table = driver.find_element(By.TAG_NAME, "table")
        width = driver.execute_script("return document.body.scrollWidth")
        height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(max(window_width, width + 32), height + 32)
        table.screenshot(out_path)
    finally:
        driver.quit()
    return out_path


def _quote(text: str) -> str:
    from urllib.parse import quote
    return quote(text)


def _font(size: int, bold: bool = False):
    names = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_table_pil(
    rows: Sequence[Sequence[str]],
    out_path: str,
    font_size: int = 15,
    padding: int = 8,
    max_col_chars: int = 40,
) -> str:
    """Dependency-free table renderer with the same visual conventions as the Selenium path."""
    font, bold = _font(font_size), _font(font_size, bold=True)
    rows = [[str(c)[:max_col_chars] for c in row] for row in rows] or [[""]]
    num_cols = max(len(r) for r in rows)
    rows = [list(r) + [""] * (num_cols - len(r)) for r in rows]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def text_size(text, fnt):
        left, top, right, bottom = probe.textbbox((0, 0), text, font=fnt)
        return right - left, bottom - top

    col_widths = [0] * num_cols
    row_height = 0
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            w, h = text_size(cell or " ", bold if r_idx == 0 else font)
            col_widths[c_idx] = max(col_widths[c_idx], w + 2 * padding)
            row_height = max(row_height, h + 2 * padding)

    margin = 16
    width = sum(col_widths) + 2 * margin + 1
    height = row_height * len(rows) + 2 * margin + 1
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for r_idx, row in enumerate(rows):
        x = margin
        fill = (221, 228, 238) if r_idx == 0 else ((243, 245, 248) if r_idx % 2 == 0 else (255, 255, 255))
        for c_idx, cell in enumerate(row):
            draw.rectangle([x, y, x + col_widths[c_idx], y + row_height], fill=fill, outline=(154, 154, 154))
            draw.text((x + padding, y + padding), cell, fill=(17, 17, 17), font=bold if r_idx == 0 else font)
            x += col_widths[c_idx]
        y += row_height
    image.save(out_path)
    return out_path


def render_table(rows: Sequence[Sequence[str]], out_path: str, renderer: str = "auto") -> str:
    """Render a table to PNG with Selenium (renderer='selenium'), PIL ('pil') or Selenium-if-available ('auto')."""
    if os.path.isfile(out_path):
        return out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if renderer in ("selenium", "auto"):
        try:
            return render_table_selenium(table_rows_to_html(rows), out_path)
        except Exception:  # noqa: BLE001 - fall back to PIL when Chrome/selenium is unavailable
            if renderer == "selenium":
                raise
    return render_table_pil(rows, out_path)
