# uv run --with pillow,numpy,fpdf2 python3 generate_fixtures.py
#
# Generates deterministic synthetic OCR eval fixtures.
# All content is fictional ("Romashka Analytics / ООО «Ромашка Аналитика»").
# No real entities, PII, or third-party copyrighted material.
# License: same as repository root LICENSE (MIT).

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
OUT = HERE / "files"
OUT.mkdir(parents=True, exist_ok=True)

REG = HERE / "assets" / "DejaVuSans.ttf"
BLD = HERE / "assets" / "DejaVuSans-Bold.ttf"

W, H = 1240, 1754  # A4 @ ~150 dpi


def f(sz, bold=False):
    return ImageFont.truetype(str(BLD if bold else REG), sz)


def page1() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Title block
    d.text((70, 70),  "ООО «Ромашка Аналитика» — пресс-релиз по итогам 2025 года",
           font=f(28, True), fill="black")
    d.text((70, 116), "Romashka Analytics — FY2025 results",
           font=f(22), fill="black")
    d.line([(70, 154), (1170, 154)], fill="black", width=2)

    # Key results
    d.text((70, 174), "Ключевые финансовые результаты",
           font=f(24, True), fill="black")
    d.text((70, 222), "• Выручка в 2025 г. составила 2 500 млн руб.",
           font=f(22), fill="black")
    d.text((70, 260), "• EBITDA в 2025 г. составила 415 млн руб., "
                      "рентабельность по EBITDA 16.6%.",
           font=f(22), fill="black")
    d.text((70, 298), "• Чистый долг на конец 2025 г.: 310 млн руб.",
           font=f(22), fill="black")

    # Financial table
    d.text((70, 360), "Финансовые результаты за 2025 и 2024 гг. | тыс. руб.",
           font=f(20, True), fill="black")

    rows = [
        ("Показатель",           "2025",       "2024",       "Изм. %"),
        ("Выручка, тыс. руб.",   "2 500 000",  "2 100 000",  "+19.0%"),
        ("EBITDA, тыс. руб.",    "415 000",    "380 000",    "+9.2%"),
        ("Рентабельность EBITDA", "16.6%",     "18.1%",      "-1.5 п.п"),
        ("Чистый долг, тыс. руб.", "310 000",  "290 000",    "+6.9%"),
    ]
    x0, y0, cols, rh = 70, 400, [0, 490, 730, 980], 52
    for r, row in enumerate(rows):
        y = y0 + r * rh
        fill = (240, 240, 240) if r % 2 == 0 else "white"
        d.rectangle([x0, y, x0 + 1100, y + rh], outline="black", fill=fill)
        for c, cell in enumerate(row):
            d.text((x0 + 12 + cols[c], y + 14), cell,
                   font=f(20, r == 0), fill="black")

    # Footer note
    d.text((70, 690),
           "Источник: консолидированная финансовая отчётность Группы.",
           font=f(18), fill="#555555")
    return img


def page2() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text((70, 70), "Программа сокращения расходов",
           font=f(24, True), fill="black")
    d.text((70, 120), "Cost reduction programme — selected line items",
           font=f(20), fill="black")
    d.line([(70, 158), (1170, 158)], fill="black", width=2)

    items = [
        ("Расходы на оплату труда",         "245 559", "373 847", "-34.3%"),
        ("Профессиональные услуги",          "48 628",  "66 285",  "-26.6%"),
        ("Расходы на аренду",               "2 189",   "8 443",   "-74.1%"),
        ("Прочие операционные расходы",     "17 876",  "23 653",  "-24.4%"),
    ]
    cols = [0, 560, 760, 960]
    hdrs = ("Статья расходов", "2025", "2024", "Изм. %")
    x0, y0, rh = 70, 200, 52
    d.rectangle([x0, y0, x0 + 1100, y0 + rh], outline="black", fill=(220, 220, 220))
    for c, h in enumerate(hdrs):
        d.text((x0 + 12 + cols[c], y0 + 14), h, font=f(20, True), fill="black")
    for r, row in enumerate(items):
        y = y0 + (r + 1) * rh
        d.rectangle([x0, y, x0 + 1100, y + rh], outline="black")
        for c, cell in enumerate(row):
            d.text((x0 + 12 + cols[c], y + 14), cell, font=f(20), fill="black")
    return img


def slide() -> Image.Image:
    img = Image.new("RGB", (1600, 900), (245, 130, 30))
    d = ImageDraw.Draw(img)
    d.text((90, 280), "Romashka Analytics", font=f(96, True), fill="white")
    d.text((90, 420), "Итоги 2025 г.",      font=f(64),       fill="white")
    d.text((90, 560), "Ключевая ставка: 21%", font=f(44),     fill="white")
    d.text((90, 800), "30 апреля 2026 г.",  font=f(30),       fill=(255, 220, 180))
    return img


def skew(img: Image.Image, deg: float = 4.0, sigma: float = 14.0,
         max_width: int = 620) -> Image.Image:
    # Downsample to keep skewed scan small enough for git (< 500 KB target)
    if img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, int(img.height * scale)), Image.LANCZOS)
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    noise = np.random.default_rng(7).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return (Image.fromarray(arr)
            .rotate(deg, expand=True, fillcolor=(255, 255, 255),
                    resample=Image.BICUBIC))


def images_to_pdf(pages: list[Image.Image], path: Path) -> None:
    """Save a list of RGB PIL images as an image-only PDF (no text layer).

    Uses fpdf2 to embed each page as a full-page PNG image inside the PDF.
    This guarantees pdffonts reports 0 fonts and probe.sh returns needs_ocr=true.
    """
    from fpdf import FPDF
    import tempfile, os

    tmp_files = []
    try:
        pdf = FPDF(unit="pt")
        for img in pages:
            # Save page to a temp PNG
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_files.append(tmp.name)
            img.save(tmp.name, format="PNG")
            tmp.close()

            w_pt, h_pt = img.width * 0.75, img.height * 0.75  # px → pt (72dpi base)
            pdf.add_page(format=(w_pt, h_pt))
            pdf.image(tmp.name, x=0, y=0, w=w_pt, h=h_pt)

        pdf.output(str(path))
    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


if __name__ == "__main__":
    p1 = page1()
    p2 = page2()

    # Image-only PDF: pages embedded as PNG images, no fonts, no text layer.
    # probe.sh will return needs_ocr:true ("no fonts found — likely fully image-based PDF").
    images_to_pdf([p1, p2], OUT / "press_release.pdf")
    print("wrote press_release.pdf")

    p1.save(str(OUT / "press_release.png"))
    print("wrote press_release.png")

    slide().save(str(OUT / "slide.png"))
    print("wrote slide.png")

    skew(p1).save(str(OUT / "skewed_scan.jpg"), quality=82, optimize=True)
    print("wrote skewed_scan.jpg")

    print("\nAll fixtures written to", OUT)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name:30s}  {p.stat().st_size:>9,} bytes")
