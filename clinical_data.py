"""GeneRx clinical knowledge base and offline demo assets.

* ``CLINICAL_SNPS``          - the five pharmacogenomic markers GeneRx keeps from a genome.
* ``generate_synthetic_23andme`` - realistic fake 23andMe raw-data files for demos.
* ``generate_mock_rx_image`` - Pillow-rendered pharmacy-bottle + diet photos for the vision step.
* ``FALLBACK_FDA_DATA``      - verbatim FDA label excerpts used when openFDA is unreachable.

Genotypes are written in gene (coding-strand) notation, e.g. VKORC1 -1639G>A is ``GG`` / ``AA``.
Real 23andMe files use the forward strand, where rs9923231 and rs3892097 appear as the
complement (``CC`` / ``TT``); ``pipeline.parse_23andme_file`` accepts both notations.
"""

from __future__ import annotations

import io
import math
import random
import re
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter, ImageFont

__all__ = [
    "CLINICAL_SNPS",
    "RX_SCENARIOS",
    "FALLBACK_FDA_DATA",
    "FALLBACK_FDA_LABELS",
    "generate_synthetic_23andme",
    "generate_mock_rx_image",
]


# ---------------------------------------------------------------------------
# 1. Clinical SNP panel
# ---------------------------------------------------------------------------
# `phenotype` is the clinically actionable phenotype carried by `risk_genotype`.
# Positions are GRCh37 (build 37, as used by 23andMe) and were checked against dbSNP.

CLINICAL_SNPS: dict[str, dict] = {
    "rs4244285": {
        "gene": "CYP2C19",
        "allele": "*2",
        "variant": "c.681G>A (splicing defect)",
        "chromosome": "10",
        "position": 96541616,
        "normal_genotype": "GG",
        "risk_genotype": "AA",
        "haplotypes": ("*1", "*2"),
        "normal_phenotype": "Normal Metabolizer",
        "carrier_phenotype": "Intermediate Metabolizer",
        "phenotype": "Poor Metabolizer - Clopidogrel activation failure",
        "drugs": ("clopidogrel", "omeprazole"),
        "aliases": ("CYP2C19", "2C19"),
    },
    "rs1057910": {
        "gene": "CYP2C9",
        "allele": "*3",
        "variant": "c.1075A>C (p.Ile359Leu)",
        "chromosome": "10",
        "position": 96741053,
        "normal_genotype": "AA",
        "risk_genotype": "CC",
        "haplotypes": ("*1", "*3"),
        "normal_phenotype": "Normal Metabolizer",
        "carrier_phenotype": "Intermediate Metabolizer",
        "phenotype": "Poor Metabolizer - Warfarin toxicity & severe bleeding risk",
        "drugs": ("warfarin",),
        "aliases": ("CYP2C9", "2C9"),
    },
    "rs9923231": {
        "gene": "VKORC1",
        "allele": "-1639G>A",
        "variant": "c.-1639G>A (promoter)",
        "chromosome": "16",
        "position": 31107689,
        "normal_genotype": "GG",
        "risk_genotype": "AA",
        "haplotypes": ("G", "A"),
        "normal_phenotype": "Normal Sensitivity",
        "carrier_phenotype": "Increased Warfarin Sensitivity (intermediate)",
        "phenotype": "High Warfarin Sensitivity",
        "drugs": ("warfarin",),
        "aliases": ("VKORC1", "VKOR"),
    },
    "rs4149056": {
        "gene": "SLCO1B1",
        "allele": "*5",
        "variant": "c.521T>C (p.Val174Ala)",
        "chromosome": "12",
        "position": 21331549,
        "normal_genotype": "TT",
        "risk_genotype": "CC",
        "haplotypes": ("*1", "*5"),
        "normal_phenotype": "Normal Function",
        "carrier_phenotype": "Intermediate Transporter Function",
        "phenotype": "Decreased Transporter Function - Simvastatin myopathy risk",
        "drugs": ("simvastatin",),
        "aliases": ("SLCO1B1", "OATP1B1"),
    },
    "rs3892097": {
        "gene": "CYP2D6",
        "allele": "*4",
        "variant": "c.506-1G>A (splicing defect)",
        "chromosome": "22",
        "position": 42524947,
        "normal_genotype": "GG",
        "risk_genotype": "AA",
        "haplotypes": ("*1", "*4"),
        "normal_phenotype": "Normal Metabolizer",
        "carrier_phenotype": "Intermediate Metabolizer",
        "phenotype": "Poor Metabolizer - Codeine failure",
        "drugs": ("codeine",),
        "aliases": ("CYP2D6", "2D6"),
    },
}


# ---------------------------------------------------------------------------
# 2. Synthetic 23andMe raw data
# ---------------------------------------------------------------------------

_CHROM_LENGTHS = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260,
    "6": 171115067, "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747,
    "11": 135006516, "12": 133851895, "13": 115169878, "14": 107349540, "15": 102531392,
    "16": 90354753, "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566, "X": 155270560, "MT": 16569,
}
_CHROM_ORDER = {name: i for i, name in enumerate(_CHROM_LENGTHS)}
_CHROM_WEIGHTS = [length if name != "MT" else 6_000_000 for name, length in _CHROM_LENGTHS.items()]
_NUCLEOTIDE_PAIRS = ("AA", "AC", "AG", "AT", "CC", "CG", "CT", "GG", "GT", "TT")
_PAIR_WEIGHTS = (14, 8, 16, 3, 14, 3, 16, 14, 8, 14)
_FILLER_SNPS = 500

_HEADER = """\
# SYNTHETIC DEMO DATA generated by GeneRx - not a real person and not a real 23andMe export.
# The layout below mirrors the 23andMe raw-data download (build 37) so parsers behave identically.
# Note: VKORC1 and CYP2D6 calls are written in gene-strand notation (see clinical_data.py).
#
# This data file generated by 23andMe at: Sat Oct 24 12:00:00 2026
#
# Below is a text version of your data.  Fields are TAB-separated
# Each line corresponds to a single SNP.  For each SNP, we provide its identifier
# (an rsid or an internal id), its location on the reference human genome, and the
# genotype call oriented with respect to the plus strand on the human reference sequence.
# We are using reference human assembly build 37 (also known as Annotation Release 104).
# Note that it is possible that data downloaded at different times may be different due to ongoing
# improvements in our ability to call genotypes. More information about these changes can be found at:
# https://www.23andme.com/you/download/revisionhistory/
#
# More information on reference human assembly build 37 (aka Annotation Release 104):
# http://www.ncbi.nlm.nih.gov/mapview/map_search.cgi?taxid=9606&build=37.1
#
# rsid\tchromosome\tposition\tgenotype
"""


def _filler_genotype(rng: random.Random, chrom: str) -> str:
    if chrom == "MT":
        return rng.choice("ACGT")
    roll = rng.random()
    if roll < 0.012:
        return "--"
    if roll < 0.02:
        return rng.choice(("DD", "DI", "II"))
    return rng.choices(_NUCLEOTIDE_PAIRS, weights=_PAIR_WEIGHTS)[0]


def generate_synthetic_23andme(profile_type: str = "danger") -> str:
    """Return a realistic 23andMe raw-data file (505 SNP rows) as text.

    ``"danger"`` writes the risk genotype for all five clinical rsids, ``"normal"`` the normal
    genotype. The 500 filler rsids are pseudo-random but deterministic per profile.
    """
    profile = str(profile_type).strip().lower()
    if profile not in ("danger", "normal"):
        raise ValueError(f"profile_type must be 'danger' or 'normal', got {profile_type!r}")

    rng = random.Random(f"generx-23andme-{profile}")
    chroms = list(_CHROM_LENGTHS)
    used = set(CLINICAL_SNPS)
    rows: list[tuple[str, int, str, str]] = []
    while len(rows) < _FILLER_SNPS:
        rsid = f"rs{rng.randint(1_000, 150_000_000)}"
        if rsid in used:
            continue
        used.add(rsid)
        chrom = rng.choices(chroms, weights=_CHROM_WEIGHTS)[0]
        rows.append((chrom, rng.randint(10_000, _CHROM_LENGTHS[chrom]), rsid, _filler_genotype(rng, chrom)))

    genotype_key = "risk_genotype" if profile == "danger" else "normal_genotype"
    for rsid, info in CLINICAL_SNPS.items():
        rows.append((info["chromosome"], info["position"], rsid, info[genotype_key]))

    rows.sort(key=lambda r: (_CHROM_ORDER[r[0]], r[1]))
    body = "\n".join(f"{rsid}\t{chrom}\t{pos}\t{genotype}" for chrom, pos, rsid, genotype in rows)
    return _HEADER + body + "\n"


# ---------------------------------------------------------------------------
# 3. Mock prescription + diet photos (Pillow)
# ---------------------------------------------------------------------------

RX_SCENARIOS: dict[str, dict] = {
    "Scenario 1 - Clopidogrel + Grapefruit Juice & Omeprazole": {
        "rx_line": "RX: CLOPIDOGREL (Plavix) 75mg Daily",
        "diet_line": "Dietary Item: Grapefruit Juice & Omeprazole 20mg",
        "generic": "clopidogrel",
        "brand": "Plavix",
        "rx_number": "4829153",
        "qty": "30",
        "refills": "3",
        "directions": "TAKE 1 TABLET BY MOUTH ONCE DAILY",
        "props": ("juice_glass", "grapefruit_half", "omeprazole_box"),
    },
    "Scenario 2 - Warfarin + Ibuprofen": {
        "rx_line": "RX: WARFARIN (Coumadin) 5mg Daily",
        "diet_line": "Dietary Item: Ibuprofen 400mg",
        "generic": "warfarin",
        "brand": "Coumadin",
        "rx_number": "4829207",
        "qty": "30",
        "refills": "5",
        "directions": "TAKE 1 TABLET BY MOUTH ONCE DAILY IN THE EVENING",
        "props": ("ibuprofen_bottle", "tablets"),
    },
    "Scenario 3 - Simvastatin + Grapefruit Juice": {
        "rx_line": "RX: SIMVASTATIN (Zocor) 40mg Daily",
        "diet_line": "Dietary Item: Grapefruit Juice",
        "generic": "simvastatin",
        "brand": "Zocor",
        "rx_number": "4829331",
        "qty": "90",
        "refills": "2",
        "directions": "TAKE 1 TABLET BY MOUTH ONCE DAILY AT BEDTIME",
        "props": ("juice_glass", "grapefruit_half", "grapefruit_whole", "juice_carton"),
    },
}

_W, _H, _SS = 1400, 860, 2
_COUNTER_Y = 640

_FONT_PATHS = {
    False: (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ),
    True: (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
    ),
}


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False):
    for path in _FONT_PATHS[bold]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _shade(color, factor: float):
    return tuple(max(0, min(255, int(c * factor))) for c in color[:3])


class _Scene:
    """Tiny supersampled drawing surface working in 1400x860 design units."""

    def __init__(self):
        self.k = _SS
        self.img = Image.new("RGB", (_W * self.k, _H * self.k), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def _box(self, box):
        return tuple(int(round(v * self.k)) for v in box)

    def _pts(self, pts):
        return [(int(round(x * self.k)), int(round(y * self.k))) for x, y in pts]

    def _w(self, width):
        return max(1, int(round(width * self.k)))

    def rect(self, box, fill=None, outline=None, width=1, radius=0):
        if radius:
            self.d.rounded_rectangle(self._box(box), radius=int(radius * self.k), fill=fill, outline=outline, width=self._w(width))
        else:
            self.d.rectangle(self._box(box), fill=fill, outline=outline, width=self._w(width))

    def ellipse(self, box, fill=None, outline=None, width=1):
        self.d.ellipse(self._box(box), fill=fill, outline=outline, width=self._w(width))

    def polygon(self, pts, fill=None, outline=None):
        self.d.polygon(self._pts(pts), fill=fill, outline=outline)

    def line(self, pts, fill, width=1):
        self.d.line(self._pts(pts), fill=fill, width=self._w(width))

    def font(self, size, bold=False):
        return _font(int(round(size * self.k)), bold)

    def text_width(self, s, size, bold=False):
        return self.font(size, bold).getlength(s) / self.k

    def text(self, xy, s, size, fill=(20, 24, 32), bold=False, anchor="la"):
        self.d.text((xy[0] * self.k, xy[1] * self.k), s, font=self.font(size, bold), fill=fill, anchor=anchor)
        return self.text_width(s, size, bold)

    def wrap(self, s, size, bold, max_w):
        lines, cur = [], ""
        for word in s.split():
            trial = f"{cur} {word}".strip()
            if cur and self.text_width(trial, size, bold) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def fit(self, s, max_size, min_size, bold, max_w):
        size = max_size
        while size > min_size and self.text_width(s, size, bold) > max_w:
            size -= 1
        return size

    @staticmethod
    def _mask(size, radius=0, polygon=None):
        mask = Image.new("L", size, 0)
        md = ImageDraw.Draw(mask)
        if polygon:
            md.polygon(polygon, fill=255)
        elif radius:
            md.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
        else:
            md.rectangle((0, 0, size[0] - 1, size[1] - 1), fill=255)
        return mask

    @staticmethod
    def _ramp(size, horizontal=False):
        ramp = Image.linear_gradient("L")
        if horizontal:
            ramp = ramp.transpose(Image.Transpose.ROTATE_90)
        return ramp.resize(size)

    def gradient(self, box, top, bottom, radius=0, horizontal=False):
        x0, y0, x1, y1 = self._box(box)
        size = (x1 - x0, y1 - y0)
        fill = Image.composite(Image.new("RGB", size, bottom), Image.new("RGB", size, top), self._ramp(size, horizontal))
        self.img.paste(fill, (x0, y0), self._mask(size, int(radius * self.k)))

    def gradient_polygon(self, pts, top, bottom):
        px = self._pts(pts)
        x0, y0 = min(p[0] for p in px), min(p[1] for p in px)
        x1, y1 = max(p[0] for p in px), max(p[1] for p in px)
        size = (x1 - x0, y1 - y0)
        fill = Image.composite(Image.new("RGB", size, bottom), Image.new("RGB", size, top), self._ramp(size))
        local = [(x - x0, y - y0) for x, y in px]
        self.img.paste(fill, (x0, y0), self._mask(size, polygon=local))

    def cylinder(self, box, base, radius=0, light=0.3, depth=1.0):
        x0, y0, x1, y1 = self._box(box)
        w, h = x1 - x0, y1 - y0
        strip = Image.new("RGB", (w, 1))
        px = strip.load()
        for x in range(w):
            t = x / max(1, w - 1)
            f = 0.58 + 0.58 * math.exp(-(((t - light) / 0.26) ** 2)) + 0.10 * math.exp(-(((t - 0.86) / 0.07) ** 2))
            px[x, 0] = _shade(base, 1 - depth * (1 - f))
        self.img.paste(strip.resize((w, h)), (x0, y0), self._mask((w, h), int(radius * self.k)))

    def sphere(self, cx, cy, r, inner, outer):
        steps = 28
        for i in range(steps):
            t = i / (steps - 1)
            rr = r * (1 - 0.92 * t)
            col = tuple(int(outer[c] + (inner[c] - outer[c]) * t) for c in range(3))
            ox, oy = -0.30 * r * t, -0.34 * r * t
            self.d.ellipse(self._box((cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr)), fill=col)

    def shadow(self, box, blur=14, alpha=90, radius=0, ellipse=False, dx=0, dy=0):
        x0, y0, x1, y1 = self._box((box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
        pad = int(blur * self.k * 3)
        w, h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
        mask = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask)
        shape = (pad, pad, pad + x1 - x0, pad + y1 - y0)
        if ellipse:
            md.ellipse(shape, fill=alpha)
        else:
            md.rounded_rectangle(shape, radius=int(radius * self.k), fill=alpha)
        mask = mask.filter(ImageFilter.GaussianBlur(blur * self.k))
        self.img.paste((18, 28, 44), (x0 - pad, y0 - pad), mask)

    def png(self) -> bytes:
        out = self.img.resize((_W, _H), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _paint_background(s: _Scene):
    s.gradient((0, 0, _W, _COUNTER_Y), (236, 241, 247), (210, 220, 232))
    s.gradient((0, _COUNTER_Y, _W, _H), (248, 250, 252), (214, 222, 232))
    s.line([(0, _COUNTER_Y), (_W, _COUNTER_Y)], fill=(172, 183, 199), width=3)
    s.line([(0, _COUNTER_Y + 4), (_W, _COUNTER_Y + 4)], fill=(255, 255, 255, 230), width=2)
    for y in (180, 330):
        s.rect((690, y, 1400, y + 8), fill=(196, 206, 220, 150))
    s.rect((1286, 36, 1362, 112), fill=(22, 163, 100), radius=14)
    s.rect((1315, 52, 1333, 96), fill=(255, 255, 255), radius=3)
    s.rect((1299, 68, 1349, 80), fill=(255, 255, 255), radius=3)


def _paint_barcode(s: _Scene, box, seed: int):
    rng = random.Random(seed)
    x = box[0]
    while x < box[2] - 4:
        w = rng.choice((2, 2, 3, 4, 5))
        s.rect((x, box[1], min(x + w, box[2]), box[3]), fill=(18, 20, 24))
        x += w + rng.choice((2, 3, 4))


def _paint_rx_bottle(s: _Scene, sc: dict):
    x0, x1, y0, y1 = 60, 640, 262, 812
    s.shadow((x0 + 20, y1 - 26, x1 + 30, y1 + 26), blur=18, alpha=110, ellipse=True)
    s.cylinder((x0, y0, x1, y1), (214, 136, 28), radius=48, light=0.28)
    s.rect((x0 + 18, y0 + 10, x1 - 18, y0 + 24), fill=(255, 255, 255, 60), radius=7)
    s.cylinder((x0 + 24, 112, x1 - 24, 282), (248, 250, 253), radius=34, light=0.3, depth=0.42)
    for x in range(x0 + 52, x1 - 46, 15):
        s.line([(x, 128), (x, 266)], fill=(150, 160, 176, 120), width=2)
    s.rect((x0 + 24, 276, x1 - 24, 292), fill=(130, 80, 10), radius=7)

    lx0, ly0, lx1, ly1 = x0 + 34, 322, x1 - 34, 788
    s.shadow((lx0, ly0, lx1, ly1), blur=6, alpha=70, radius=10, dx=3, dy=4)
    s.rect((lx0, ly0, lx1, ly1), fill=(252, 252, 249), outline=(205, 205, 198), width=2, radius=10)
    for i in range(26):
        a = int(70 * (1 - i / 26) ** 2)
        s.rect((lx0 + i, ly0 + 6, lx0 + i + 1, ly1 - 6), fill=(60, 40, 10, a))
        s.rect((lx1 - i - 1, ly0 + 6, lx1 - i, ly1 - 6), fill=(60, 40, 10, a))

    pad = 24
    tx0, tx1 = lx0 + pad, lx1 - pad
    s.rect((lx0 + 2, ly0 + 2, lx1 - 2, ly0 + 78), fill=(12, 112, 120), radius=9)
    s.rect((lx0 + 2, ly0 + 40, lx1 - 2, ly0 + 78), fill=(12, 112, 120))
    s.ellipse((tx0, ly0 + 14, tx0 + 50, ly0 + 64), fill=(255, 255, 255))
    s.rect((tx0 + 21, ly0 + 22, tx0 + 29, ly0 + 56), fill=(12, 112, 120))
    s.rect((tx0 + 8, ly0 + 35, tx0 + 42, ly0 + 43), fill=(12, 112, 120))
    s.text((tx0 + 66, ly0 + 10), "CITYCARE PHARMACY", 31, fill=(255, 255, 255), bold=True)
    s.text((tx0 + 66, ly0 + 48), "214 Harbor St  |  Tel (555) 014-2290", 16, fill=(214, 240, 240))

    y = ly0 + 98
    s.text((tx0, y), f"Rx# {sc['rx_number']}", 20, bold=True)
    s.text((tx1, y), "Dr. A. PATEL", 20, anchor="ra")
    s.text((tx0, y + 32), "PATIENT: DOE, JANE", 22, bold=True)
    s.line([(tx0, y + 70), (tx1, y + 70)], fill=(150, 150, 146), width=2)

    y += 86
    for line in s.wrap(sc["rx_line"], 33, True, tx1 - tx0):
        s.text((tx0, y), line, 33, fill=(8, 10, 14), bold=True)
        y += 42
    y += 6
    for line in s.wrap(sc["directions"], 20, False, tx1 - tx0):
        s.text((tx0, y), line, 20, fill=(40, 44, 52))
        y += 26

    yb = ly1 - 128
    s.text((tx0, yb), f"QTY: {sc['qty']}", 19, bold=True)
    s.text((tx0 + 130, yb), f"REFILLS: {sc['refills']}", 19, bold=True)
    s.text((tx1, yb), "FILLED: 10/24/2026", 19, bold=True, anchor="ra")
    _paint_barcode(s, (tx0, ly1 - 90, tx0 + 190, ly1 - 40), seed=int(sc["rx_number"]))
    s.text((tx1, ly1 - 84), "DISCARD AFTER", 15, fill=(90, 94, 100), anchor="ra")
    s.text((tx1, ly1 - 64), "10/24/2027", 19, bold=True, anchor="ra")


def _paint_juice_glass(s: _Scene, cx, base_y, h=292, top_w=196, bot_w=146):
    top_y = base_y - h

    def half_width(y, w_top, w_bot):
        return (w_top + (w_bot - w_top) * (y - top_y) / h) / 2

    s.shadow((cx - bot_w / 2 - 10, base_y - 14, cx + bot_w / 2 + 70, base_y + 22), blur=12, alpha=100, ellipse=True)
    glass = [(cx - top_w / 2, top_y), (cx + top_w / 2, top_y), (cx + bot_w / 2, base_y), (cx - bot_w / 2, base_y)]
    s.polygon(glass, fill=(255, 255, 255, 70))
    level = top_y + 44
    juice = [(cx - half_width(level, top_w - 8, bot_w - 8), level), (cx + half_width(level, top_w - 8, bot_w - 8), level),
             (cx + bot_w / 2 - 6, base_y - 6), (cx - bot_w / 2 + 6, base_y - 6)]
    s.gradient_polygon(juice, (255, 186, 150), (238, 112, 86))
    s.ellipse((cx - half_width(level, top_w - 8, bot_w - 8), level - 11, cx + half_width(level, top_w - 8, bot_w - 8), level + 11),
              fill=(255, 204, 176), outline=(250, 160, 130), width=2)
    s.polygon([(cx - top_w / 2 + 14, top_y + 20), (cx - top_w / 2 + 30, top_y + 20), (cx - bot_w / 2 + 28, base_y - 24), (cx - bot_w / 2 + 14, base_y - 24)],
              fill=(255, 255, 255, 95))
    s.polygon([(cx + top_w / 2 - 26, top_y + 30), (cx + top_w / 2 - 18, top_y + 30), (cx + bot_w / 2 - 20, base_y - 40), (cx + bot_w / 2 - 28, base_y - 40)],
              fill=(255, 255, 255, 60))
    s.line([glass[0], glass[3]], fill=(255, 255, 255, 190), width=3)
    s.line([glass[1], glass[2]], fill=(255, 255, 255, 190), width=3)
    s.ellipse((cx - top_w / 2, top_y - 11, cx + top_w / 2, top_y + 11), outline=(255, 255, 255, 230), width=3)
    s.ellipse((cx - bot_w / 2 + 2, base_y - 12, cx + bot_w / 2 - 2, base_y + 6), fill=(255, 255, 255, 80), outline=(255, 255, 255, 160), width=2)
    wedge = (cx + top_w / 2 - 70, top_y - 40, cx + top_w / 2 + 14, top_y + 44)
    s.d.pieslice(s._box(wedge), start=200, end=340, fill=(238, 104, 98), outline=(244, 176, 62), width=s._w(6))


def _paint_grapefruit_half(s: _Scene, cx, cy, r):
    s.shadow((cx - r, cy + r - 18, cx + r + 26, cy + r + 18), blur=10, alpha=100, ellipse=True)
    s.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(244, 178, 58), outline=(206, 134, 28), width=3)
    s.ellipse((cx - r * 0.9, cy - r * 0.9, cx + r * 0.9, cy + r * 0.9), fill=(254, 241, 208))
    s.sphere(cx, cy, r * 0.82, (247, 142, 134), (226, 92, 92))
    for i in range(10):
        a = math.radians(i * 36 + 8)
        s.line([(cx + math.cos(a) * r * 0.10, cy + math.sin(a) * r * 0.10), (cx + math.cos(a) * r * 0.82, cy + math.sin(a) * r * 0.82)],
               fill=(254, 238, 206), width=3)
    s.ellipse((cx - r * 0.11, cy - r * 0.11, cx + r * 0.11, cy + r * 0.11), fill=(254, 238, 206))
    s.d.arc(s._box((cx - r * 0.72, cy - r * 0.72, cx + r * 0.72, cy + r * 0.72)), start=200, end=262, fill=(255, 255, 255, 190), width=s._w(5))


def _paint_grapefruit_whole(s: _Scene, cx, cy, r):
    s.shadow((cx - r, cy + r - 16, cx + r + 20, cy + r + 16), blur=10, alpha=100, ellipse=True)
    s.sphere(cx, cy, r, (252, 214, 104), (226, 142, 40))
    rng = random.Random(7)
    for _ in range(46):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.1, 0.9) * r
        x, y = cx + math.cos(a) * d, cy + math.sin(a) * d
        s.ellipse((x - 1.4, y - 1.4, x + 1.4, y + 1.4), fill=(204, 120, 30, 110))
    s.d.arc(s._box((cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7)), start=205, end=255, fill=(255, 255, 255, 170), width=s._w(5))


def _paint_omeprazole_box(s: _Scene, x, y, w=232, h=312, depth=34):
    s.shadow((x - 4, y + h - 18, x + w + depth + 24, y + h + 20), blur=12, alpha=100, ellipse=True)
    s.polygon([(x + w, y), (x + w + depth, y + 18), (x + w + depth, y + h + 14), (x + w, y + h)], fill=(52, 28, 98))
    s.polygon([(x, y), (x + depth, y - 16), (x + w + depth, y + 18), (x + w, y)], fill=(120, 80, 190))
    s.gradient((x, y, x + w, y + h), (112, 66, 182), (66, 34, 122), radius=4)
    s.text((x + w / 2, y + 22), "OMEPRAZOLE", s.fit("OMEPRAZOLE", 34, 20, True, w - 30), fill=(255, 255, 255), bold=True, anchor="ma")
    s.text((x + w / 2, y + 66), "20 mg", 64, fill=(255, 214, 92), bold=True, anchor="ma")
    s.rect((x + 18, y + 156, x + w - 18, y + 238), fill=(255, 255, 255), radius=10)
    s.text((x + w / 2, y + 168), "Delayed-Release", 22, fill=(60, 34, 110), bold=True, anchor="ma")
    s.text((x + w / 2, y + 196), "Acid Reducer Tablets", 20, fill=(60, 34, 110), anchor="ma")
    s.ellipse((x + w / 2 - 38, y + 246, x + w / 2 + 38, y + 306), fill=(255, 150, 40))
    s.text((x + w / 2, y + 252), "14", 28, fill=(255, 255, 255), bold=True, anchor="ma")
    s.text((x + w / 2, y + 282), "TABLETS", 13, fill=(255, 255, 255), bold=True, anchor="ma")


def _paint_ibuprofen_bottle(s: _Scene, x, y, w=262, h=392):
    s.shadow((x - 6, y + h - 22, x + w + 40, y + h + 22), blur=14, alpha=110, ellipse=True)
    s.cylinder((x + 30, y - 8, x + w - 30, y + 62), (36, 96, 190), radius=14, light=0.3)
    for i in range(x + 44, x + w - 40, 11):
        s.line([(i, y - 2), (i, y + 56)], fill=(10, 40, 100, 90), width=2)
    s.cylinder((x, y + 54, x + w, y + h), (250, 251, 253), radius=44, light=0.3, depth=0.4)
    s.rect((x + 4, y + 130, x + w - 4, y + 330), fill=(238, 118, 28), outline=(200, 90, 10), width=2)
    s.text((x + w / 2, y + 144), "IBUPROFEN", s.fit("IBUPROFEN", 40, 22, True, w - 28), fill=(255, 255, 255), bold=True, anchor="ma")
    s.text((x + w / 2, y + 190), "400 mg", 60, fill=(255, 255, 255), bold=True, anchor="ma")
    s.rect((x + 4, y + 268, x + w - 4, y + 330), fill=(255, 255, 255))
    s.text((x + w / 2, y + 278), "Pain Reliever / Fever Reducer", 17, fill=(40, 40, 44), bold=True, anchor="ma")
    s.text((x + w / 2, y + 302), "(NSAID)  |  100 TABLETS", 17, fill=(40, 40, 44), anchor="ma")


def _paint_tablets(s: _Scene, positions):
    for cx, cy, angle in positions:
        s.shadow((cx - 36, cy + 8, cx + 48, cy + 24), blur=5, alpha=90, ellipse=True)
        a = math.radians(angle)
        pts = []
        for t in range(0, 360, 10):
            rad = math.radians(t)
            ex, ey = 34 * math.cos(rad), 19 * math.sin(rad)
            pts.append((cx + ex * math.cos(a) - ey * math.sin(a), cy + ex * math.sin(a) + ey * math.cos(a)))
        s.polygon(pts, fill=(226, 138, 70), outline=(176, 96, 38))
        s.polygon([(px * 0.9 + cx * 0.1 - 3, py * 0.9 + cy * 0.1 - 4) for px, py in pts[:18]], fill=(244, 176, 118, 150))


def _paint_juice_carton(s: _Scene, x, y, w=176, h=318, depth=30):
    s.shadow((x - 4, y + h - 18, x + w + depth + 24, y + h + 20), blur=12, alpha=100, ellipse=True)
    s.polygon([(x + w, y), (x + w + depth, y + 14), (x + w + depth, y + h + 12), (x + w, y + h)], fill=(196, 90, 60))
    s.polygon([(x, y), (x + w / 2, y - 62), (x + w / 2 + depth, y - 50), (x + w + depth, y + 14), (x + w, y)], fill=(255, 238, 226))
    s.polygon([(x, y), (x + w / 2, y - 62), (x + w, y)], fill=(250, 226, 208))
    s.gradient((x, y, x + w, y + h), (255, 170, 130), (232, 98, 74), radius=3)
    s.sphere(x + w / 2, y + 82, 46, (252, 150, 142), (226, 92, 92))
    s.ellipse((x + w / 2 - 46, y + 36, x + w / 2 + 46, y + 128), outline=(244, 178, 62), width=5)
    s.text((x + w / 2, y + 150), "GRAPEFRUIT", s.fit("GRAPEFRUIT", 30, 18, True, w - 20), fill=(255, 255, 255), bold=True, anchor="ma")
    s.text((x + w / 2, y + 188), "JUICE", 40, fill=(255, 255, 255), bold=True, anchor="ma")
    s.rect((x + 18, y + 246, x + w - 18, y + 288), fill=(255, 255, 255, 235), radius=8)
    s.text((x + w / 2, y + 255), "100% Juice", 22, fill=(190, 70, 50), bold=True, anchor="ma")


def _paint_placard(s: _Scene, diet_line: str):
    box = (700, 748, 1356, 842)
    label, _, item = diet_line.partition(":")
    s.shadow(box, blur=9, alpha=80, radius=14, dy=6)
    s.rect(box, fill=(255, 253, 244), outline=(214, 205, 168), width=2, radius=14)
    s.rect((box[0], box[1], box[0] + 18, box[3]), fill=(238, 118, 28), radius=14)
    s.rect((box[0] + 10, box[1], box[0] + 18, box[3]), fill=(238, 118, 28))
    s.text((box[0] + 44, box[1] + 10), f"{label}:", 21, fill=(110, 106, 94), bold=True)
    item = item.strip()
    size = s.fit(item, 38, 22, True, box[2] - box[0] - 80)
    s.text((box[0] + 44, box[1] + 38), item, size, fill=(14, 16, 20), bold=True)


def _scenario_key(name: str) -> str:
    if name in RX_SCENARIOS:
        return name
    text = str(name).strip().lower()
    for key in RX_SCENARIOS:
        if text == key.lower():
            return key
    digit = re.search(r"\b([1-3])\b", text)
    if digit:
        return list(RX_SCENARIOS)[int(digit.group(1)) - 1]
    for key, sc in RX_SCENARIOS.items():
        if sc["generic"] in text or sc["brand"].lower() in text:
            return key
    raise ValueError(f"Unknown scenario {name!r}. Choose one of: {list(RX_SCENARIOS)}")


def generate_mock_rx_image(scenario_name: str) -> bytes:
    """Render a pharmacy bottle label plus a dietary item and return the picture as PNG bytes.

    ``scenario_name`` is a key of ``RX_SCENARIOS``; "Scenario 2", "2" or a drug name also work.
    """
    sc = RX_SCENARIOS[_scenario_key(scenario_name)]
    s = _Scene()
    _paint_background(s)
    _paint_rx_bottle(s, sc)
    props = sc["props"]
    if "juice_glass" in props:
        _paint_juice_glass(s, 790 if "omeprazole_box" in props else 770, 716)
    if "grapefruit_whole" in props:
        _paint_grapefruit_whole(s, 1090, 660, 66)
    if "grapefruit_half" in props:
        _paint_grapefruit_half(s, 980 if "omeprazole_box" not in props else 1000, 640, 88)
    if "omeprazole_box" in props:
        _paint_omeprazole_box(s, 1096, 408)
    if "juice_carton" in props:
        _paint_juice_carton(s, 1176, 392)
    if "ibuprofen_bottle" in props:
        _paint_ibuprofen_bottle(s, 850, 322)
    if "tablets" in props:
        _paint_tablets(s, [(1196, 704, 12), (1262, 690, -20), (1228, 726, 34), (770, 716, -8)])
    _paint_placard(s, sc["diet_line"])
    return s.png()


# ---------------------------------------------------------------------------
# 4. Offline FDA label excerpts
# ---------------------------------------------------------------------------

# Snapshot of real U.S. FDA drug labels captured from the openFDA `drug/label` endpoint on 2026-10-24.
# Every passage is a contiguous, verbatim excerpt (whitespace collapsed) of the named label's field, capped at
# 2,500 characters. The simvastatin and omeprazole labels carry no boxed warning, so that field is left empty
# rather than invented.

FALLBACK_FDA_LABELS = {
    "clopidogrel": {
        "label": "Clopidogrel (clopidogrel bisulfate) - Aurobindo Pharma Limited",
        "set_id": "0078fb3d-3595-4ae1-a059-1d5e81c879cf",
        "effective_time": "2024-05-17",
    },
    "warfarin": {
        "label": "Warfarin Sodium (warfarin sodium) - Teva Pharmaceuticals USA, Inc.",
        "set_id": "0cbce382-9c88-4f58-ae0f-532a841e8f95",
        "effective_time": "2025-06-17",
    },
    "simvastatin": {
        "label": "SIMVASTATIN (simvastatin) - PD-Rx Pharmaceuticals, Inc.",
        "set_id": "00896fff-081d-4553-be8c-1999a8a73dda",
        "effective_time": "2026-02-26",
    },
    "codeine": {
        "label": "Codeine sulfate (codeine sulfate) - Hikma Pharmaceuticals USA Inc.",
        "set_id": "010905f9-3bcb-4b50-9fe8-a3ad0010f14c",
        "effective_time": "2025-12-22",
    },
    "omeprazole": {
        "label": "Omeprazole (omeprazole) - PD-Rx Pharmaceuticals, Inc.",
        "set_id": "0344268f-d8e6-48d5-b1e1-f750ed1d8a52",
        "effective_time": "2026-07-17",
    },
}

FALLBACK_FDA_DATA = {
    "clopidogrel": {
        "boxed_warning": (
            "WARNING: DIMINISHED ANTIPLATELET EFFECT IN PATIENTS WITH TWO LOSS-OF-FUNCTION ALLELES OF THE "
            "CYP2C19 GENE The effectiveness of clopidogrel results from its antiplatelet activity, which "
            "is dependent on its conversion to an active metabolite by the cytochrome P450 (CYP) system, "
            "principally CYP2C19 [see Warnings and Precautions (5.1) , Clinical Pharmacology (12.3) ] . "
            "Clopidogrel at recommended doses forms less of the active metabolite and so has a reduced "
            "effect on platelet activity in patients who are homozygous for nonfunctional alleles of the "
            "CYP2C19 gene, (termed “CYP2C19 poor metabolizers”). Tests are available to identify patients "
            "who are CYP2C19 poor metabolizers [see Clinical Pharmacology (12.5) ] . Consider use of "
            "another platelet P2Y 12 inhibitor in patients identified as CYP2C19 poor metabolizers. "
            "WARNING: DIMINISHED ANTIPLATELET EFFECT IN PATIENTS WITH TWO LOSS-OF-FUNCTION ALLELES OF THE "
            "CYP2C19 GENE See full prescribing information for complete boxed warning. Effectiveness of "
            "clopidogrel depends on conversion to an active metabolite by the cytochrome P450 (CYP) "
            "system, principally CYP2C19. ( 5.1 , 12.3 ) Tests are available to identify patients who are "
            "CYP2C19 poor metabolizers. (12.5) Consider use of another platelet P2Y 12 inhibitor in "
            "patients identified as CYP2C19 poor metabolizers. ( 5.1)"
        ),
        "clinical_pharmacology": (
            "The effect of proton pump inhibitors (PPI) on the systemic exposure to the clopidogrel "
            "active metabolite following multiple doses of clopidogrel 75 mg evaluated in dedicated drug "
            "interaction studies is presented in Figure 1. Pharmacodynamic and pharmacokinetic parameters "
            "measured in these studies showed that the interaction was highest with omeprazole and least "
            "with dexlansoprazole. Opioids Coadministration of 5 mg intravenous morphine with 600 mg "
            "loading dose of clopidogrel in healthy adults decreased the AUC and C max of clopidogrel’s "
            "thiol metabolites by 34%. Mean platelet aggregation was higher up to 2 to 4 hours with "
            "morphine coadministration. Effect of clopidogrel on other drugs In vitro studies have shown "
            "that the glucuronide metabolite of clopidogrel is a strong inhibitor of CYP2C8. Concomitant "
            "administration of repaglinide with clopidogrel increased the systemic exposure to "
            "repaglinide (AUC 0-∞ ) by 5.1-fold following the loading dose (300 mg) and by 3.9-fold on "
            "day 3 of the maintenance dose (75 mg) of clopidogrel [see Drug Interactions (7.8) ] . Figure "
            "1 12.5 Pharmacogenomics CYP2C19 is involved in the formation of both the active metabolite "
            "and the 2-oxo-clopidogrel intermediate metabolite. Clopidogrel active metabolite "
            "pharmacokinetics and antiplatelet effects, as measured by ex vivo platelet aggregation "
            "assays, differ according to CYP2C19 genotype. Patients who are homozygous for nonfunctional "
            "alleles of the CYP2C19 gene are termed “CYP2C19 poor metabolizers.” Approximately 2% of "
            "White and 4% of Black patients are poor metabolizers; the prevalence of poor metabolism is "
            "higher in Asian patients (e.g., 14% of Chinese). Tests are available to identify patients "
            "who are CYP2C19 poor metabolizers. A crossover study in 40 healthy subjects, 10 each in the "
            "four CYP2C19 metabolizer groups, evaluated pharmacokinetic and antiplatelet responses using "
            "300 mg followed by 75 mg per day and 600 mg followed by 150 mg per day, each for a total of "
            "5 days. Decreased active metabolite exposure and diminished inhibition of platelet "
            "aggregation were observed in the poor metabolizers as compared to the other groups."
        ),
        "drug_interactions": (
            "7 DRUG INTERACTIONS CYP2C19 inducers: Increases levels of clopidogrel active metabolite and "
            "increases platelet inhibition. ( 7.1 ) Opioids: Decreased exposure to clopidogrel. Consider "
            "use of parenteral antiplatelet agent. ( 7.3 ) Nonsteroidal anti-inflammatory drugs (NSAIDs), "
            "warfarin, selective serotonin and serotonin norepinephrine reuptake inhibitors (SSRIs, "
            "SNRIs): Increases risk of bleeding. ( 7.4 , 7.5 , 7.6 ) Other Antiplatelet Agents: Increases "
            "the risk of bleeding due to an additive effect. ( 7.7 ) Repaglinide (CYP2C8 substrates): "
            "Increases substrate plasma concentrations. ( 7.8 ) 7.1 CYP2C19 Inducers Since clopidogrel is "
            "metabolized to its active metabolite partly by CYP2C19, use of drugs that induce the "
            "activity of this enzyme would be expected to result in increased drug levels of the active "
            "metabolite of clopidogrel. Rifampin strongly induces CYP2C19 resulting to both an increase "
            "level of clopidogrel active metabolite and platelet inhibition, which in particular might "
            "potentiate the risk of bleeding. As a precaution, avoid concomitant use of strong CYP2C19 "
            "inducers [see Warnings and Precautions (5.1) and Clinical Pharmacology (12.3) ] . 7.2 "
            "CYP2C19 Inhibitors Clopidogrel is metabolized to its active metabolite in part by CYP2C19. "
            "Concomitant use of drugs that inhibit the activity of this enzyme results in reduced plasma "
            "concentrations of the active metabolite of clopidogrel and a reduction in platelet "
            "inhibition [see Warnings and Precautions (5.1) ] . Omeprazole or Esomeprazole Avoid "
            "concomitant use of clopidogrel with omeprazole or esomeprazole. In clinical studies, "
            "omeprazole was shown to reduce significantly the antiplatelet activity of clopidogrel when "
            "given concomitantly or 12 hours apart. A similar reduction in antiplatelet activity was "
            "observed with esomeprazole when given concomitantly with clopidogrel. Dexlansoprazole, "
            "lansoprazole, and pantoprazole had less effect on the antiplatelet activity of clopidogrel "
            "than did omeprazole or esomeprazole [see Warnings and Precautions (5.1) and Clinical "
            "Pharmacology (12.3) ]. 7.3 Opioids As with other oral P2Y 12 inhibitors, coadministration of "
            "opioid agonists delay and reduce the absorption of clopidogrel, presumably because of slowed "
            "gastric emptying, resulting in reduced exposure to its metabolites [see Clinical "
            "Pharmacology (12.3) ] . Consider the use of a parenteral antiplatelet agent in acute "
            "coronary syndrome patients requiring coadministration of morphine or other opioid agonists."
        ),
    },
    "warfarin": {
        "boxed_warning": (
            "WARNING: BLEEDING RISK Warfarin sodium can cause major or fatal bleeding [see Warnings and "
            "Precautions ( 5.1 )] . Perform regular monitoring of INR in all treated patients [see Dosage "
            "and Administration ( 2.1 )] . Drugs, dietary changes, and other factors affect INR levels "
            "achieved with warfarin sodium therapy [see Drug Interactions ( 7 )] . Instruct patients "
            "about prevention measures to minimize risk of bleeding and to report signs and symptoms of "
            "bleeding [see Patient Counseling Information ( 17 )] . WARNING: BLEEDING RISK See full "
            "prescribing information for complete boxed warning. Warfarin sodium can cause major or fatal "
            "bleeding. ( 5.1 ) Perform regular monitoring of INR in all treated patients. ( 2.1 ) Drugs, "
            "dietary changes, and other factors affect INR levels achieved with warfarin sodium therapy. "
            "( 7 ) Instruct patients about prevention measures to minimize risk of bleeding and to report "
            "signs and symptoms of bleeding. ( 17 )"
        ),
        "clinical_pharmacology": (
            "12.5 Pharmacogenomics CYP2C9 and VKORC1 Polymorphisms The S -enantiomer of warfarin is "
            "mainly metabolized to 7-hydroxywarfarin by CYP2C9, a polymorphic enzyme. The variant "
            "alleles, CYP2C9*2 and CYP2C9*3, result in decreased in vitro CYP2C9 enzymatic "
            "7-hydroxylation of S-warfarin. The frequencies of these alleles in Caucasians are "
            "approximately 11% and 7% for CYP2C9*2 and CYP2C9*3, respectively. Other CYP2C9 alleles "
            "associated with reduced enzymatic activity occur at lower frequencies, including *5, *6, and "
            "*11 alleles in populations of African ancestry and *5, *9, and *11 alleles in Caucasians. "
            "Warfarin reduces the regeneration of vitamin K from vitamin K epoxide in the vitamin K cycle "
            "through inhibition of VKOR, a multiprotein enzyme complex. Certain single nucleotide "
            "polymorphisms in the VKORC1 gene (e.g., –1639G > A) have been associated with variable "
            "warfarin dose requirements. VKORC1 and CYP2C9 gene variants generally explain the largest "
            "proportion of known variability in warfarin dose requirements. CYP2C9 and VKORC1 genotype "
            "information, when available, can assist in selection of the initial dose of warfarin [see "
            "Dosage and Administration ( 2.3 )] ."
        ),
        "drug_interactions": (
            "7.3 Drugs that Increase Bleeding Risk Examples of drugs known to increase the risk of "
            "bleeding are presented in Table 3 . Because bleeding risk is increased when these drugs are "
            "used concomitantly with warfarin, closely monitor patients receiving any such drug with "
            "warfarin. Table 3: Drugs that Can Increase the Risk of Bleeding Drug Class Specific Drugs "
            "Anticoagulants argatroban, dabigatran, bivalirudin, desirudin, heparin, lepirudin "
            "Antiplatelet Agents aspirin, cilostazol, clopidogrel, dipyridamole, prasugrel, ticlopidine "
            "Non-steroidal Anti-Inflammatory Agents celecoxib, diclofenac, diflunisal, fenoprofen, "
            "ibuprofen, indomethacin, ketoprofen, ketorolac, mefenamic acid, naproxen, oxaprozin, "
            "piroxicam, sulindac Serotonin Reuptake Inhibitors citalopram, desvenlafaxine, duloxetine, "
            "escitalopram, fluoxetine, fluvoxamine, milnacipran, paroxetine, sertraline, venlafaxine, "
            "vilazodone 7.4 Antibiotics and Antifungals There have been reports of changes in INR in "
            "patients taking warfarin and antibiotics or antifungals, but clinical pharmacokinetic "
            "studies have not shown consistent effects of these agents on plasma concentrations of "
            "warfarin. Closely monitor INR when starting or stopping any antibiotic or antifungal in "
            "patients taking warfarin. 7.5 Botanical (Herbal) Products and Foods More frequent INR "
            "monitoring should be performed when starting or stopping botanicals. Few adequate, "
            "well-controlled studies evaluating the potential for metabolic and/or pharmacologic "
            "interactions between botanicals and warfarin sodium exist. Due to a lack of manufacturing "
            "standardization with botanical medicinal preparations, the amount of active ingredients may "
            "vary. This could further confound the ability to assess potential interactions and effects "
            "on anticoagulation. Some botanicals may cause bleeding events when taken alone (e.g., garlic "
            "and Ginkgo biloba) and may have anticoagulant, antiplatelet, and/or fibrinolytic properties. "
            "These effects would be expected to be additive to the anticoagulant effects of warfarin "
            "sodium. Conversely, some botanicals may decrease the effects of warfarin sodium (e.g., "
            "co-enzyme Q 10 , St. John’s wort, ginseng). Some botanicals and foods can interact with "
            "warfarin sodium through CYP450 interactions (e.g., echinacea, grapefruit juice, ginkgo, "
            "goldenseal, St. John’s wort). The amount of vitamin K in food may affect therapy with "
            "warfarin sodium."
        ),
    },
    "simvastatin": {
        "boxed_warning": "",
        "clinical_pharmacology": (
            "Drug Interaction Studies Simvastatin acid is a substrate of the transport protein OATP1B1. "
            "Concomitant administration of inhibitors of the transport protein OATP1B1 and/or CYP3A4 may "
            "lead to increased exposure of simvastatin acid. Cyclosporine has been shown to increase the "
            "AUC of statins; although the mechanism is not fully understood, the increase in AUC for "
            "simvastatin acid is presumably due, in part, to inhibition of CYP3A4 and/or OATP1B1 [see "
            "DRUG INTERACTIONS ( 7 )] . Table 4 displays the effect of coadministered drugs or grapefruit "
            "juice on simvastatin systemic exposure [see DRUG INTERACTIONS ( 7 )] . Table 4: Effect of "
            "Coadministered Drugs or Grapefruit Juice on Simvastatin Systemic Exposure * Results based on "
            "a chemical assay except results with propranolol as indicated. † Results could be "
            "representative of the following CYP3A4 inhibitors: ketoconazole, erythromycin, "
            "clarithromycin, HIV protease inhibitors, and nefazodone. ‡ Simvastatin acid refers to the "
            "β-hydroxyacid of simvastatin. § The effect of amounts of grapefruit juice between those used "
            "in these two studies on simvastatin pharmacokinetics has not been studied. ¶ "
            "Double-strength: one can of frozen concentrate diluted with one can of water. Grapefruit "
            "juice was administered TID for 2 days, and 200 mL together with single dose simvastatin and "
            "30 and 90 minutes following single dose simvastatin on Day 3. & Single-strength: one can of "
            "frozen concentrate diluted with 3 cans of water. Grapefruit juice was administered with "
            "breakfast for 3 days, and simvastatin was administered in the evening on Day 3. "
            "Coadministered Drug or Grapefruit Juice Dosing of Coadministered Drug or Grapefruit Juice "
            "Dosing of Simvastatin Geometric Mean Ratio (Ratio * with / without coadministered drug) No "
            "Effect = 1.00 AUC C max Telithromycin † 200 mg QD for 4 days 80 mg simvastatin acid ‡ 12 15 "
            "simvastatin 8.9 5.3 Nelfinavir † 1250 mg BID for 14 days 20 mg QD for 28 days simvastatin "
            "acid ‡ simvastatin 6 6.2 Itraconazole † 200 mg QD for 4 days 80 mg simvastatin acid ‡ 13.1 "
            "simvastatin 13.1 Posaconazole 100 mg (oral suspension) QD for 13 days 40 mg simvastatin acid "
            "simvastatin 7.3 10.3 9.2 9.4 200 mg (oral suspension) QD for 13 days 40 mg simvastatin acid "
            "simvastatin 8.5 10.6 9.5 11.4 Gemfibrozil 600 mg BID for 3 days 40 mg simvastatin acid "
            "simvastatin 2.85 1.35 2.18 0.91 Grapefruit Juice § (high dose) 200 mL of double-strength TID "
            "¶ 60 mg single dose simvastatin acid simvastatin 7 16 Grapefruit Juice § (low dose)"
        ),
        "drug_interactions": (
            "Colchicine Clinical Impact: Cases of myopathy and rhabdomyolysis have been reported with "
            "concomitant use of colchicine with simvastatin. Intervention: Consider if the benefit of "
            "using colchicine concomitantly with simvastatin outweighs the increased risk of myopathy and "
            "rhabdomyolysis. If concomitant use is decided, monitor patients for signs and symptoms of "
            "myopathy, particularly during initiation of therapy and during upward dose titration of "
            "either drug. Grapefruit Juice Clinical Impact: Grapefruit juice can raise the plasma levels "
            "of simvastatin and may increase the risk of myopathy and rhabdomyolysis. Intervention: Avoid "
            "grapefruit juice when taking simvastatin. 7.2 Simvastatin Effects on Other Drugs Table 3 "
            "presents Simvastatin's effect on other drugs and instructions for preventing or managing "
            "them. Table 3: Simvastatin Effects on Other Drugs Coumarin Anticoagulants Clinical Impact: "
            "Simvastatin may potentiate the effect of coumarin anticoagulants and increase the INR. The "
            "concomitant use of simvastatin (20 to 40 mg) and coumarin anticoagulants increased the INR "
            "from a baseline of 1.7 to 1.8 in healthy subjects and from 2.6 to 3.4 in patients with "
            "hyperlipidemia. There are postmarketing reports of clinically evident bleeding and/or "
            "increased INR in patients taking concomitant statins and warfarin. Intervention: In patients "
            "taking coumarin anticoagulants, obtain an INR before starting simvastatin and frequently "
            "enough after initiation, dose titration, or discontinuation to ensure that no significant "
            "alteration in INR occurs. Once the INR is stable, monitor INR at regularly recommended "
            "intervals. Digoxin Clinical Impact: Concomitant use of digoxin with simvastatin may result "
            "in elevated plasma digoxin concentrations [ see CLINICAL PHARMACOLOGY ( 12.3 ) ]. "
            "Intervention: Monitor digoxin levels in patients taking digoxin when simvastatin is "
            "initiated."
        ),
    },
    "codeine": {
        "boxed_warning": (
            "Ultra-Rapid Metabolism of Codeine and Other Risk Factors for Life-Threatening Respiratory "
            "Depression in Children Life-threatening respiratory depression and death have occurred in "
            "children who received codeine. Most of the reported cases occurred following tonsillectomy "
            "and/or adenoidectomy, and many of the children had evidence of being an ultra-rapid "
            "metabolizer of codeine due to a CYP2D6 polymorphism [see Warnings and Precautions ( 5.6 )]. "
            "Codeine Sulfate Tablets are contraindicated in children younger than 12 years of age and in "
            "children younger than 18 years of age following tonsillectomy and/or adenoidectomy [see "
            "Contraindications ( 4 )]. Avoid the use of Codeine Sulfate Tablets in adolescents 12 to 18 "
            "years of age who have other risk factors that may increase their sensitivity to the "
            "respiratory depressant effects of codeine. Interactions with Drugs Affecting Cytochrome P450 "
            "Isoenzymes The effects of concomitant use or discontinuation of cytochrome P450 3A4 "
            "inducers, 3A4 inhibitors, or 2D6 inhibitors with codeine are complex. Use of cytochrome P450 "
            "3A4 inducers, 3A4 inhibitors, or 2D6 inhibitors with Codeine Sulfate Tablets requires "
            "careful consideration of the effects on the parent drug, codeine, and the active metabolite, "
            "morphine. [See Warnings and Precautions ( 5.7 ), Drug Interactions ( 7 )] ."
        ),
        "clinical_pharmacology": (
            "Metabolism: About 70% to 80% of the administered dose of codeine is metabolized by "
            "conjugation with glucuronic acid to codeine-6-glucuronide (C6G) and via O -demethylation to "
            "morphine (about 5% to 10%) and N -demethylation to norcodeine (about 10%) respectively. "
            "UDP-glucuronosyltransferase (UGT) 2B7 and 2B4 are the major enzymes mediating "
            "glucuronidation of codeine to C6G. Cytochrome P450 2D6 is the major enzyme responsible for "
            "conversion of codeine to morphine and P450 3A4 is the major enzyme mediating conversion of "
            "codeine to norcodeine. Morphine and norcodeine are further metabolized by conjugation with "
            "glucuronic acid. The glucuronide metabolites of morphine are morphine-3-glucuronide (M3G) "
            "and morphine-6-glucuronide (M6G). Morphine and M6G are known to have analgesic activity in "
            "humans. The analgesic activity of C6G in humans is unknown. Norcodeine and M3G are generally "
            "not considered to possess analgesic properties. Excretion: Approximately 90% of the total "
            "dose of codeine is excreted through the kidneys, of which approximately 10% is unchanged "
            "codeine. Plasma half-lives of codeine and its metabolites have been reported to be "
            "approximately 3 hours."
        ),
        "drug_interactions": (
            "Inhibitors of CYP2D6 Clinical Impact: Codeine is metabolized by CYP2D6 to form morphine. The "
            "concomitant use of Codeine Sulfate Tablets and CYP2D6 inhibitors can increase the plasma "
            "concentration of codeine, but can decrease the plasma concentration of active metabolite "
            "morphine, which could result in reduced analgesic efficacy or symptoms of opioid withdrawal, "
            "particularly when an inhibitor is added after a stable dose of Codeine Sulfate Tablets is "
            "achieved [see Clinical Pharmacology ( 12.3 )] . After stopping a CYP2D6 inhibitor, as the "
            "effects of the inhibitor decline, the codeine plasma concentration will decrease but the "
            "active metabolite morphine plasma concentration will increase, which could increase or "
            "prolong adverse reactions and may cause potentially fatal respiratory depression [see "
            "Clinical Pharmacology ( 12.3 )] . Intervention: If concomitant use with a CYP2D6 inhibitor "
            "is necessary, or if a CYP2D6 inhibitor is discontinued after concomitant use, consider "
            "dosage adjustment of Codeine Sulfate Tablets and evaluate patients at frequent intervals. If "
            "concomitant use with CYP2D6 inhibitors is necessary, evaluate patients for reduced efficacy "
            "or signs and symptoms of opioid withdrawal and consider increasing the dosage of Codeine "
            "Sulfate Tablets as needed. After stopping use of a CYP2D6 inhibitor, consider reducing the "
            "dosage of Codeine Sulfate Tablets and evaluate patients at frequent intervals for signs and "
            "symptoms of respiratory depression or sedation. Examples: Paroxetine, fluoxetine, bupropion, "
            "quinidine."
        ),
    },
    "omeprazole": {
        "boxed_warning": "",
        "clinical_pharmacology": (
            "Drug Interaction Studies Effect of Omeprazole on Other Drugs Omeprazole is a time-dependent "
            "inhibitor of CYP2C19 and can increase the systemic exposure of co-administered drugs that "
            "are CYP2C19 substrates. In addition, administration of omeprazole increases intragastric pH "
            "and can alter the systemic exposure of certain drugs that exhibit pH-dependent solubility. "
            "Antiretrovirals For some antiretroviral drugs, such as rilpivirine, atazanavir and "
            "nelfinavir, decreased serum concentrations have been reported when given together with "
            "omeprazole [see Drug Interactions (7) ]. Rilpivirine: Following multiple doses of "
            "rilpivirine (150 mg, daily) and omeprazole (20 mg, daily), AUC was decreased by 40%, C max "
            "by 40%, and C min by 33% for rilpivirine. Nelfinavir: Following multiple doses of nelfinavir "
            "(1250 mg, twice daily) and omeprazole (40 mg daily), AUC was decreased by 36% and 92%, Cmax "
            "by 37% and 89% and Cmin by 39% and 75% respectively for nelfinavir and M8. Atazanavir: "
            "Following multiple doses of atazanavir (400 mg, daily) and omeprazole (40 mg, daily, 2 hours "
            "before atazanavir), AUC was decreased by 94%, C max by 96%, and C min by 95%. Saquinavir: "
            "Following multiple dosing of saquinavir/ritonavir (1000/100 mg) twice daily for 15 days with "
            "omeprazole 40 mg daily co-administered days 11 to 15. AUC was increased by 82%, Cmax by 75%, "
            "and Cmin by 106%. The mechanism behind this interaction is not fully elucidated. Therefore, "
            "clinical and laboratory monitoring for saquinavir toxicity is recommended during concurrent "
            "use with omeprazole. Clopidogrel In a crossover clinical study, 72 healthy subjects were "
            "administered clopidogrel (300 mg loading dose followed by 75 mg per day) alone and with "
            "omeprazole (80 mg at the same time as clopidogrel) for 5 days. The exposure to the active "
            "metabolite of clopidogrel was decreased by 46% (Day 1) and 42% (Day 5) when clopidogrel and "
            "omeprazole were administered together. Results from another crossover study in healthy "
            "subjects showed a similar pharmacokinetic interaction between clopidogrel (300 mg loading "
            "dose/75 mg daily maintenance dose) and omeprazole 80 mg daily when co-administered for 30 "
            "days. Exposure to the active metabolite of clopidogrel was reduced by 41% to 46% over this "
            "time period."
        ),
        "drug_interactions": (
            "CYP2C19 Substrates (e.g., clopidogrel, citalopram, cilostazol, phenytoin, diazepam) "
            "Clopidogrel Clinical Impact: Concomitant use of omeprazole 80 mg results in reduced plasma "
            "concentrations of the active metabolite of clopidogrel and a reduction in platelet "
            "inhibition [ see Clinical Pharmacology (12.3) ]. There are no adequate combination studies "
            "of a lower dose of omeprazole or a higher dose of clopidogrel in comparison with the "
            "approved dose of clopidogrel. Intervention: Avoid concomitant use with omeprazole. Consider "
            "use of alternative anti-platelet therapy [see Warnings and Precautions (5.7) ]. Citalopram "
            "Clinical Impact: Increased exposure of citalopram leading to an increased risk of QT "
            "prolongation [see Clinical Pharmacology (12.3) ]. Intervention: Limit the dose of citalopram "
            "to a maximum of 20 mg per day. See prescribing information for citalopram. Cilostazol "
            "Clinical Impact: Increased exposure of one of the active metabolites of cilostazol "
            "(3,4-dihydro-cilostazol) [see Clinical Pharmacology (12.3) ]. Intervention: Reduce the dose "
            "of cilostazol to 50 mg twice daily. See prescribing information for cilostazol. Phenytoin "
            "Clinical Impact: Potential for increased exposure of phenytoin. Intervention: Monitor "
            "phenytoin serum concentrations. Dose adjustment may be needed to maintain therapeutic drug "
            "concentrations. See prescribing information for phenytoin. Diazepam Clinical Impact: "
            "Increased exposure of diazepam [see Clinical Pharmacology (12.3) ]. Intervention: Monitor "
            "patients for increased sedation and reduce the dose of diazepam as needed. Digoxin Clinical "
            "Impact: Potential for increased exposure of digoxin [see Clinical Pharmacology (12.3) ]. "
            "Intervention: Monitor digoxin concentrations. Dose adjustment may be needed to maintain "
            "therapeutic drug concentrations. See digoxin prescribing information. Drugs Dependent on "
            "Gastric pH for Absorption (e.g., iron salts, erlotinib, dasatinib, nilotinib, mycophenolate "
            "mofetil, ketoconazole/itraconazole) Clinical Impact: Omeprazole can reduce the absorption of "
            "other drugs due to its effect on reducing intragastric acidity. Intervention: Mycophenolate "
            "mofetil (MMF): Co-administration of omeprazole in healthy subjects and in transplant "
            "patients receiving MMF has been reported to reduce the exposure to the active metabolite, "
            "mycophenolic acid (MPA), possibly due to a decrease in MMF solubility at an increased "
            "gastric pH."
        ),
    },
}
