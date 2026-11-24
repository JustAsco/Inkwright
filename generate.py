#!/usr/bin/env python3
"""Generate Wonderdraft symbol variants.

    .venv/bin/python generate.py                       # everything, 12 of each
    .venv/bin/python generate.py -t cottage,cave -n 30 --seed 7   # types or category words
    .venv/bin/python generate.py -t land                         # every Land - * category
    .venv/bin/python generate.py --style line           # transparent line art (no white fill)

Symbols go to out/<pack>/sprites/symbols/<pack> - <Group>/ (line style: out/<pack>Line/.../<pack> Line - <Group>/),
contact sheets to preview/<Group>/<Category>_<style>.png. Copy the pack folder into
Wonderdraft's assets folder.
"""
import argparse
import json
import math
import zlib
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wdgen import castles, divine, houses, lairs, ruins, terrain
from wdgen.ink import Canvas, Pen
from wdgen.proj import Proj

# name -> (category folder, draw fn, canvas w, canvas h[, ground line as fraction of h])
TYPES = {
    # settlements
    "house": ("Settlements - Houses", houses.single_house, 200, 200, 0.62),
    "hamlet": ("Settlements - Hamlets", houses.hamlet, 260, 240, 0.6),
    "town_village": ("Settlements - Villages", houses.village, 420, 340, 0.58),
    # ruins
    "cottage": ("Ruins - Cottages", ruins.ruined_cottage, 260, 220),
    "village": ("Ruins - Villages", ruins.ruined_village, 420, 320),
    "round_tower": ("Ruins - Round Towers", ruins.ruined_tower, 220, 260),
    "keep": ("Ruins - Keeps", ruins.ruined_keep, 220, 260),
    "castle": ("Ruins - Castles", castles.castle, 400, 400, 0.66),
    "round_spire": ("Ruins - Round Spires", partial(castles.spire, kind="round"), 200, 340, 0.86),
    "gothic_spire": ("Ruins - Gothic Spires", partial(castles.spire, kind="gothic"), 200, 340, 0.86),
    # lairs
    "hill_cave": ("Lairs - Hill Caves", lairs.hill_cave, 280, 220),
    "cliff_cave": ("Lairs - Cliff Caves", lairs.cliff_cave, 280, 240, 0.75),
    "boulder_cave": ("Lairs - Boulder Caves", lairs.boulder_cave, 280, 220),
    "sunken_cave": ("Lairs - Sunken Entrances", lairs.low_entrance, 280, 200),
    "burrow": ("Lairs - Burrows", partial(lairs.burrow_lair, warren=False), 260, 200),
    "warren": ("Lairs - Warrens", partial(lairs.burrow_lair, warren=True), 300, 220),
    "crag_nest": ("Lairs - Crag Nests", partial(lairs.nest_lair, crag=True), 240, 240),
    "ground_nest": ("Lairs - Ground Nests", partial(lairs.nest_lair, crag=False), 240, 200),
    # divine
    "temple": ("Divine - Temples", divine.temple, 420, 340, 0.64),
    "stone_circle": ("Divine - Stone Circles", divine.stone_circle, 300, 260, 0.62),
    "colossus": ("Divine - Colossi", divine.statue_legs, 260, 300, 0.75),
    "robed_statue": ("Divine - Robed Statues", divine.statue_robed, 220, 300, 0.8),
    "sword": ("Divine - Swords", divine.gods_sword, 220, 260, 0.82),
    "obelisk": ("Divine - Obelisks", divine.obelisk, 200, 300, 0.8),
    # land
    "field": ("Land - Fields", terrain.fields, 320, 240, 0.5),
    "cliff": ("Land - Cliffs", terrain.cliff, 280, 200, 0.5),
    "steep_cliff": ("Land - Steep Cliffs", terrain.cliff_steep, 260, 260, 0.5),
    "canyon": ("Land - Canyons", terrain.canyon, 300, 200, 0.5),
    "canyon_bend": ("Land - Canyon Bends", terrain.canyon_bend, 300, 240, 0.5),
    "mesa": ("Land - Mesas", terrain.mesa, 260, 200, 0.5),
    "terraced_hill": ("Land - Terraced Hills", terrain.terraced_hill, 280, 260, 0.55),
}
LINE_PIECES = {"cliff", "steep_cliff", "canyon", "canyon_bend", "mesa", "terraced_hill"}
for _role in terrain.ARC_ROLES:
    TYPES[f"curve_{_role}"] = ("Land - Cliff Curves", partial(terrain.cliff_curve, role=_role), 260, 220, 0.5)
    TYPES[f"bay_{_role}"] = ("Land - Cliff Bays", partial(terrain.cliff_curve, role=_role, inner=True), 260, 220, 0.5)
    LINE_PIECES |= {f"curve_{_role}", f"bay_{_role}"}


def render(kind, seed, style="filled", res=2, ink=(30, 30, 30), wash=(150, 98, 62), meta=False,
           paper=(255, 255, 255)):
    """Draw one symbol. With meta=True also return its ground pivot in image pixels (x right, y down)."""
    _, fn, w, h, *oy = TYPES[kind]
    rng = np.random.default_rng([seed, zlib.crc32(kind.encode())])
    cv = Canvas(w, h, res=res)
    pen = Pen(cv, rng, width=1.6)
    P = Proj(w / 2, h * (oy[0] if oy else 0.68))
    fn(pen, P, rng)
    img = cv.to_image(ink=ink, paper=paper, filled=(style == "filled"), wash=wash, seed=seed)
    if not meta:
        return img
    cx, cy = cv.crop_origin
    return img, (P.ox * res - cx, P.oy * res - cy)


def symbol_meta(stem, kind, i, img, pivot_x, draw_mode):
    """One .wonderdraft_symbols entry. Offsets are from the image centre, +x right, +y up (pixels).

    Like Wonderdraft's own sprites, the pivot sits near the bottom of the drawing (the front ground
    point): Wonderdraft draws whichever symbol has the lower pivot in front. Open cliff/canyon
    pieces pivot on their centre instead so they chain naturally.
    """
    w, h = img.size
    bbox = img.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox() or (0, 0, w, h)
    if kind in LINE_PIECES:
        px, py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    else:
        px = (bbox[0] + bbox[2]) / 2 if pivot_x is None else pivot_x
        py = bbox[3] - max(3, 0.03 * (bbox[3] - bbox[1]))
    px, py = min(max(px, 0), w), min(max(py, 0), h)
    return {
        "name": f"{kind.replace('_', ' ').title()} {i + 1}",
        "radius": max(8, round((0.12 if kind in LINE_PIECES else 0.35) * (bbox[2] - bbox[0]))),
        "offset_x": round(px - w / 2),
        "offset_y": round(h / 2 - py),
        "draw_mode": draw_mode,
    }


def refresh_meta(out_dir, draw_mode):
    """Rewrite every .wonderdraft_symbols under out_dir from the PNGs already there (no re-render)."""
    n = 0
    for folder in sorted(Path(out_dir).glob("*/sprites/symbols/*")):
        meta_path = folder / ".wonderdraft_symbols"
        try:
            old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except json.JSONDecodeError:
            old = {}
        meta = {}
        for png in sorted(folder.glob("*.png")):
            stem = png.stem
            kind, _, idx = stem.rsplit("_", 2)
            img = Image.open(png)
            px = img.width / 2 + old[stem]["offset_x"] if stem in old else None
            meta[stem] = symbol_meta(stem, kind, int(idx) if idx.isdigit() else 0, img, px, draw_mode)
            n += 1
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return n


def contact_sheet(images, cols=6, cell=200, bg=(150, 178, 150), labels=None):
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGBA", (cols * cell, max(1, rows) * cell), (*bg, 255))
    d = ImageDraw.Draw(sheet)
    for i, img in enumerate(images):
        if img is None:
            continue
        im = img.copy()
        im.thumbnail((cell - 16, cell - 22))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.alpha_composite(im, (x + (cell - im.width) // 2, y + 4 + (cell - 22 - im.height) // 2))
        d.text((x + 6, y + cell - 16), labels[i] if labels else f"#{i:03d}", fill=(40, 60, 40, 255))
    return sheet


PARCHMENT = (226, 220, 190)
GRASS = (150, 178, 150)


def select(tokens):
    """Exact type names, or case-insensitive pieces of a category name (e.g. 'land', 'cave')."""
    out = []
    for tok in tokens:
        if tok in TYPES:
            hits = [tok]
        else:
            hits = [k for k, v in TYPES.items() if tok.lower() in v[0].lower() or tok.lower() in k]
        if not hits:
            raise SystemExit(f"unknown type or category: {tok!r}")
        out += [h for h in hits if h not in out]
    return out


def category_sheet(cat, per_kind, cols=6):
    """One sheet per category; several piece types get one labelled row each."""
    kinds = list(per_kind)
    if len(kinds) == 1:
        imgs = per_kind[kinds[0]]
        labels = [f"#{i:03d}" for i in range(len(imgs))]
    else:
        imgs, labels = [], []
        for k in kinds:
            row = per_kind[k][:cols]
            short = k.split("_", 1)[1] if "_" in k else k
            imgs += row + [None] * (cols - len(row))
            labels += [f"{short} #{i:03d}" for i in range(len(row))] + [""] * (cols - len(row))
    bg = PARCHMENT if cat.startswith("Land") else GRASS
    return contact_sheet(imgs, cols=cols, bg=bg, labels=labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--types", default=",".join(TYPES),
                    help="comma list of type names or category words, e.g. 'cottage,land,cave'")
    ap.add_argument("-n", "--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--style", choices=["filled", "line"], default="filled",
                    help="filled = white paper inside shapes; line = ink only, transparent inside")
    ap.add_argument("--res", type=int, default=3, help="output pixels per drawing unit")
    ap.add_argument("--ink", default="1e1e1e", help="hex ink colour")
    ap.add_argument("--paper", default="ffffff",
                    help="hex colour of the fill inside shapes (filled style); with sample_color a mid grey "
                         "may match the ground better than white")
    ap.add_argument("--wash", default="96623e", help="hex colour of the cliff/canyon shading, or 'none'")
    ap.add_argument("--draw-mode", default="sample_color",
                    choices=["normal", "sample_color", "custom_colors", "custom_colors_2"],
                    help="Wonderdraft draw_mode for .wonderdraft_symbols: sample_color tints the white fill to the land "
                         "colour (like Wonderdraft's own line-art sprites); normal draws as-is")
    ap.add_argument("--refresh-meta", action="store_true",
                    help="only rewrite the .wonderdraft_symbols files for the PNGs already in --out, then exit")
    ap.add_argument("--pack", default="Inkwright", help="pack name, used for the pack folder and palette folder names")
    ap.add_argument("--out", default="out")
    ap.add_argument("--preview", default="preview")
    args = ap.parse_args()

    if args.refresh_meta:
        print(f"rewrote metadata for {refresh_meta(args.out, args.draw_mode)} symbols")
        return
    ink = tuple(int(args.ink[i:i + 2], 16) for i in (0, 2, 4))
    wash = None if args.wash == "none" else tuple(int(args.wash[i:i + 2], 16) for i in (0, 2, 4))
    paper = tuple(int(args.paper[i:i + 2], 16) for i in (0, 2, 4))
    pack = args.pack + ("" if args.style == "filled" else "Line")
    label = args.pack + ("" if args.style == "filled" else " Line")
    by_cat = {}
    for kind in select([t.strip() for t in args.types.split(",") if t.strip()]):
        cat = TYPES[kind][0]
        folder = Path(args.out) / pack / "sprites" / "symbols" / f"{label} - {cat.split(' - ', 1)[0]}"  # one per group
        folder.mkdir(parents=True, exist_ok=True)
        imgs = []
        meta_path = folder / ".wonderdraft_symbols"
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except json.JSONDecodeError:
            meta = {}
        for i in range(args.count):
            img, pivot = render(kind, args.seed * 100003 + i, args.style, args.res, ink, wash, meta=True, paper=paper)
            stem = f"{kind}_{args.seed}_{i:03d}"
            img.save(folder / f"{stem}.png")
            meta[stem] = symbol_meta(stem, kind, i, img, pivot[0], args.draw_mode)
            imgs.append(img)
        # keep entries only for images that exist (other kinds share some folders)
        meta = {k: v for k, v in sorted(meta.items()) if (folder / f"{k}.png").exists()}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        by_cat.setdefault(cat, {})[kind] = imgs
        print(f"{kind:14s} -> {folder}  ({len(imgs)} symbols)")
    for cat, per_kind in by_cat.items():
        group, name = cat.split(" - ", 1)
        d = Path(args.preview) / group
        d.mkdir(parents=True, exist_ok=True)
        category_sheet(cat, per_kind).save(d / f"{name.replace(' ', '_')}_{args.style}.png")


if __name__ == "__main__":
    main()
