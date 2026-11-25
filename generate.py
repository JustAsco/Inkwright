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

from wdgen import castles, divine, flora, houses, lairs, mountains, ruins, terrain
from wdgen.ink import Canvas, Pen
from wdgen.proj import Proj

# name -> (category folder, draw fn, canvas w, canvas h[, ground line as fraction of h])
TYPES = {
    # trees (Wonderdraft Tree Brush pools -> sprites/trees/<pack> - <Name>/)
    "pine": ("Trees - Pines", flora.pine_brush, 120, 120, 0.84),
    "broadleaf": ("Trees - Broadleaf", flora.broadleaf_brush, 120, 120, 0.84),
    "cypress": ("Trees - Cypress", flora.cypress_brush, 110, 120, 0.84),
    "palm": ("Trees - Palms", flora.palm_brush, 120, 120, 0.84),
    "deadwood": ("Trees - Deadwood", flora.deadwood_brush, 120, 120, 0.84),
    # mountains & hills (Wonderdraft Mountain Brush pools -> sprites/mountains/<pack> - <Name>/)
    "mountain": ("Mountains - Mountains", mountains.mountain, 390, 250, 0.66),
    "varied_mountain": ("Mountains - Varied Mountains", mountains.varied_mountain, 410, 260, 0.66),
    "crag": ("Mountains - Crags", mountains.crag, 330, 270, 0.68),
    "hill": ("Mountains - Hills", mountains.hill, 310, 190, 0.62),
    # settlements
    "house": ("Settlements - Houses", houses.single_house, 200, 200, 0.62),
    "hamlet": ("Settlements - Hamlets", houses.hamlet, 280, 280, 0.58),
    "town_village": ("Settlements - Villages", houses.village, 420, 400, 0.56),
    # ruins
    "cottage": ("Ruins - Cottages", ruins.ruined_cottage, 260, 220),
    "village": ("Ruins - Villages", ruins.ruined_village, 420, 380, 0.58),
    "round_tower": ("Ruins - Round Towers", ruins.ruined_tower, 220, 260),
    "keep": ("Ruins - Keeps", ruins.ruined_keep, 220, 260),
    "castle": ("Ruins - Castles", castles.castle, 420, 420, 0.62),
    "round_spire": ("Ruins - Round Spires", partial(castles.spire, kind="round"), 200, 340, 0.86),
    "gothic_spire": ("Ruins - Gothic Spires", partial(castles.spire, kind="gothic"), 200, 340, 0.86),
    # lairs
    "hill_cave": ("Lairs - Hill Caves", lairs.hill_cave, 280, 220),
    "cliff_cave": ("Lairs - Cliff Caves", lairs.cliff_cave, 280, 240, 0.75),
    "boulder_cave": ("Lairs - Boulder Caves", lairs.boulder_cave, 280, 220),
    "sunken_cave": ("Lairs - Sunken Entrances", lairs.low_entrance, 280, 200),
    "burrow": ("Lairs - Burrows", lairs.burrow_lair, 260, 200),
    "crag_nest": ("Lairs - Crag Nests", partial(lairs.nest_lair, crag=True), 240, 240),
    "ground_nest": ("Lairs - Ground Nests", partial(lairs.nest_lair, crag=False), 240, 200),
    # divine
    "temple": ("Divine - Temples", divine.temple, 420, 400, 0.60),
    "stone_circle": ("Divine - Stone Circles", divine.stone_circle, 300, 260, 0.62),
    "colossus": ("Divine - Colossi", divine.statue_legs, 260, 300, 0.75),
    "robed_statue": ("Divine - Robed Statues", divine.statue_robed, 220, 300, 0.8),
    "sword": ("Divine - Swords", divine.gods_sword, 220, 260, 0.82),
    "obelisk": ("Divine - Obelisks", divine.obelisk, 200, 300, 0.8),
    # land
    "field": ("Land - Fields", terrain.fields, 340, 320, 0.5),
    # terraforming (stackable cliffs, mesas, canyons & neutralizing end-caps)
    "cliff": ("Terraforming - Cliffs", terrain.cliff, 280, 200, 0.5),
    "steep_cliff": ("Terraforming - Steep Cliffs", terrain.cliff_steep, 260, 260, 0.5),
    "canyon": ("Terraforming - Canyons", terrain.canyon, 300, 200, 0.5),
    "canyon_bend": ("Terraforming - Canyon Bends", terrain.canyon_bend, 300, 240, 0.5),
    "mesa": ("Terraforming - Mesas", terrain.mesa, 260, 240, 0.5),
}
LINE_PIECES = {"cliff", "steep_cliff", "canyon", "canyon_bend", "mesa"}
TREE_PIECES = {"pine", "broadleaf", "cypress", "palm", "deadwood"}
MOUNTAIN_PIECES = {"mountain", "varied_mountain", "crag", "hill"}
for _role in terrain.CLIFF_END_ROLES:
    TYPES[f"cliff_end_{_role}"] = ("Terraforming - Cliff Ends", partial(terrain.cliff_end, role=_role), 260, 220, 0.5)
    LINE_PIECES.add(f"cliff_end_{_role}")
for _role in terrain.CANYON_END_ROLES:
    TYPES[f"canyon_end_{_role}"] = ("Terraforming - Canyon Ends", partial(terrain.canyon_end, role=_role), 260, 240, 0.5)
    LINE_PIECES.add(f"canyon_end_{_role}")
for _role in terrain.ARC_ROLES:
    TYPES[f"curve_{_role}"] = ("Terraforming - Cliff Curves", partial(terrain.cliff_curve, role=_role), 260, 220, 0.5)
    TYPES[f"bay_{_role}"] = ("Terraforming - Cliff Bays", partial(terrain.cliff_curve, role=_role, inner=True), 260, 220, 0.5)
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
    pieces pivot on their centre instead so they chain naturally. Trees and mountains use tighter
    radii so Wonderdraft's brush tools pack and overlap them organically into forests and ranges.
    """
    w, h = img.size
    bbox = img.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox() or (0, 0, w, h)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Compact footprint measure so wide shapes (mountains, hills, villages, fields) don't get oversized circular radii
    span = min(bw, bh * 1.35)
    if kind in LINE_PIECES:
        px, py = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        rad = max(8, round(0.10 * span))
    elif kind in TREE_PIECES:
        px = (bbox[0] + bbox[2]) / 2 if pivot_x is None else pivot_x
        py = bbox[3] - max(2, 0.03 * bh)
        # Balanced canopy-overlap radius (~60-70% spacing at density 1.0; neither stacked nor sparse)
        rad = max(22, round(0.32 * bw))
    elif kind in MOUNTAIN_PIECES:
        px = (bbox[0] + bbox[2]) / 2
        py = bbox[3] - max(4, 0.16 * bh)
        # Interlocking foothill radius (~45% overlap at density 1.0, midpoint between old 53 and WD default 134)
        rad = max(28, round(0.22 * span))
    else:
        px = (bbox[0] + bbox[2]) / 2 if pivot_x is None else pivot_x
        py = bbox[3] - max(3, 0.03 * bh)
        rad = max(10, round(0.18 * span))
    px, py = min(max(px, 0), w), min(max(py, 0), h)
    return {
        "name": f"{kind.replace('_', ' ').title()} {i + 1}",
        "radius": rad,
        "offset_x": round(px - w / 2),
        "offset_y": round(h / 2 - py),
        "draw_mode": draw_mode,
    }


def pool_meta(label_name, entries, draw_mode):
    """Flat single-object .wonderdraft_symbols entry required by Wonderdraft for trees/ and mountains/ pools."""
    rads = sorted(e["radius"] for e in entries)
    oxs = sorted(e["offset_x"] for e in entries)
    oys = sorted(e["offset_y"] for e in entries)
    mid = len(entries) // 2
    return {
        "name": label_name,
        "radius": rads[mid] if rads else 24,
        "offset_x": oxs[mid] if oxs else 0,
        "offset_y": oys[mid] if oys else 0,
        "draw_mode": draw_mode,
    }


def _is_pool_folder(folder):
    """Wonderdraft trees/ and mountains/ subfolders are brush pools using a flat JSON object."""
    return folder.parent.name in ("trees", "mountains")


def refresh_meta(out_dir, draw_mode):
    """Rewrite every .wonderdraft_symbols under out_dir (flat object for trees/mountains, per-sprite dict for symbols)."""
    n = 0
    for folder in sorted(Path(out_dir).glob("*/sprites/*/*")):
        if not folder.is_dir():
            continue
        meta_path = folder / ".wonderdraft_symbols"
        try:
            old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except json.JSONDecodeError:
            old = {}
        is_pool = _is_pool_folder(folder)
        meta = {}
        for png in sorted(folder.glob("*.png")):
            stem = png.stem
            parts = stem.rsplit("_", 2)
            if len(parts) == 3 and parts[2].isdigit():
                kind, idx = parts[0], int(parts[2])
            else:
                kind, idx = ("mountain" if folder.parent.name == "mountains" else "pine" if folder.parent.name == "trees" else stem), 0
            img = Image.open(png)
            px = (img.width / 2 + old[stem]["offset_x"]) if (not is_pool and isinstance(old.get(stem), dict)) else None
            meta[stem] = symbol_meta(stem, kind, idx, img, px, draw_mode)
            n += 1
        if is_pool:
            out_meta = pool_meta(folder.name, list(meta.values()), draw_mode)
        else:
            out_meta = meta
        meta_path.write_text(json.dumps(out_meta, indent=2) + "\n")
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
    bg = PARCHMENT if (cat.startswith(("Land", "Terraforming")) or cat in ("Mountains - Mountains", "Mountains - Varied Mountains", "Mountains - Crags")) else GRASS
    return contact_sheet(imgs, cols=cols, bg=bg, labels=labels)


def sprite_folder(out_dir, pack, label, cat):
    """Route Trees -> sprites/trees/<pool>/, Mountains -> sprites/mountains/<pool>/, others -> sprites/symbols/<group>/."""
    group, name = cat.split(" - ", 1)
    if group == "Trees":
        return Path(out_dir) / pack / "sprites" / "trees" / f"{label} - {name}"
    if group == "Mountains":
        return Path(out_dir) / pack / "sprites" / "mountains" / f"{label} - {name}"
    return Path(out_dir) / pack / "sprites" / "symbols" / f"{label} - {group}"


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
        folder = sprite_folder(args.out, pack, label, cat)
        folder.mkdir(parents=True, exist_ok=True)
        imgs = []
        meta_path = folder / ".wonderdraft_symbols"
        is_pool = _is_pool_folder(folder)
        try:
            meta = json.loads(meta_path.read_text()) if (meta_path.exists() and not is_pool) else {}
        except json.JSONDecodeError:
            meta = {}
        pool_entries = []
        for i in range(args.count):
            img, pivot = render(kind, args.seed * 100003 + i, args.style, args.res, ink, wash, meta=True, paper=paper)
            stem = f"{kind}_{args.seed}_{i:03d}"
            img.save(folder / f"{stem}.png")
            entry = symbol_meta(stem, kind, i, img, pivot[0], args.draw_mode)
            if is_pool:
                pool_entries.append(entry)
            else:
                meta[stem] = entry
            imgs.append(img)
        if is_pool:
            out_meta = pool_meta(folder.name, pool_entries, args.draw_mode)
        else:
            # keep entries only for images that exist (other kinds share some folders)
            out_meta = {k: v for k, v in sorted(meta.items()) if (folder / f"{k}.png").exists()}
        meta_path.write_text(json.dumps(out_meta, indent=2) + "\n")
        by_cat.setdefault(cat, {})[kind] = imgs
        print(f"{kind:14s} -> {folder}  ({len(imgs)} symbols)")
    for cat, per_kind in by_cat.items():
        group, name = cat.split(" - ", 1)
        d = Path(args.preview) / group
        d.mkdir(parents=True, exist_ok=True)
        category_sheet(cat, per_kind).save(d / f"{name.replace(' ', '_')}_{args.style}.png")


if __name__ == "__main__":
    main()
