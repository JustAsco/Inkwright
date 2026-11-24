#!/usr/bin/env python3
"""Generate Wonderdraft symbol variants.

    .venv/bin/python generate.py                       # everything, 12 of each
    .venv/bin/python generate.py -t cottage,cave -n 30 --seed 7
    .venv/bin/python generate.py --style line           # transparent line art (no white fill)

Symbols go to out/<pack>/sprites/symbols/<Category>/ (line style: out/<pack>Line/...),
contact sheets to preview/. Copy the pack folder into Wonderdraft's assets folder.
"""
import argparse
import math
import zlib
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wdgen import castles, divine, lairs, ruins, terrain
from wdgen.ink import Canvas, Pen
from wdgen.proj import Proj

# name -> (category folder, draw fn, canvas w, canvas h[, ground line as fraction of h])
TYPES = {
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
    "temple": ("Divine - Temples", divine.temple, 320, 260, 0.62),
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
for _role in terrain.ARC_ROLES:
    TYPES[f"curve_{_role}"] = ("Land - Cliff Curves", partial(terrain.cliff_curve, role=_role), 260, 220, 0.5)
    TYPES[f"bay_{_role}"] = ("Land - Cliff Bays", partial(terrain.cliff_curve, role=_role, inner=True), 260, 220, 0.5)


def render(kind, seed, style="filled", res=2, ink=(30, 30, 30), wash=(150, 98, 62)):
    _, fn, w, h, *oy = TYPES[kind]
    rng = np.random.default_rng([seed, zlib.crc32(kind.encode())])
    cv = Canvas(w, h, res=res)
    pen = Pen(cv, rng, width=1.6)
    fn(pen, Proj(w / 2, h * (oy[0] if oy else 0.68)), rng)
    return cv.to_image(ink=ink, filled=(style == "filled"), wash=wash, seed=seed)


def contact_sheet(images, cols=6, cell=200, bg=(150, 178, 150)):
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGBA", (cols * cell, rows * cell), (*bg, 255))
    d = ImageDraw.Draw(sheet)
    for i, img in enumerate(images):
        im = img.copy()
        im.thumbnail((cell - 16, cell - 22))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.alpha_composite(im, (x + (cell - im.width) // 2, y + 4 + (cell - 22 - im.height) // 2))
        d.text((x + 6, y + cell - 16), f"#{i:03d}", fill=(40, 60, 40, 255))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--types", default=",".join(TYPES), help="comma list: " + ",".join(TYPES))
    ap.add_argument("-n", "--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--style", choices=["filled", "line"], default="filled",
                    help="filled = white paper inside shapes; line = ink only, transparent inside")
    ap.add_argument("--res", type=int, default=3, help="output pixels per drawing unit")
    ap.add_argument("--ink", default="1e1e1e", help="hex ink colour")
    ap.add_argument("--wash", default="96623e", help="hex colour of the cliff/canyon shading, or 'none'")
    ap.add_argument("--pack", default="RuinsAndLairs")
    ap.add_argument("--out", default="out")
    ap.add_argument("--preview", default="preview")
    args = ap.parse_args()

    ink = tuple(int(args.ink[i:i + 2], 16) for i in (0, 2, 4))
    wash = None if args.wash == "none" else tuple(int(args.wash[i:i + 2], 16) for i in (0, 2, 4))
    for kind in [t.strip() for t in args.types.split(",") if t.strip()]:
        cat = TYPES[kind][0]
        pack = args.pack + ("" if args.style == "filled" else "Line")
        folder = Path(args.out) / pack / "sprites" / "symbols" / cat
        folder.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i in range(args.count):
            img = render(kind, args.seed * 100003 + i, args.style, args.res, ink, wash)
            img.save(folder / f"{kind}_{args.seed}_{i:03d}.png")
            imgs.append(img)
        Path(args.preview).mkdir(exist_ok=True)
        contact_sheet(imgs).save(Path(args.preview) / f"{kind}_{args.style}.png")
        print(f"{kind:8s} -> {folder}  ({len(imgs)} symbols)")


if __name__ == "__main__":
    main()
