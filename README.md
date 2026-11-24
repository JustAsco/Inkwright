# Inkwright

A Wonderdraft asset generator.


A procedural generator for hand-inked fantasy map symbols: settlements, ruins, beast lairs,
divine sites and land features. Each run draws as many variants as you ask for. Every symbol is built from random
parts, so no two come out the same, and the stroke weight and shading stay the same across the
whole set. Output is transparent PNGs laid out the way Wonderdraft expects custom symbols.

## Examples

**Ruined cottages**: walls collapse differently each time. Some keep part of their roof, some
only the bare rafters or a standing chimney.

![Ruined cottages, filled style](docs/cottage_filled.png)
![Ruined cottages, line style](docs/cottage_line.png)

**Round spires**: intact, caved-in or snapped roofs, with an optional lower drum and a corbelled
balcony.

![Round spires, filled style](docs/round_spire_filled.png)
![Round spires, line style](docs/round_spire_line.png)

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python generate.py                 # every type, 12 variants each
.venv/bin/python generate.py --style line    # the same, as pure line art
```

Then copy the pack folder from `out/` (for example `out/Inkwright/`) into Wonderdraft's
`assets` folder and restart Wonderdraft.

## Options

| Option | Default | Meaning |
|---|---|---|
| `-t`, `--types` | all | Comma list of type names (`cottage,round_spire`) or category words (`land`, `cave`, `divine`) |
| `-n`, `--count` | 12 | Variants per type |
| `--seed` | 1 | Change it for a fresh batch. The same seed always redraws the same symbols |
| `--style` | `filled` | `filled`: ink with white paper inside shapes, so symbols hide roads and each other. `line`: ink only, transparent inside |
| `--res` | 3 | Output pixels per drawing unit (higher = bigger, crisper PNGs) |
| `--ink` | `1e1e1e` | Ink colour (hex) |
| `--wash` | `96623e` | Colour of the cliff/canyon shading (hex), or `none` for ink only |
| `--draw-mode` | `sample_color` | Wonderdraft `draw_mode` written to `.wonderdraft_symbols`: `sample_color` tints the white fill to the land colour underneath, `normal` draws as-is |
| `--paper` | `ffffff` | Fill colour inside shapes (filled style) |
| `--refresh-meta` | | Only rewrite `.wonderdraft_symbols` for the PNGs already in `out/` |
| `--pack` | `Inkwright` | Pack name, used for the pack folder and the palette folders (the line style adds `Line`) |

File names include the type, seed and index (`cottage_1_007.png`), so you can always regenerate a
symbol you like.

## Output and previews

```
out/Inkwright/sprites/symbols/Inkwright - <Group>/<type>_<seed>_<n>.png
out/Inkwright/sprites/symbols/Inkwright - <Group>/.wonderdraft_symbols
out/InkwrightLine/sprites/symbols/Inkwright Line - <Group>/...       line style
preview/<Group>/<Category>_<style>.png      contact sheet per category
```

All categories of a group (Settlements, Ruins, Lairs, Divine, Land) share one folder, so
Wonderdraft shows one palette per group, named like `Inkwright - Land`. The
file names start with their type (`hill_cave_…`, `burrow_…`). Each group folder gets a
`.wonderdraft_symbols` metadata file with an entry per image:

- **Pivot** (`offset_x`, `offset_y`): near the bottom of the drawing, at the front ground point,
  like Wonderdraft's own sprites. Wonderdraft draws the symbol with the lower pivot in front, so
  these layer correctly against each other and against the default sprites. Cliff and canyon
  pieces pivot on their centre.
- **Radius**: estimated from the drawn width. It's smaller for cliff and canyon pieces so they can
  overlap.
- **draw_mode**: from `--draw-mode`, `sample_color` by default.

To get symbols that look like Wonderdraft's own line-art sprites, use the **filled** style with
`sample_color`. The white fill takes the map colour underneath, so a house turns green on grass,
yet it is still opaque: it hides the road behind it, and houses overlap each other cleanly. The
**line** style is transparent inside, so anything behind it always shows through. `custom_colors`
colourises through the image's RGB channels, so it doesn't suit plain black ink.

Re-running a type into the same folder updates its entries and keeps the others.
`generate.py --refresh-meta` rewrites the metadata for every PNG already in `out/` without
re-rendering. Add `--draw-mode` to change the draw mode at the same time.

Every run writes a contact sheet for each category it generated, so previewing a category is just
generating it, e.g. `generate.py -t robed_statue -n 18`. The `#003` labels match the file names.
Categories made of several piece types (like the cliff curves) get one labelled row per type.

## Categories

Buildings, lairs and monuments are drawn in the same 3/4 top-down view as Wonderdraft's default
houses, lit from the west, so east-facing walls and slopes are hatched into shadow.

**Settlements**
- **Houses**: cottages, two-storey houses, longhouses and barns. Gables face front or side.
  Variants add timber framing, chimneys, dormers, lean-tos, and tiled or thatched roofs.
- **Hamlets**: 3–4 houses packed closely, sometimes with a well or a fence.
- **Villages**: 6–12 houses along a winding road. Some have a chapel with a buttressed bell tower,
  a well, garden fences and trees.
- Houses and hamlets share one scale and are cropped tightly with no stray ground clutter, so you
  can place them close together to build up a town. Some come with one or two trees (spruce,
  cypress or a broadleaf oak).

**Ruins**
- **Cottages** and ruined **Villages**: stepped masonry breaches, partial roofs or bare rafters,
  standing chimneys, ivy, rubble.
- **Round Towers** and square **Keeps**.
- **Castles**: curtain walls with battlements, corner towers and turrets, a twin-towered gatehouse
  and a central keep.
- **Round Spires**: wizard towers with corbelled balconies and cone roofs (intact, caved in or
  snapped).
- **Gothic Spires**: buttressed towers with lancet windows and a pyramid spire.

**Lairs**
- Four kinds of cave, each with a dark, jagged entrance:
  - **Hill Caves**: a craggy hillside, with trees and a trail leading in.
  - **Cliff Caves**: a cavern under a headland bluff.
  - **Boulder Caves**: a grotto of tumbled megaliths under a capstone.
  - **Sunken Entrances**: a barrow or stone stairway leading down below ground.
- **Burrows** and **Warrens**: pits dug down into an earth bank, either a single hollow or a
  whole colony.
- **Crag Nests** and **Ground Nests**: woven twig eyries holding speckled eggs or bones.
- Lairs scatter bones, skulls, ribcages, claw marks, tracks and skulls on stakes.

**Divine**
- **Temples**: Greek-style ruins on stepped platforms. Fluted columns (standing, snapped or
  stumps), entablatures and pediments, multi-room inner halls, altars and fallen drums. Some
  are forecourt sanctuaries or L-shaped complexes.
- **Stone Circles**: rough-hewn standing stones, trilithon gates, an altar and a worn ring track.
- **Colossi**: sculpted legs on a plinth, with a shattered stone face half-sunk in the earth.
- **Robed Statues**: hooded figures in different poses on square, octagonal or round pedestals.
  They hold staffs, greatswords, orbs, tomes or flames; some have halos, some are headless.
- **Swords**: a giant blade planted in a cracked crater or a boulder.
- **Obelisks**: glyph-carved, cracked or broken, sometimes with vines.

**Land**
- **Fields**: patchworks of furrows, grain, orchards, vineyards, fallow land and pasture with
  haystacks. Edges are hedges, walls, fences or tracks.
- **Cliffs**, **Steep Cliffs**, **Cliff Curves**, **Cliff Bays**, **Mesas**, **Terraced Hills**,
  **Canyons** and **Canyon Bends**: see below.

## Cliffs, canyons and mountains

Land edges are open lines: an ink edge, short strokes hanging below it where the cliff face is
visible, and a fading wash. All pieces share one face-height range and have tapered, open ends,
so you build longer edges by overlapping pieces. You never need to rotate a piece.

Every edge knows which side is the low ground. The face only appears where the drop faces the
viewer, and it fades out as the edge turns sideways. So:

- **Cliff Curves** has one piece for each part of a plateau outline: `curve_front`,
  `curve_back`, `curve_left_end`, `curve_right_end` and the four corners. A front piece, two ends
  and a back piece make a closed ring.
- **Cliff Bays** are the inner (concave) versions, for notches in an outline.
- **Mesas** are a complete plateau in one symbol. **Terraced Hills** stack 2–4 of them into a
  whole hill or mountain.
- **Canyons** and **Canyon Bends** follow the same rules, so bends join straight runs.

The wash is a second colour. These symbols won't tint as one colour unless you use `--wash none`.

## Project layout

| File | Contents |
|---|---|
| `generate.py` | CLI and the list of types (`TYPES`: category, draw function, canvas size) |
| `wdgen/ink.py` | Canvas (ink, paper and wash layers) and the brush: tapered strokes, hatching, tufts, scalloped bushes, stones |
| `wdgen/proj.py` | The 3/4 orthographic projection shared by every 3D symbol |
| `wdgen/solids.py` | Boxes, cylinders, cones, pyramids, broken columns, earth mounds |
| `wdgen/houses.py` | Intact houses (gable roofs, timber framing, dormers, lean-tos, chimneys), hamlets, villages, chapel, well |
| `wdgen/flora.py` | Trees: spruce, cypress, broadleaf oak |
| `wdgen/ruins.py` | Damaged walls and roofs, ivy, cottages, round towers, keeps, ruined villages |
| `wdgen/castles.py` | Crenellated curtain walls, turrets, gatehouses, castles, round and gothic spires |
| `wdgen/lairs.py` | 3D hills, cave portals, boulders, trails, burrow pits, nests, bones and other remains |
| `wdgen/divine.py` | Temples (stylobates, Doric columns, entablatures, halls), stone circles, colossi, robed statues and their props, swords, obelisks |
| `wdgen/terrain.py` | Fields, cliff edges, curves, mesas, terraced hills, canyons |

To add a symbol type, write a `draw(pen, P, rng)` function (`P` projects world `(x, y, height)` to
the canvas, and `pen` draws strokes) and add it to `TYPES` in `generate.py`.
