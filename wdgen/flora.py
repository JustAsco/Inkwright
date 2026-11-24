"""Hand-inked fantasy map trees: spruce, cypress, round deciduous tree. (x, y) is the trunk foot."""
import math

import numpy as np

from .ink import TAU, chaikin, clip_halfplane


def spruce(pen, x, y, h):
    """Layered evergreen conifer with drooping irregular boughs and east-side shadow."""
    rng, W0 = pen.rng, pen.width
    w = h * rng.uniform(0.26, 0.34)
    lean = rng.normal(0, h * 0.03)
    top = y - h
    trunk_h = h * rng.uniform(0.10, 0.16)
    tiers = max(5, int(h / 4.8))

    # Build a natural jagged tiered evergreen silhouette
    right, left = [(x + lean, top - 0.8)], []
    for i in range(tiers):
        t = (i + 0.85) / tiers
        yy = top + t * (h - trunk_h)
        hw = w * (t ** 0.88)
        cx = x + lean * (1 - t)
        # outer bough tip + tucked inner notch beneath it
        rw = hw * rng.uniform(0.88, 1.14)
        lw = hw * rng.uniform(0.88, 1.14)
        right.append((cx + rw, yy + rng.uniform(0.6, 1.8)))
        if i < tiers - 1:
            right.append((cx + rw * rng.uniform(0.55, 0.72), yy + 0.5))
        left.append((cx - lw, yy + rng.uniform(0.6, 1.8)))
        if i < tiers - 1:
            left.append((cx - lw * rng.uniform(0.55, 0.72), yy + 0.5))

    sil = right + [(x, y - trunk_h * 0.7)] + left[::-1]
    pen.cv.erase(sil)

    # Short rooted trunk
    pen.stroke([(x - 0.5, y), (x + lean * 0.15, y - trunk_h - 1.5)], W0 * 1.0, taper=(0.0, 0.4), wobble=0.1)
    pen.stroke([(x + 0.6, y), (x + 0.3, y - trunk_h)], W0 * 0.7, taper=(0.0, 0.5), wobble=0.1)
    # Central spire leader near top
    pen.stroke([(x + lean * 0.7, top + h * 0.3), (x + lean, top - 0.6)], W0 * 0.55, taper=(0.2, 0.85), wobble=0.1)

    # Tiered drooping needle strokes — fuller & darker on the east/right shadow side
    rows = max(5, int((h - trunk_h) / 2.6))
    for r in range(rows):
        t = (r + 0.55) / rows
        yy = top + t * (h - trunk_h)
        cx = x + lean * (1 - t)
        half = w * (t ** 0.88)
        for side in (-1, 1):
            n = max(1, int(half / (2.0 if side > 0 else 2.6)))
            for j in range(n):
                f = (j + rng.uniform(0.25, 0.92)) / n
                if side < 0 and f < 0.45 and rng.random() < 0.38:
                    continue  # leave crisp sunlit highlight on upper-left flank
                bx = cx + side * f * half
                by = yy + f * 1.4 + rng.normal(0, 0.35)
                sz = rng.uniform(1.3, 2.2) * (1.08 if side > 0 else 0.92)
                pen.dab(bx, by, sz, angle=side * 0.38 + rng.normal(0, 0.12),
                        w=W0 * (1.0 if side > 0 else 0.72))

    # Crisp outer bough strokes along the shadowed right edge & lower skirt
    pen.stroke([(x - w * 0.4, y + 0.2), (x + w * 0.45, y + 0.2)], W0 * 0.55, taper=(0.35, 0.35), wobble=0.1)


def cypress(pen, x, y, h):
    """Slender flame-shaped Mediterranean/fantasy cypress with east-side shadow hatching."""
    rng, W0 = pen.rng, pen.width
    w = h * rng.uniform(0.11, 0.15)
    lean = rng.normal(0, h * 0.04)
    trunk_h = h * rng.uniform(0.10, 0.15)
    top = y - h
    ts = np.linspace(0, 1, 14)

    def half_w(t):
        # slender flame silhouette: sharp tip, fullest around lower-middle, tucks into trunk
        return w * (math.sin(math.pi * (t ** 0.72)) ** 0.85)

    right, left = [(x + lean, top)], []
    for t in ts[1:]:
        yy = top + t * (h - trunk_h)
        cx = x + lean * (1 - t)
        hw = half_w(t)
        right.append((cx + hw * rng.uniform(0.90, 1.12), yy))
        left.append((cx - hw * rng.uniform(0.90, 1.12), yy))
    sil = chaikin(right + [(x, y - trunk_h * 0.85)] + left[::-1] + [right[0]], 1)[:-1]
    pen.cv.erase(sil)

    # Trunk base
    pen.stroke([(x, y), (x, y - trunk_h - 1.5)], W0 * 0.95, taper=(0.0, 0.35), wobble=0.08)

    # Vertical-angled dark shadow strokes on the east/right half of the cypress
    sh = clip_halfplane(sil, np.array([1.0, 0.08]), x + lean * 0.4 + w * 0.05)
    if len(sh) >= 3:
        pen.hatch(sh, 85, 1.35, W0 * 0.62, coverage=(0.65, 1.0))

    # Fine vertical organic foliage ticks along the lit left & center
    rows = max(5, int((h - trunk_h) / 3.4))
    for r in range(rows):
        t = (r + 0.5) / rows
        hw = half_w(t)
        if hw < 1.2:
            continue
        yy = top + t * (h - trunk_h)
        cx = x + lean * (1 - t)
        for _ in range(max(1, int(hw / 2.2))):
            xx = cx + rng.uniform(-hw * 0.75, hw * 0.35)
            L = rng.uniform(1.8, 3.4)
            pen.stroke([(xx, yy - L * 0.5), (xx + rng.normal(0, 0.3), yy + L * 0.5)],
                       W0 * 0.48, taper=(0.35, 0.35), wobble=0.1)

    # Crisp hand-drawn outer contour (heavier on shadow side)
    pen.stroke(chaikin(right, 1), W0 * 0.92, taper=(0.25, 0.08), wobble=0.18)
    pen.stroke(chaikin([right[0]] + left, 1), W0 * 0.72, taper=(0.25, 0.12), wobble=0.18)
    pen.stroke([(x - w * 0.5, y + 0.2), (x + w * 0.55, y + 0.2)], W0 * 0.5, taper=(0.35, 0.35), wobble=0.1)


def round_tree(pen, x, y, h):
    """Multi-lobed broadleaf/oak tree with forked trunk, billowy scalloped crown, and east shadow."""
    rng, W0 = pen.rng, pen.width
    r = h * rng.uniform(0.30, 0.38)
    trunk = h - r * 1.45
    lean = rng.normal(0, h * 0.05)
    cx, cy = x + lean, y - trunk - r * 0.62

    # 3-4 overlapping organic crown sub-lobes merged into one billowy canopy silhouette
    lobes = [(cx, cy - r * 0.15, r * 0.85, r * 0.76)]
    lobes.append((cx - r * rng.uniform(0.45, 0.62), cy + r * rng.uniform(0.05, 0.24), r * 0.62, r * 0.56))
    lobes.append((cx + r * rng.uniform(0.45, 0.62), cy + r * rng.uniform(0.02, 0.22), r * 0.64, r * 0.58))
    if rng.random() < 0.65:
        lobes.append((cx + rng.uniform(-0.25, 0.25) * r, cy - r * rng.uniform(0.42, 0.58), r * 0.58, r * 0.50))

    # Sample the outer envelope of the union of lobes in polar coordinates around (cx, cy)
    n_ang = 44
    ts = np.linspace(0, TAU, n_ang, endpoint=False)
    pts = []
    for t in ts:
        ct, st = math.cos(t), math.sin(t)
        best = 0.0
        for lx, ly, lrx, lry in lobes:
            # ray from (cx, cy) along (ct, st) intersecting ellipse (lx, ly, lrx, lry)
            dx, dy = cx - lx, cy - ly
            A = (ct / lrx) ** 2 + (st / lry) ** 2
            B = 2 * (dx * ct / (lrx * lrx) + dy * st / (lry * lry))
            C = (dx / lrx) ** 2 + (dy / lry) ** 2 - 1.0
            disc = B * B - 4 * A * C
            if disc >= 0:
                dist = (-B + math.sqrt(disc)) / (2 * A)
                if dist > best:
                    best = dist
        # subtle fine foliage scalloping
        best *= 1.0 + 0.055 * math.sin(9 * t + rng.uniform(-0.2, 0.2))
        pts.append((cx + best * ct, cy + best * st))
    crown = chaikin(pts + [pts[0]], 1)[:-1]

    # Tapered trunk with small rooted flare and forked boughs entering the canopy
    tx_top, ty_top = cx - lean * 0.15, y - trunk
    trunk_poly = [
        (x - 1.5, y + 0.3), (x - 0.85, y - trunk * 0.5), (tx_top - 0.8, ty_top - 1.5),
        (tx_top + 0.8, ty_top - 1.5), (x + 0.95, y - trunk * 0.5), (x + 1.6, y + 0.3),
    ]
    pen.cv.erase(trunk_poly)
    pen.stroke([(x - 0.9, y + 0.2), (x - 0.5, y - trunk * 0.55), (tx_top - 1.8, ty_top - 3.2)],
               W0 * 0.72, taper=(0.0, 0.4), smooth=1, wobble=0.1)
    pen.stroke([(x + 0.9, y + 0.2), (x + 0.6, y - trunk * 0.55), (tx_top + 2.0, ty_top - 2.8)],
               W0 * 0.95, taper=(0.0, 0.4), smooth=1, wobble=0.1)
    pen.stroke([(x - 2.2, y + 0.3), (x + 2.4, y + 0.3)], W0 * 0.55, taper=(0.35, 0.35), wobble=0.1)

    # Erase & shade the crown
    pen.cv.erase(crown)

    # Rich crescent foliage clusters along the lower-right shadowed hemisphere & lobe undersides
    for _ in range(int(r * 3.4)):
        a = rng.uniform(0, TAU)
        rr = math.sqrt(rng.uniform(0.08, 0.90))
        px, py = cx + math.cos(a) * r * 1.02 * rr, cy + math.sin(a) * r * 0.86 * rr
        shade_w = (px - cx) / max(1.0, r) * 0.65 + (py - cy) / max(1.0, r) * 0.75
        if shade_w < -0.20 and rng.random() < 0.76:
            continue  # preserve sunlit upper-left crown highlights
        sz = rng.uniform(1.5, 2.5) * (1.12 if shade_w > 0.25 else 0.88)
        pen.dab(px, py, sz, angle=rng.normal(0.2 if shade_w > 0 else -0.1, 0.3),
                w=W0 * (1.02 if shade_w > 0.32 else 0.75 if shade_w > 0 else 0.55))

    # Crisp shaded crescent hatching along the deepest lower-right crown edge
    sh = clip_halfplane(crown, np.array([0.75, 0.66]), cx * 0.75 + cy * 0.66 + r * 0.38)
    if len(sh) >= 3:
        pen.hatch(sh, 74, 1.8, W0 * 0.46, coverage=(0.45, 0.85))

    # Scalloped hand-inked crown outline (heavier on lower/right shadow side, lighter on sunlit top-left)
    pen.closed(crown, W0 * 0.85, wobble=0.15)


def tree(pen, x, y, h=None, kind=None):
    rng = pen.rng
    kind = kind or str(rng.choice(["spruce", "cypress", "round"], p=[0.4, 0.28, 0.32]))
    h = h or {"spruce": rng.uniform(32, 46), "cypress": rng.uniform(30, 44), "round": rng.uniform(26, 36)}[kind]
    {"spruce": spruce, "cypress": cypress, "round": round_tree}[kind](pen, x, y, h)
