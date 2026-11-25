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

    # Crisp outer bough strokes along both edges so overlapping brushed pines separate cleanly!
    pen.stroke([(x + lean, top - 0.8)] + left[1::2], W0 * 0.52, taper=(0.25, 0.35), wobble=0.14)
    pen.stroke([(x + lean, top - 0.8)] + right[1::2], W0 * 0.68, taper=(0.25, 0.25), wobble=0.14)
    pen.stroke([(x - w * 0.36, y + 0.2), (x + w * 0.42, y + 0.2)], W0 * 0.50, taper=(0.40, 0.40), wobble=0.1)


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
    sh = clip_halfplane(sil, np.array([1.0, 0.08]), x + lean * 0.4 + w * 0.14)
    if len(sh) >= 3:
        pen.hatch(sh, 85, 1.52, W0 * 0.60, coverage=(0.65, 1.0))

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


def palm(pen, x, y, h, lean_dir=None):
    """Gracefully curved ringed palm tree with arched sweeping feather-fronds (oasis/tropical brush)."""
    rng, W0 = pen.rng, pen.width
    side = int(rng.choice([-1, 1])) if lean_dir is None else int(lean_dir)
    bend = side * h * rng.uniform(0.18, 0.34)
    n = 11
    ts = np.linspace(0, 1, n)
    # Smooth curved trunk spine rising from root foot (x, y) to crown hub (cx, cy)
    xs = x + bend * (ts ** 1.35) + side * h * 0.04 * np.sin(math.pi * ts)
    ys = y - h * 0.84 * ts
    # Slender tapered trunk polygon (opaque so foreground palms cleanly mask rear palms!)
    l_edge, r_edge = [], []
    for i, t in enumerate(ts):
        hw = (1.65 - 0.90 * t) * (h / 42.0)
        l_edge.append((xs[i] - hw, ys[i]))
        r_edge.append((xs[i] + hw, ys[i]))
    trunk_poly = l_edge + r_edge[::-1]
    pen.cv.erase(trunk_poly)
    # Delicate palm-bark ring ticks along the shadowed east/mid trunk (avoids heavy ladder rungs)
    for i in range(1, n - 1):
        lx, ly = l_edge[i]
        rx, ry = r_edge[i]
        pen.stroke([(lx + (rx - lx) * 0.32, ly + 0.3), (rx, ry)], W0 * 0.34, taper=(0.3, 0.1), wobble=0.05)
    pen.stroke(l_edge, W0 * 0.72, taper=(0.05, 0.25), wobble=0.08)
    pen.stroke(r_edge, W0 * 0.96, taper=(0.05, 0.20), wobble=0.08)
    pen.stroke([(x - 2.4, y + 0.2), (x + 2.6, y + 0.2)], W0 * 0.48, taper=(0.4, 0.4), wobble=0.08)

    # Crown hub + 7-8 sweeping arched feather fronds radiating outward & drooping gracefully
    cx, cy = float(xs[-1]), float(ys[-1])
    frond_r = h * rng.uniform(0.30, 0.38)
    # Small opaque coconut/crown core
    core = [(cx + math.cos(a) * 2.1, cy + math.sin(a) * 1.8) for a in np.linspace(0, TAU, 10, endpoint=False)]
    pen.cv.erase(core)
    pen.closed(core, W0 * 0.56, wobble=0.1)

    angles = [-2.60, -2.10, -1.58, -1.06, -0.54, -0.05, 0.34]
    if rng.random() < 0.60:
        angles.append(-2.98)
    for ang_base in angles:
        ang = ang_base + rng.normal(0, 0.13)
        L = frond_r * rng.uniform(0.84, 1.18)
        # Frond shoots outward along ang, arching gently downward under gravity
        p0 = (cx, cy)
        p1 = (cx + math.cos(ang) * L * 0.52, cy + math.sin(ang) * L * 0.52 - L * 0.20)
        p2 = (cx + math.cos(ang) * L, cy + math.sin(ang) * L + L * 0.28)
        spine = chaikin([p0, p1, p2], 2)
        # Razor-tight narrow frond mask so fronds layer cleanly without thick white halos
        f_top = [(p[0], p[1] - 0.65) for p in spine]
        f_bot = [(p[0], p[1] + 1.45) for p in spine[::-1]]
        pen.cv.erase(f_top + f_bot)
        east_sh = math.cos(ang) > 0.05
        pen.stroke(spine, W0 * (0.80 if east_sh else 0.66), taper=(0.05, 0.84), wobble=0.08)
        # Hanging feather leaflets along the arched rachis
        for k in range(1, len(spine) - 1):
            px, py = spine[k]
            f = k / (len(spine) - 1)
            ll = L * 0.28 * math.sin(math.pi * f) * rng.uniform(0.82, 1.16)
            if ll < 1.1:
                continue
            dx = math.copysign(ll * 0.30, math.cos(ang))
            pen.stroke([(px, py), (px + dx, py + ll)], W0 * (0.56 if east_sh else 0.42),
                       taper=(0.08, 0.88), wobble=0.06)


def deadwood(pen, x, y, h):
    """Twisted gnarly deadwood oak or lightning-struck conifer snag with tapered bark trunk & bare boughs."""
    rng, W0 = pen.rng, pen.width
    is_pine = rng.random() < 0.36
    lean = rng.normal(0, h * 0.09)

    if is_pine:
        # Skeletal blasted conifer snag: tall tapering trunk with snapped crown & crooked dead stubs
        n = 7
        ts = np.linspace(0, 1, n)
        xs = x + lean * (ts ** 1.1) + rng.normal(0, 0.55, n) * ts
        ys = y - h * ts
        l_e = [(xs[i] - (2.0 - 1.55 * t), ys[i]) for i, t in enumerate(ts)]
        r_e = [(xs[i] + (2.0 - 1.55 * t), ys[i]) for i, t in enumerate(ts)]
        pen.cv.erase(l_e + r_e[::-1])
        # Subtle staggered east-side bark ticks
        for k in range(1, n - 2):
            bx, by = xs[k] + (0.6 - 0.3 * ts[k]), ys[k]
            pen.stroke([(bx, by + 1.8), (bx + 0.2, by - 2.0)], W0 * 0.38, taper=(0.3, 0.3))
        pen.stroke(l_e, W0 * 0.72, taper=(0.05, 0.65), wobble=0.12)
        pen.stroke(r_e, W0 * 0.96, taper=(0.05, 0.55), wobble=0.12)
        # Crooked bare dead limbs poking sideways & drooping slightly
        for f in np.linspace(0.28, 0.88, int(rng.integers(5, 8))):
            bx = float(np.interp(f, ts, xs))
            by = float(np.interp(f, ts, ys))
            for sgn in (-1, 1):
                if rng.random() < 0.68:
                    bl = h * (0.28 - 0.16 * f) * rng.uniform(0.7, 1.25)
                    ang = (0.22 if sgn > 0 else math.pi - 0.22) + rng.normal(0, 0.25)
                    p1 = (bx + math.cos(ang) * bl * 0.55, by + math.sin(ang) * bl * 0.55)
                    p2 = (bx + math.cos(ang) * bl, by + math.sin(ang) * bl + rng.uniform(-2, 3))
                    pen.stroke([(bx, by), p1, p2], W0 * (0.64 if sgn > 0 else 0.52), taper=(0.08, 0.88), wobble=0.12)
                    if rng.random() < 0.55:
                        pen.stroke([p1, (p1[0] + sgn * bl * 0.3, p1[1] - rng.uniform(1.5, 4.0))],
                                   W0 * 0.40, taper=(0.1, 0.9))
        pen.stroke([(x - 2.8, y + 0.2), (x + 3.0, y + 0.2)], W0 * 0.46, taper=(0.4, 0.4))
        return

    # Twisted gnarled dead broadleaf oak: stout curved trunk splitting into crooked recursive bare branches!
    tr_h = h * rng.uniform(0.40, 0.54)
    mx, my = x + lean * 0.55, y - tr_h
    mid_x, mid_y = x + lean * 0.25 + rng.normal(0, 1.5), y - tr_h * 0.52
    trunk_l = [(x - 2.4, y + 0.3), (mid_x - 1.5, mid_y), (mx - 1.1, my)]
    trunk_r = [(x + 2.4, y + 0.3), (mid_x + 1.5, mid_y), (mx + 1.1, my)]
    pen.cv.erase(trunk_l + trunk_r[::-1])
    # Delicate staggered vertical bark shading ticks along the east half of the trunk (avoids harsh zebra stripe!)
    for f0, f1 in ((0.06, 0.40), (0.34, 0.68), (0.58, 0.92)):
        ax = x + (mx - x) * f0 + rng.uniform(0.3, 0.9)
        ay = y - tr_h * f0
        bx = x + (mx - x) * f1 + rng.uniform(0.2, 0.7)
        by = y - tr_h * f1
        pen.stroke([(ax, ay), (bx, by)], W0 * 0.42, taper=(0.30, 0.35), wobble=0.08)
    pen.stroke(trunk_l, W0 * 0.76, taper=(0.05, 0.35), smooth=1, wobble=0.14)
    pen.stroke(trunk_r, W0 * 0.96, taper=(0.05, 0.30), smooth=1, wobble=0.14)
    # Root flare feet
    pen.stroke([(x - 1.7, y - 2.0), (x - 4.0, y + 0.5)], W0 * 0.58, taper=(0.1, 0.85))
    pen.stroke([(x + 1.7, y - 2.0), (x + 4.2, y + 0.5)], W0 * 0.74, taper=(0.1, 0.85))

    def _limb(x0, y0, ang, L, w_s, depth):
        if depth <= 0 or L < 3.0:
            return
        a1 = ang + rng.normal(0, 0.22)
        p1 = (x0 + math.cos(a1) * L * 0.52, y0 + math.sin(a1) * L * 0.52)
        a2 = a1 + rng.normal(0, 0.28)
        p2 = (p1[0] + math.cos(a2) * L * 0.48, p1[1] + math.sin(a2) * L * 0.48)
        pen.stroke([(x0, y0), p1, p2], W0 * w_s, taper=(0.05, 0.72), smooth=1, wobble=0.14)
        if depth > 1:
            n_ch = 2 if depth >= 2 else 1
            for sgn in (-1, 1)[:n_ch]:
                _limb(p1[0], p1[1], a1 + sgn * rng.uniform(0.35, 0.72), L * rng.uniform(0.55, 0.74), w_s * 0.68, depth - 1)
            _limb(p2[0], p2[1], a2 + rng.normal(0, 0.38), L * rng.uniform(0.52, 0.72), w_s * 0.66, depth - 1)

    n_main = int(rng.integers(3, 5))
    for i in range(n_main):
        f = i / max(1, n_main - 1)
        ang = -math.pi * (0.82 - 0.64 * f) + rng.normal(0, 0.12)
        _limb(mx + (f - 0.5) * 1.8, my, ang, h * rng.uniform(0.28, 0.42), 0.78 if f > 0.5 else 0.64, 3)


def tree(pen, x, y, h=None, kind=None):
    rng = pen.rng
    kind = kind or str(rng.choice(["spruce", "cypress", "round"], p=[0.4, 0.28, 0.32]))
    h = h or {"spruce": rng.uniform(32, 46), "cypress": rng.uniform(30, 44), "round": rng.uniform(26, 36)}[kind]
    {"spruce": spruce, "cypress": cypress, "round": round_tree}[kind](pen, x, y, h)


# --------------------------------------------------------------------------- Wonderdraft Tree Brush pool generators
# Strictly single trees (no pre-grouped multi-tree clusters) so Wonderdraft's Tree Brush packs forests naturally.

def pine_brush(pen, P, rng):
    """Wonderdraft Tree Brush pool: single alpine spruce/pine trees with organic height & bough variety."""
    spruce(pen, P.ox, P.oy, rng.uniform(32, 54))


def broadleaf_brush(pen, P, rng):
    """Wonderdraft Tree Brush pool: single billowy oak/broadleaf woodland trees."""
    round_tree(pen, P.ox, P.oy, rng.uniform(28, 44))


def cypress_brush(pen, P, rng):
    """Wonderdraft Tree Brush pool: single slender flame cypress trees."""
    cypress(pen, P.ox, P.oy, rng.uniform(34, 52))


def palm_brush(pen, P, rng):
    """Wonderdraft Tree Brush pool: single gracefully curved oasis/tropical palm trees."""
    palm(pen, P.ox, P.oy, rng.uniform(36, 52))


def deadwood_brush(pen, P, rng):
    """Wonderdraft Tree Brush pool: single twisted dead oaks & blasted conifer snags."""
    deadwood(pen, P.ox, P.oy, rng.uniform(32, 48))
