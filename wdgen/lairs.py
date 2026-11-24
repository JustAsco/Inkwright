"""Beast lairs: caves in rocky outcrops, burrows, nests. Drawn directly in screen space."""
import math

import numpy as np

from .ink import TAU, blob_pts, chaikin, clip_halfplane, ellipse_pts
from .solids import mound


def _inside(poly, x, y):
    poly = np.asarray(poly)
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _height_at(pts, x, by):
    return by - float(np.interp(x, pts[:, 0], pts[:, 1]))


def hill_pts(rng, cx, by, Bw, Hh, m=None, rocky=True):
    """Silhouette from left base to right base (screen coords, y down)."""
    m = m or int(rng.integers(7, 11))
    xs = np.linspace(-Bw / 2, Bw / 2, m)
    xs[1:-1] += rng.normal(0, Bw / m * 0.15, m - 2)
    env = np.clip(np.sin(np.pi * (xs / Bw + 0.5)), 0, None) ** rng.uniform(0.5, 1.0)
    hs = Hh * env * (1 + rng.normal(0, 0.12 if rocky else 0.05, m))
    hs[0] = hs[-1] = 0
    pts = np.stack([cx + xs, by - hs], 1)
    if rocky:  # extra angular facet points
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            if rng.random() < 0.6:
                mid = (a + b) / 2 + rng.normal(0, 2.2, 2)
                out.append(mid)
            out.append(b)
        return np.array(out)
    return chaikin(pts, 3)


def base_curve(rng, x0, x1, by, sag=3.0, n=9):
    """Gently bowed ground contour from x1 back to x0 (closes a silhouette)."""
    xs = np.linspace(x1, x0, n)
    t = (xs - x0) / max(1e-6, x1 - x0)
    ys = by + sag * np.sin(np.pi * t) + rng.normal(0, 0.6, n)
    ys[0] = ys[-1] = by
    return np.stack([xs, ys], 1)


def ground_line(pen, base):
    """Broken, tapered strokes along a base contour plus a few stray ground marks."""
    rng, W0 = pen.rng, pen.width
    n = len(base)
    i = 0
    while i < n - 1:
        j = min(n, i + int(rng.integers(3, 6)))
        if rng.random() < 0.8:
            pen.stroke(base[i:j], W0 * 0.9, taper=(0.3, 0.3), wobble=0.3)
        i = j - 1 if rng.random() < 0.7 else j
    for x0, sgn in ((base[-1][0], -1), (base[0][0], 1)):
        for _ in range(rng.integers(0, 3)):
            x = x0 + sgn * rng.uniform(2, 14)
            y = base[0][1] + rng.uniform(-1, 3)
            pen.stroke([(x, y), (x + sgn * rng.uniform(4, 9), y + rng.normal(0, 0.3))], W0 * 0.55, taper=(0.4, 0.4))


def rock(pen, pts, shade=True, facets=True, base_sag=None):
    rng, W0 = pen.rng, pen.width
    pts = np.asarray(pts, float)
    base = base_curve(rng, pts[0][0], pts[-1][0], pts[0][1], sag=base_sag or rng.uniform(2, 5), n=15)
    poly = list(pts) + list(base[1:-1])
    pen.cv.erase(poly)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    if shade:
        split = xmin + (xmax - xmin) * rng.uniform(0.55, 0.7)
        pen.hatch(clip_halfplane(poly, (1, 0), split), rng.uniform(55, 70), rng.uniform(2.0, 2.8),
                  W0 * 0.6, coverage=(0.5, 1.0))
    if facets:
        idx = rng.choice(np.arange(1, len(pts) - 1), size=min(len(pts) - 2, int(rng.integers(2, 6))), replace=False)
        for i in idx:
            x, y = pts[i]
            L = rng.uniform(8, 22)
            ang = math.pi / 2 + rng.normal(0, 0.5)
            path = [(x, y + 1.5),
                    (x + math.cos(ang) * L * 0.5 + rng.normal(0, 1.5), y + math.sin(ang) * L * 0.5),
                    (x + math.cos(ang) * L, y + math.sin(ang) * L)]
            if _inside(poly, *path[-1]):
                pen.stroke(path, W0 * 0.7, taper=(0.1, 0.8))
        for _ in range(rng.integers(1, 4)):
            x = rng.uniform(xmin + 10, xmax - 10)
            y = float(np.interp(x, pts[:, 0], pts[:, 1])) + rng.uniform(8, 25)
            L = rng.uniform(6, 14)
            if _inside(poly, x, y) and _inside(poly, x + L, y):
                pen.stroke([(x, y), (x + L, y + rng.normal(0, 1))], W0 * 0.6, taper=(0.3, 0.5))
    pen.stroke(pts, W0 * 1.25, taper=(0.05, 0.05), wobble=0.3)
    ground_line(pen, base)
    if rng.random() < 0.5:
        i0 = int(rng.integers(0, len(pts) // 2))
        pen.stroke(pts[i0:i0 + len(pts) // 2] + rng.normal(0, 0.6, 2), W0 * 0.5, taper=(0.3, 0.3))


def boulder(pen, x, y, r):
    rng, W0 = pen.rng, pen.width
    pts = blob_pts(rng, x, y - r * 0.6, r, n=9, irregular=0.18, squash=0.7)
    pen.cv.erase(pts)
    pen.hatch(clip_halfplane(list(pts), (1, 0), x + r * 0.15), 60, 2.0, W0 * 0.55, coverage=(0.5, 1.0))
    pen.closed(pts, W0 * 0.9, overshoot=0)
    if rng.random() < 0.5:
        pen.stroke([(x - r * 0.3, y - r * 1.0), (x - r * 0.1, y - r * 0.6), (x - r * 0.25, y - r * 0.2)],
                   W0 * 0.5, taper=(0.2, 0.6))


# --------------------------------------------------------------------------- remains

def bone(pen, x, y, L, ang):
    cv, W0 = pen.cv, pen.width
    kr = max(1.0, L * 0.12)
    d = np.array([math.cos(ang), math.sin(ang) * 0.6])
    d /= np.hypot(*d)
    n = np.array([-d[1], d[0]])
    c = np.array([x, y])
    e1, e2 = c - d * L / 2, c + d * L / 2
    hw = kr * 0.55
    lobes = [e + n * kr * 0.75 * s for e in (e1, e2) for s in (-1, 1)]
    for p in lobes:
        cv.erase(ellipse_pts(p[0], p[1], kr, kr, 12))
    for p in lobes:
        pen.closed(ellipse_pts(p[0], p[1], kr, kr, 12), W0 * 0.5, wobble=0)
    cv.erase([e1 + n * hw, e2 + n * hw, e2 - n * hw, e1 - n * hw])
    pen.line(e1 + n * hw, e2 + n * hw, W0 * 0.5, sketch=0, overshoot=0)
    pen.line(e1 - n * hw, e2 - n * hw, W0 * 0.5, sketch=0, overshoot=0)


def skull(pen, x, y, s, horns=None):
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    horns = rng.random() < 0.5 if horns is None else horns
    if horns:
        for side in (-1, 1):
            base = (x + side * s * 0.5, y - s * 0.75)
            tip = (x + side * s * rng.uniform(1.3, 1.9), y - s * rng.uniform(1.4, 2.1))
            mid = (x + side * s * 1.3, y - s * 0.75)
            pen.stroke([base, mid, tip], W0 * 1.6, taper=(0.0, 0.9), smooth=2, wobble=0)
    cran = ellipse_pts(x, y - s * 0.55, s * 0.62, s * 0.55, 18)
    jaw = [(x - s * 0.38, y - s * 0.4), (x - s * 0.3, y + s * 0.15), (x + s * 0.3, y + s * 0.15), (x + s * 0.38, y - s * 0.4)]
    cv.erase(cran)
    cv.erase(jaw)
    pen.closed(cran, W0 * 0.6, wobble=0)
    pen.stroke(jaw, W0 * 0.6, taper=(0.1, 0.1), wobble=0)
    for side in (-1, 1):
        pen.blot(ellipse_pts(x + side * s * 0.25, y - s * 0.45, s * 0.17, s * 0.15, 10), jitter=0.05)
    pen.cv.disc(x, y - s * 0.18, s * 0.07)
    for f in (-0.15, 0.0, 0.15):
        pen.stroke([(x + f * s, y + s * 0.02), (x + f * s, y + s * 0.14)], W0 * 0.35, taper=(0, 0))


def ribcage(pen, x, y, L, flip=False):
    rng, W0 = pen.rng, pen.width
    sgn = -1 if flip else 1
    spine = [(x - sgn * L / 2, y), (x, y - L * 0.06), (x + sgn * L / 2, y + L * 0.03)]
    nr = int(rng.integers(4, 7))
    for i in range(nr):
        f = 0.15 + 0.7 * i / max(1, nr - 1)
        bx = x - sgn * L / 2 + sgn * L * f
        rh = L * (0.42 - 0.25 * abs(f - 0.4)) * rng.uniform(0.85, 1.1)
        if rng.random() < 0.15:
            continue
        pts = [(bx, y - L * 0.04), (bx + sgn * rh * 0.25, y - rh), (bx + sgn * rh * 0.55, y - rh * 0.7),
               (bx + sgn * rh * 0.6, y + L * 0.12)]
        pen.stroke(pts, W0 * 0.7, taper=(0.05, 0.6), smooth=2, wobble=0.1)
    pen.stroke(spine, W0 * 1.0, taper=(0.2, 0.2), smooth=1)
    for i in range(int(L / 3)):
        p = np.array(spine[0]) + (np.array(spine[-1]) - np.array(spine[0])) * i / (L / 3)
        pen.stroke([p + (0, -1.2), p + (0, 1.2)], W0 * 0.45, taper=(0, 0))


def remains(pen, cx, cy, spread, n=None):
    rng = pen.rng
    n = n or int(rng.integers(3, 8))
    items = []
    for _ in range(n):
        x, y = cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread * 0.2, spread * 0.25)
        r = rng.random()
        kind = "skull" if r < 0.15 else "ribs" if r < 0.25 else "bone"
        items.append((y, x, kind))
    for y, x, kind in sorted(items):
        if kind == "bone":
            bone(pen, x, y, rng.uniform(7, 13), rng.uniform(0, math.pi))
        elif kind == "skull":
            skull(pen, x, y, rng.uniform(5, 7))
        else:
            ribcage(pen, x, y, rng.uniform(14, 22), flip=rng.random() < 0.5)


def tracks(pen, x, y, heading, n=5):
    rng, W0 = pen.rng, pen.width
    for i in range(n):
        side = 1 if i % 2 else -1
        heading += rng.normal(0, 0.15)
        d = np.array([math.cos(heading), math.sin(heading) * 0.6])
        nrm = np.array([-d[1], d[0]])
        x, y = x + d[0] * 6, y + d[1] * 6
        p = np.array([x, y]) + nrm * side * 2.2
        pen.blot(ellipse_pts(p[0], p[1], 1.1, 0.8, 8), jitter=0.05)
        for a in (-0.5, 0, 0.5):
            dd = np.array([math.cos(heading + a), math.sin(heading + a) * 0.6])
            pen.stroke([p + dd * 1.6, p + dd * 3.2], W0 * 0.55, taper=(0.0, 0.7), wobble=0)


def claw_marks(pen, x, y, size=16):
    rng, W0 = pen.rng, pen.width
    a = rng.uniform(1.0, 1.4)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    for i in (-1, 0, 1):
        c = np.array([x, y]) + n * i * 3.0
        pts = [c - d * size / 2, c + n * 0.8, c + d * size / 2]
        pen.stroke(pts, W0 * 1.0, taper=(0.4, 0.5), smooth=2, wobble=0)


# --------------------------------------------------------------------------- lairs

def crack(pen, x, y, ang, L, w=None, branch=True, inside=None):
    rng = pen.rng
    pts = [(x, y)]
    segs = int(rng.integers(3, 6))
    for _ in range(segs):
        ang += rng.normal(0, 0.5)
        x, y = x + math.cos(ang) * L / segs, y + math.sin(ang) * L / segs * 0.8
        if inside is not None and not _inside(inside, x, y):
            break
        pts.append((x, y))
    if len(pts) < 2:
        return
    pen.stroke(pts, w or pen.width * 0.9, taper=(0.0, 0.85), wobble=0)
    if branch and len(pts) > 2 and rng.random() < 0.6:
        bx, by = pts[int(rng.integers(1, len(pts) - 1))]
        crack(pen, bx, by, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.0), L * 0.45, w, False, inside)


def maw(pen, mx, by, mw, mh, rock_poly=None, teeth=True, cracks=True):
    """A ragged, lopsided black opening in a rock face, with teeth-like rock and cracks."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    skew = rng.normal(0, 0.25)
    n = int(rng.integers(18, 26))
    pts = []
    for i, t in enumerate(np.linspace(math.pi, 0, n)):
        j = rng.uniform(0.72, 0.95) if rng.random() < 0.35 else rng.uniform(0.95, 1.2)
        if i in (0, n - 1):
            j = 1.0
        y = by - (math.sin(t) ** 0.6) * mh * j
        pts.append((mx + math.cos(t) * mw / 2 * j + (by - y) * skew, y))
    bottom = [(mx + mw / 2 * f + rng.normal(0, 1.2), by + rng.uniform(0.5, 3)) for f in np.linspace(0.9, -0.9, 6)]
    centre = np.array([mx + skew * mh * 0.3, by - mh * 0.3])
    # darkness bleeding out of the opening
    for i in range(1, n - 1):
        p = np.array(pts[i])
        base = math.atan2(*(p - centre)[::-1])
        for _ in range(rng.integers(1, 3)):
            a = base + rng.normal(0, 0.3)
            d = np.array([math.cos(a), math.sin(a)])
            pen.stroke([p - d * 1.0, p + d * rng.uniform(3, 9)], W0 * 0.75, taper=(0.0, 0.9), wobble=0)
    pen.blot(pts + bottom, jitter=0.5)
    for _ in range(rng.integers(2, 6) if cracks else 0):
        p = np.array(pts[int(rng.integers(2, n - 2))])
        a = math.atan2(*(p - centre)[::-1]) + rng.normal(0, 0.4)
        crack(pen, p[0], p[1], a, rng.uniform(12, 30), W0 * 0.9, inside=rock_poly)
    if teeth and rng.random() < 0.75:  # jagged rock teeth hanging into the dark
        for i in range(2, n - 2):
            if rng.random() < 0.5:
                p = pts[i]
                tw, tl = rng.uniform(2, 5), rng.uniform(3, 6) * (2.4 if rng.random() < 0.25 else 1)
                cv.erase([(p[0] - tw / 2, p[1] - 0.8), (p[0] + tw / 2, p[1] - 0.8),
                          (p[0] + rng.normal(0, 1.2), p[1] + tl)])
        for _ in range(rng.integers(1, 5)):
            fx, tw = mx + rng.uniform(-mw * 0.35, mw * 0.35), rng.uniform(2.5, 5)
            cv.erase([(fx - tw / 2, by + 2), (fx + tw / 2, by + 2), (fx + rng.normal(0, 1), by - rng.uniform(4, 11))])
    return pts


def skull_pike(pen, x, y, h=None):
    rng = pen.rng
    h = h or rng.uniform(16, 24)
    tx, ty = x + rng.normal(0, 2), y - h
    pen.beam((x, y), (tx, ty), 1.5)
    if rng.random() < 0.4:  # crossbar with a hanging bone
        cy = ty + h * 0.3
        pen.beam((tx - 6, cy), (tx + 6, cy + rng.normal(0, 1)), 1.2)
        pen.stroke([(tx + 4, cy), (tx + 4.5, cy + 5)], pen.width * 0.4, taper=(0, 0), wobble=0)
        bone(pen, tx + 4.5, cy + 8, 6, math.pi / 2)
    skull(pen, tx, ty + 3, rng.uniform(4.5, 6))


def _foreground(pen, cx, by, spread, mouth_x, mouth_w):
    rng = pen.rng
    fg = []
    for _ in range(rng.integers(1, 4)):
        fg.append((by + rng.uniform(0, 6), "boulder", cx + rng.choice([-1, 1]) * rng.uniform(0.3, 0.55) * spread))
    for _ in range(rng.integers(2, 6)):
        fg.append((by + rng.uniform(-2, 8), "tuft", cx + rng.uniform(-0.55, 0.55) * spread))
    if rng.random() < 0.4:
        for _ in range(rng.integers(1, 3)):
            fg.append((by + rng.uniform(-1, 4), "pike", mouth_x + rng.choice([-1, 1]) * rng.uniform(0.6, 1.0) * mouth_w))
    for y, kind, x in sorted(fg):
        if kind == "boulder":
            boulder(pen, x, y, rng.uniform(5, 10))
        elif kind == "pike":
            skull_pike(pen, x, y)
        else:
            pen.tuft(x, y, rng.uniform(3.5, 6))
    if rng.random() < 0.9:
        remains(pen, mouth_x, by + 9, mouth_w * 0.8, n=int(rng.integers(4, 10)))
    if rng.random() < 0.45:
        tracks(pen, mouth_x + rng.uniform(-4, 4), by + 3, math.pi / 2 + rng.uniform(-0.9, 0.9), int(rng.integers(4, 8)))


def hill_cave(pen, P, rng):
    cx, by = P.ox, P.oy
    Bw, Hh = rng.uniform(125, 180), rng.uniform(65, 105)
    hill = hill_pts(rng, cx, by, Bw, Hh)
    if rng.random() < 0.5:
        off = rng.choice([-1, 1]) * Bw * rng.uniform(0.3, 0.45)
        rock(pen, hill_pts(rng, cx + off, by - 4, Bw * rng.uniform(0.45, 0.65), Hh * rng.uniform(0.5, 0.8)))
    rock(pen, hill)
    mx = cx + rng.uniform(-0.18, 0.12) * Bw
    mw = rng.uniform(34, 54)
    mh = min(rng.uniform(26, 42), _height_at(hill, mx, by) * 0.75)
    maw(pen, mx, by, mw, mh, rock_poly=list(hill))
    if rng.random() < 0.6:
        clx = mx + (-1 if rng.random() < 0.7 else 1) * (mw / 2 + rng.uniform(10, 22))
        if _height_at(hill, clx, by) > 25:
            claw_marks(pen, clx, by - _height_at(hill, clx, by) * 0.45, rng.uniform(12, 18))
    _foreground(pen, cx, by, Bw, mx, mw)


def cliff_cave(pen, P, rng):
    """Sheer rock wall with the opening at its foot."""
    W0 = pen.width
    cx, by = P.ox, P.oy
    Bw, Hh = rng.uniform(120, 170), rng.uniform(85, 120)
    m = int(rng.integers(6, 10))
    top = [(cx - Bw / 2 + rng.uniform(8, 20) + i * (Bw - 30) / (m - 1) + rng.normal(0, 3),
            by - Hh * rng.uniform(0.85, 1.0) + rng.normal(0, 3)) for i in range(m)]
    def side(x_out, x_in, h_top, sgn):
        out = []
        for f in np.linspace(0.15, 0.85, int(rng.integers(3, 6))):
            out.append((x_out + (x_in - x_out) * f ** 1.5 + sgn * rng.uniform(-6, 2), by - h_top * f + rng.normal(0, 2)))
        return out
    lside = side(cx - Bw / 2, top[0][0], by - top[0][1], -1)
    rside = side(cx + Bw / 2, top[-1][0], by - top[-1][1], 1)[::-1]
    pts = [(cx - Bw / 2, by)] + lside + top + rside + [(cx + Bw / 2, by)]
    rock(pen, pts)
    for _ in range(rng.integers(3, 7)):  # vertical joints in the cliff
        x = cx + rng.uniform(-0.4, 0.4) * Bw
        y0 = by - Hh * rng.uniform(0.6, 0.85)
        pen.stroke([(x, y0), (x + rng.normal(0, 2), y0 + rng.uniform(15, 35))], W0 * 0.6, taper=(0.2, 0.7))
    mx = cx + rng.uniform(-0.2, 0.2) * Bw
    mw, mh = rng.uniform(34, 52), rng.uniform(30, 46)
    maw(pen, mx, by, mw, mh, rock_poly=pts)
    if rng.random() < 0.5:
        claw_marks(pen, mx + rng.choice([-1, 1]) * (mw / 2 + rng.uniform(10, 18)), by - Hh * 0.4, rng.uniform(12, 18))
    _foreground(pen, cx, by, Bw, mx, mw)


def boulder_cave(pen, P, rng):
    """Two great rocks leaning together under a capstone; the dark opening between them."""
    cx, by = P.ox, P.oy
    gap = rng.uniform(30, 44)
    hl, hr = rng.uniform(50, 75), rng.uniform(45, 70)
    wl, wr = rng.uniform(45, 70), rng.uniform(40, 65)
    left = hill_pts(rng, cx - gap / 2 - wl * 0.35, by, wl, hl, m=6)
    right = hill_pts(rng, cx + gap / 2 + wr * 0.35, by, wr, hr, m=6)
    cap_y = by - min(hl, hr) * rng.uniform(0.6, 0.8)
    cw = gap + rng.uniform(30, 50)
    cap = blob_pts(rng, cx + rng.normal(0, 4), cap_y, cw / 2, n=10, irregular=0.12, squash=rng.uniform(0.35, 0.5))
    first, second = (left, right) if rng.random() < 0.5 else (right, left)
    rock(pen, first)
    rock(pen, second)
    pen.cv.erase(cap)
    pen.hatch(clip_halfplane(list(cap), (1, 0), cx + cw * 0.1), 60, 2.2, pen.width * 0.6)
    pen.closed(cap, pen.width * 1.2)
    maw(pen, cx, by, gap * 1.1, by - cap_y - 2, rock_poly=list(cap))
    _foreground(pen, cx, by, gap + wl + wr, cx, gap)


def low_entrance(pen, P, rng):
    """A low rock lip over a wide mouth, with a worn path leading down into it."""
    W0 = pen.width
    cx, by = P.ox, P.oy - 10
    Bw, Hh = rng.uniform(110, 150), rng.uniform(32, 48)
    hill = hill_pts(rng, cx, by, Bw, Hh, m=int(rng.integers(8, 12)))
    rock(pen, hill)
    mx = cx + rng.uniform(-0.1, 0.1) * Bw
    mw = rng.uniform(44, 64)
    maw(pen, mx, by, mw, _height_at(hill, mx, by) * rng.uniform(0.7, 0.85), rock_poly=list(hill))
    # path walls converging on the mouth
    for sgn in (-1, 1):
        a = (mx + sgn * mw * 0.42, by + 2)
        b = (mx + sgn * mw * rng.uniform(0.7, 0.9), by + rng.uniform(18, 26))
        pen.stroke([a, ((a[0] + b[0]) / 2 + sgn * 2, (a[1] + b[1]) / 2), b], W0 * 0.8, taper=(0.1, 0.8), smooth=2)
        for f in rng.uniform(0.2, 0.9, int(rng.integers(1, 4))):
            pen.stone(a[0] + (b[0] - a[0]) * f + sgn * 3, a[1] + (b[1] - a[1]) * f, rng.uniform(1.8, 3.2))
    _foreground(pen, cx, by + 14, Bw, mx, mw)


CAVE_KINDS = [(hill_cave, 0.35), (cliff_cave, 0.25), (boulder_cave, 0.25), (low_entrance, 0.15)]


def cave_lair(pen, P, rng):
    fns, wts = zip(*CAVE_KINDS)
    fns[int(rng.choice(len(fns), p=np.array(wts) / sum(wts)))](pen, P, rng)


def scratches(pen, x, y, size=8):
    rng, W0 = pen.rng, pen.width
    a = rng.uniform(-0.5, 0.5) + (math.pi if rng.random() < 0.5 else 0)
    d = np.array([math.cos(a), math.sin(a) * 0.5])
    n = np.array([-d[1], d[0]]) / max(1e-6, np.hypot(*d))
    for i in (-1, 0, 1):
        c = np.array([x, y]) + n * i * 2.0
        pen.stroke([c - d * size / 2, c + d * size / 2], W0 * 0.7, taper=(0.4, 0.6), wobble=0)


def _burrow(pen, P, X, Y, R, big=True):
    """Earth mound on an oval footprint with a dug opening facing the viewer."""
    rng = pen.rng
    Ry, Hh = R * rng.uniform(0.75, 0.95), R * rng.uniform(0.38, 0.5)
    poly, rim, top_at = mound(pen, P, X, Y, R, Ry, Hh)
    tf = P.az - math.pi / 2 + rng.uniform(-0.35, 0.35)
    mx, by = P(X + R * 0.95 * math.cos(tf), Y + Ry * 0.95 * math.sin(tf), 0)
    mw = R * rng.uniform(0.6, 0.8)
    mh = (by - top_at(mx)) * rng.uniform(0.5, 0.7)
    edge = maw(pen, mx, by, mw, mh, rock_poly=poly, teeth=False, cracks=False)
    for _ in range(rng.integers(3, 8)):  # roots dangling over the opening
        p = edge[int(rng.integers(3, len(edge) - 3))]
        L = rng.uniform(3, mh * 0.6)
        pen.cv.erase([(p[0] - 0.7, p[1] - 0.5), (p[0] + 0.7, p[1] - 0.5), (p[0] + rng.normal(0, 1), p[1] + L)])
    for _ in range(rng.integers(1, 4) if big else 0):
        sx = mx + rng.choice([-1, 1]) * rng.uniform(mw * 0.6, mw * 1.0)
        if by - top_at(sx) > 12:
            scratches(pen, sx, by - (by - top_at(sx)) * 0.45, rng.uniform(7, 11))
    for _ in range(rng.integers(2, 5)):  # loose dirt spilling out
        pen.stone(mx + rng.uniform(-mw * 0.6, mw * 0.6), by + rng.uniform(1, 6), rng.uniform(1.0, 2.2))
    for _ in range(rng.integers(5, 11)):  # grass all over the mound
        t = rng.uniform(0, math.tau)
        rr = rng.uniform(0.0, 0.85)
        if math.sin(t - P.az) > 0.4 and rr > 0.5:
            continue  # far side: hidden
        x, y = P(X + R * rr * math.cos(t), Y + Ry * rr * math.sin(t), Hh * (1 - rr * rr) ** 0.7)
        if abs(x - mx) < mw * 0.6 and y > by - mh * 1.1:
            continue  # keep the opening clear
        pen.tuft(x, y + 1, rng.uniform(3.5, 6))
    return mx, by, mw


def _spoil_heap(pen, P, X, Y, R):
    rng = pen.rng
    poly, rim, _ = mound(pen, P, X, Y, R, R * 0.75, R * rng.uniform(0.35, 0.5), texture=False)
    xs = [p[0] for p in poly]
    for _ in range(int(R * 1.2)):
        t = rng.uniform(0, math.tau)
        r = rng.uniform(0, 0.8)
        pen.cv.disc(*P(X + R * r * math.cos(t), Y + R * 0.75 * r * math.sin(t), R * 0.3 * (1 - r * r)), 0.45)
    for _ in range(rng.integers(2, 6)):
        p = rim[int(rng.integers(0, len(rim)))]
        pen.stone(p[0] + rng.normal(0, 2), p[1] + rng.uniform(0, 3), rng.uniform(1.0, 2.0))


def _ground_scatter(pen, P, items, area, avoid, n_tufts=(3, 7), n_stones=(1, 4)):
    """Add tufts/stones on the ground at world points not inside `avoid` circles [(x, y, r)]."""
    rng = pen.rng
    (x0, x1), (y0, y1) = area
    for kind, (lo, hi) in (("tuft", n_tufts), ("stone", n_stones)):
        for _ in range(rng.integers(lo, hi)):
            for _ in range(20):
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if all(math.hypot(x - ax, (y - ay) * 1.2) > ar for ax, ay, ar in avoid):
                    items.append((P.depth(x, y), kind, (x, y)))
                    break


def _draw_items(pen, P, items, handlers):
    for _, kind, args in sorted(items, key=lambda it: -it[0]):
        handlers[kind](*args)


def burrow_lair(pen, P, rng, warren=None):
    if warren is None:
        warren = rng.random() >= 0.6
    if not warren:
        spots = [(0.0, 0.0, rng.uniform(45, 58), True)]
    else:  # warren
        spots = []
        for _ in range(100):
            if len(spots) >= int(rng.integers(2, 4)):
                break
            x, y, R = rng.uniform(-55, 55), rng.uniform(-35, 35), rng.uniform(28, 38)
            if all(math.hypot(x - sx, y - sy) > (R + sr) * 0.95 for sx, sy, sr, _ in spots):
                spots.append((x, y, R, False))
    items = [(P.depth(x, y), "burrow", (x, y, R, big)) for x, y, R, big in spots]
    for x, y, R, _ in spots:
        for _ in range(rng.integers(0, 3)):
            t = rng.uniform(0, math.tau)
            if math.sin(t - P.az) < -0.3:  # not in front of the opening
                continue
            hx, hy = x + math.cos(t) * R * 1.05, y + math.sin(t) * R * 0.95
            items.append((P.depth(hx, hy), "heap", (hx, hy, R * rng.uniform(0.35, 0.5))))
    avoid = [(x, y, R * 1.1) for x, y, R, _ in spots]
    _ground_scatter(pen, P, items, ((-90, 90), (-60, 50)), avoid)
    for x, y, R, _ in spots:
        for _ in range(rng.integers(0, 3)):
            items.append((P.depth(x, y - R * 1.3), "scratch", (x + rng.uniform(-R, R), y - R * rng.uniform(1.1, 1.5))))
    front = {}

    def do_burrow(x, y, R, big):
        mx, by, mw = _burrow(pen, P, x, y, R, big)
        if not front or by > front["by"]:
            front.update(mx=mx, by=by, mw=mw)

    _draw_items(pen, P, items, {
        "burrow": do_burrow,
        "heap": lambda x, y, R: _spoil_heap(pen, P, x, y, R),
        "tuft": lambda x, y: pen.tuft(*P(x, y), rng.uniform(3.5, 6)),
        "stone": lambda x, y: pen.stone(*P(x, y), rng.uniform(1.5, 3)),
        "scratch": lambda x, y: scratches(pen, *P(x, y), rng.uniform(6, 9)),
    })
    if rng.random() < 0.75:
        remains(pen, front["mx"], front["by"] + 9, front["mw"] * 0.9, n=int(rng.integers(2, 6)))
    if rng.random() < 0.5:
        tracks(pen, front["mx"], front["by"] + 3, math.pi / 2 + rng.uniform(-1.0, 1.0), int(rng.integers(3, 7)))


def _twig(pen, x, y, ang, L):
    rng = pen.rng
    d = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-d[1], d[0]])
    c = np.array([x, y])
    pen.stroke([c - d * L / 2, c + n * rng.normal(0, L * 0.08), c + d * L / 2], pen.width * 0.6,
               taper=(0.25, 0.25), smooth=1, wobble=0.1)


def nest(pen, cx, cy, rx, ry, contents=None):
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    contents = contents or rng.choice(["eggs", "eggs", "bones"])
    bh = min(ry * 1.6, rx * 0.45)
    body = list(ellipse_pts(cx, cy, rx * 0.98, ry * 1.05, 40, math.pi, TAU)) + \
        list(ellipse_pts(cx, cy + ry * 0.2, rx * 0.96, bh, 30, 0, math.pi))
    cv.erase(body)

    def rim_twig(theta, f_lo, f_hi):
        f = rng.uniform(f_lo, f_hi)
        x, y = cx + rx * f * math.cos(theta), cy + ry * f * math.sin(theta)
        tang = math.atan2(ry * math.cos(theta), -rx * math.sin(theta))
        _twig(pen, x, y, tang + rng.normal(0, 0.45), rng.uniform(8, 16))

    for _ in range(int(rx * 0.9)):
        rim_twig(rng.uniform(math.pi, TAU), 0.88, 1.12)
    inner = ellipse_pts(cx, cy - ry * 0.05, rx * 0.8, ry * 0.62, 30)
    cv.erase(inner)
    pen.hatch(inner, rng.uniform(5, 20), 1.3, W0 * 0.7, coverage=(0.8, 1.0))
    items = []
    if contents == "eggs":
        ex = cx + rng.uniform(-rx * 0.25, rx * 0.25)
        for _ in range(rng.integers(3, 6)):
            items.append(("egg", ex + rng.uniform(-rx * 0.3, rx * 0.3), cy + rng.uniform(-0.15, 0.2) * ry))
    else:
        for _ in range(rng.integers(2, 5)):
            items.append(("bone", cx + rng.uniform(-rx * 0.5, rx * 0.5), cy + rng.uniform(-ry * 0.2, ry * 0.3)))
        if rng.random() < 0.6:
            items.append(("skull", cx + rng.uniform(-rx * 0.4, rx * 0.4), cy + ry * 0.25))
    for kind, x, y in sorted(items, key=lambda it: it[2]):
        if kind == "egg":
            ew, eh = rng.uniform(3.6, 4.8), rng.uniform(5.0, 6.5)
            e = ellipse_pts(x, y - eh, ew, eh, 18)
            tilt = rng.normal(0, 0.25)
            e = np.stack([x + (e[:, 0] - x) + (e[:, 1] - y) * tilt * -1, e[:, 1]], 1)
            cv.erase(e)
            pen.closed(e, W0 * 0.6, wobble=0)
            pen.stroke([(x + ew * 0.35, y - eh * 1.6), (x + ew * 0.6, y - eh * 1.1)], W0 * 0.5, taper=(0.4, 0.4))
            for _ in range(rng.integers(2, 6)):
                cv.disc(x + rng.uniform(-2, 2), y - eh + rng.uniform(-3, 3), 0.4)
        elif kind == "bone":
            bone(pen, x, y, rng.uniform(7, 11), rng.uniform(0, math.pi))
        else:
            skull(pen, x, y, 5, horns=False)
    cv.erase(list(ellipse_pts(cx, cy, rx * 0.98, ry * 1.05, 24, 0, math.pi))
             + list(ellipse_pts(cx, cy - ry * 0.05, rx * 0.8, ry * 0.62, 24, math.pi, 0)))
    cv.erase(list(ellipse_pts(cx, cy + ry * 0.2, rx * 0.96, bh, 24, 0, math.pi)) + [(cx - rx * 0.96, cy), (cx + rx * 0.96, cy)])
    for _ in range(int(rx * 1.3)):
        rim_twig(rng.uniform(0, math.pi), 0.85, 1.15)
    for _ in range(int(rx * 0.8)):  # underside
        th = rng.uniform(0.15, math.pi - 0.15)
        f = rng.uniform(0.55, 1.0)
        x, y = cx + rx * 1.08 * f * math.cos(th), cy + ry * 0.2 + bh * 1.1 * f * math.sin(th)
        tang = math.atan2(bh * 1.1 * math.cos(th), -rx * 1.08 * math.sin(th))
        _twig(pen, x, y, tang + rng.normal(0, 0.5), rng.uniform(6, 12))


def nest_lair(pen, P, rng, crag=None):
    cx, by = P.ox, P.oy
    rx = rng.uniform(32, 46)
    ry = rx * rng.uniform(0.48, 0.56)
    on_crag = rng.random() < 0.6 if crag is None else crag
    foot = rng.uniform(70, 100) / 2 if on_crag else rx * 1.05
    # things on the ground around the base, split into behind / in front
    ring = []
    for _ in range(rng.integers(4, 9)):
        t = rng.uniform(0, math.tau)
        d = foot * rng.uniform(1.05, 1.6)
        x, y = cx + d * math.cos(t), by - (0 if on_crag else ry * 0.9) + d * 0.5 * math.sin(t)
        ring.append((y, x, str(rng.choice(["tuft", "tuft", "boulder", "stone", "bone"]))))

    def draw_ring(sel):
        for y, x, kind in sorted(r for r in ring if sel(r)):
            if kind == "tuft":
                pen.tuft(x, y, rng.uniform(3.5, 6))
            elif kind == "boulder":
                boulder(pen, x, y, rng.uniform(4, 8))
            elif kind == "stone":
                pen.stone(x, y, rng.uniform(1.5, 3))
            else:
                bone(pen, x, y, rng.uniform(7, 12), rng.uniform(0, math.pi))

    mid = by - (0 if on_crag else ry * 0.9)
    draw_ring(lambda r: r[0] < mid)
    if on_crag:
        ch, bw, tw = rng.uniform(60, 95), foot * 2, rx * rng.uniform(0.6, 0.9)
        flare = rng.uniform(0.35, 0.6)
        lean = rng.normal(0, 8)

        def edge(sgn):
            out = []
            for f in np.linspace(0, 1, 9):
                w = bw / 2 - (bw - tw) / 2 * f ** flare
                out.append((cx + lean * f + sgn * (w + rng.normal(0, 4.5) * (0 < f < 1)), by - ch * f))
            return out
        left, right = edge(-1), edge(1)[::-1]
        top = [(cx + lean + rng.uniform(-tw / 4, tw / 4), by - ch - rng.uniform(0, 5))]
        rock(pen, left + top + right, base_sag=bw / 2 * 0.5)
        nest(pen, cx + lean, by - ch - ry * 0.1, rx, ry)
    else:
        nest(pen, cx, by - ry * 0.9, rx, ry)
    draw_ring(lambda r: r[0] >= mid)
    if rng.random() < 0.7:
        remains(pen, cx, by + (foot * 0.45 if on_crag else 8), 40, n=int(rng.integers(2, 6)))
