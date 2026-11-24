"""Landscape symbols: farmland patchworks and canyons."""
import math

import numpy as np

from .ink import TAU, chaikin
from .solids import cone


def _bilerp(q, u, v):
    a, b, c, d = (np.asarray(p, float) for p in q)  # q = [(0,0), (1,0), (1,1), (0,1)] corners
    return tuple(a * (1 - u) * (1 - v) + b * u * (1 - v) + c * u * v + d * (1 - u) * v)


# --------------------------------------------------------------------------- fields

def _field_cell(pen, P, quad, kind, extras):
    rng, W0 = pen.rng, pen.width

    def S(u, v, h=0.0):
        x, y = _bilerp(quad, u, v)
        return P(x, y, h)

    along_u = rng.random() < 0.5

    def rows(spacing_world):
        size = math.dist(quad[0], quad[3] if along_u else quad[1])
        n = max(2, int(size / spacing_world))
        return np.linspace(0.06, 0.94, n)

    def row_line(v):
        return (S(0.04, v), S(0.96, v)) if along_u else (S(v, 0.04), S(v, 0.96))

    if kind == "furrow":
        for v in rows(3.2):
            a, b = row_line(v)
            pen.stroke([a, b], W0 * 0.45, taper=(0.2, 0.2), wobble=0.4)
        if rng.random() < 0.35:
            extras.append("stooks")
    elif kind == "grain":
        for v in rows(4.0):
            a, b = row_line(v)
            n = int(math.dist(a, b) / 2.4)
            for i in range(n):
                f = (i + rng.uniform(0.2, 0.8)) / n
                x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                hgt = rng.uniform(2.0, 3.2)
                pen.stroke([(x, y), (x + rng.normal(0, 0.3), y - hgt)], W0 * 0.45, taper=(0.0, 0.6), wobble=0)
    elif kind == "orchard":
        pts = []
        nu, nv = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        for i in range(nu):
            for j in range(nv):
                u, v = (i + 0.5) / nu, (j + 0.5) / nv
                x, y = _bilerp(quad, u, v)
                pts.append((P.depth(x, y), x, y))
        for _, x, y in sorted(pts, reverse=True):
            sx, sy = P(x, y)
            pen.stroke([(sx, sy), (sx, sy - 3)], W0 * 0.7, taper=(0, 0), wobble=0)
            pen.bush(sx, sy - 2, rng.uniform(3.5, 4.5))
    elif kind == "vineyard":
        for v in rows(5.0):
            a, b = row_line(v)
            n = int(math.dist(a, b) / 5)
            tops = []
            for i in range(n + 1):
                x, y = a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n
                pen.stroke([(x, y), (x, y - 4)], W0 * 0.5, taper=(0, 0), wobble=0)
                tops.append((x, y - 3.5))
            pen.stroke(tops, W0 * 0.4, taper=(0.1, 0.1), wobble=0.2)
            for _ in range(n * 2):
                f = rng.uniform(0, 1)
                pen.dab(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f - 3, 1.3, w=W0 * 0.6)
    elif kind == "fallow":
        for _ in range(int(math.dist(quad[0], quad[2]) * 0.8)):
            pen.cv.disc(*S(rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)), 0.45)
        for _ in range(rng.integers(1, 4)):
            pen.tuft(*S(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)), rng.uniform(3, 4.5))
    else:  # pasture
        for _ in range(int(math.dist(quad[0], quad[2]) * 0.12)):
            pen.tuft(*S(rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)), rng.uniform(3, 4.5))
        if rng.random() < 0.5:
            extras.append("haystack")


def _boundary(pen, P, a, b, kind):
    rng, W0 = pen.rng, pen.width
    A, B = P(*a), P(*b)
    L = math.dist(A, B)
    if kind == "line":
        pen.stroke([A, B], W0 * 0.6, taper=(0.1, 0.1), wobble=0.5)
    elif kind == "hedge":
        n = max(2, int(L / 2.6))
        for i in range(n + 1):
            f = i / n
            x, y = A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f
            for k in range(2):
                pen.dab(x + rng.normal(0, 0.6), y - 1.2 - k * 1.8 + rng.normal(0, 0.4), rng.uniform(1.4, 2.0),
                        w=W0 * (0.9 if k == 0 else 0.7))
    elif kind == "wall":
        pen.stroke([A, B], W0 * 0.7, taper=(0.05, 0.05), wobble=0.4)
        pen.stroke([(A[0], A[1] - 2.2), (B[0], B[1] - 2.2)], W0 * 0.6, taper=(0.05, 0.05), wobble=0.4)
        for i in range(int(L / 3)):
            f = (i + rng.uniform(0.2, 0.8)) / int(L / 3)
            x, y = A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f
            pen.stroke([(x, y), (x, y - 2.2)], W0 * 0.4, taper=(0, 0), wobble=0)
    elif kind == "fence":
        n = max(1, int(L / 6))
        tops = []
        for i in range(n + 1):
            x, y = A[0] + (B[0] - A[0]) * i / n, A[1] + (B[1] - A[1]) * i / n
            pen.stroke([(x, y), (x, y - 4.5)], W0 * 0.6, taper=(0, 0.1), wobble=0)
            tops.append((x, y - 3.8))
        pen.stroke(tops, W0 * 0.45, taper=(0.05, 0.05), wobble=0.2)
        pen.stroke([(p[0], p[1] + 2) for p in tops], W0 * 0.45, taper=(0.05, 0.05), wobble=0.2)
    elif kind == "track":
        for off in (-1.6, 1.6):
            d = np.array([B[1] - A[1], A[0] - B[0]]) / max(L, 1e-6) * off
            pen.stroke([np.add(A, d), np.add(B, d)], W0 * 0.45, taper=(0.1, 0.1), wobble=0.6)


def fields(pen, P, rng):
    nx, ny = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    W, D = rng.uniform(140, 200), rng.uniform(90, 130)
    cx = np.concatenate([[0], np.cumsum(rng.uniform(0.7, 1.3, nx))])
    cy = np.concatenate([[0], np.cumsum(rng.uniform(0.7, 1.3, ny))])
    cx, cy = cx / cx[-1] * W - W / 2, cy / cy[-1] * D - D / 2
    V = [[(cx[i] + rng.normal(0, 4) * (0 < i < nx), cy[j] + rng.normal(0, 4) * (0 < j < ny)) for j in range(ny + 1)]
         for i in range(nx + 1)]
    # ragged outer border: skip some corner cells
    skip = {(0, ny - 1)} if rng.random() < 0.4 else set()
    if rng.random() < 0.4:
        skip.add((nx - 1, 0))
    extras = []
    cells = []
    for i in range(nx):
        for j in range(ny):
            if (i, j) in skip:
                continue
            quad = [V[i][j], V[i + 1][j], V[i + 1][j + 1], V[i][j + 1]]
            kind = str(rng.choice(["furrow", "furrow", "grain", "grain", "orchard", "vineyard", "fallow", "pasture"],
                                  p=[0.2, 0.1, 0.18, 0.1, 0.12, 0.1, 0.1, 0.1]))
            cells.append((quad, kind))
    for quad, kind in sorted(cells, key=lambda c: -P.depth(*np.mean(c[0], axis=0))):
        ex = []
        _field_cell(pen, P, quad, kind, ex)
        for e in ex:
            for _ in range(rng.integers(2, 6) if e == "stooks" else rng.integers(1, 3)):
                u, v = rng.uniform(0.15, 0.85, 2)
                extras.append((quad, e, u, v))
    edges = set()
    for i in range(nx):
        for j in range(ny):
            if (i, j) in skip:
                continue
            for e in (((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)), ((i, j + 1), (i + 1, j + 1)), ((i, j), (i, j + 1))):
                edges.add(e)
    kinds = ["hedge", "wall", "fence", "line", "track"]
    style = rng.dirichlet(np.ones(len(kinds)) * 0.7)
    items = []
    for (i0, j0), (i1, j1) in edges:
        a, b = V[i0][j0], V[i1][j1]
        items.append((P.depth((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), "edge", (a, b, str(rng.choice(kinds, p=style)))))
    for quad, e, u, v in extras:
        x, y = _bilerp(quad, u, v)
        items.append((P.depth(x, y), e, (x, y)))
    for _ in range(rng.integers(0, 3)):
        x, y = rng.uniform(-W / 2, W / 2), -D / 2 - rng.uniform(4, 12)
        items.append((P.depth(x, y), "bush", (x, y)))
    for _, kind, args in sorted(items, key=lambda it: -it[0]):
        if kind == "edge":
            _boundary(pen, P, *args)
        elif kind == "stooks":
            x, y = args
            cone(pen, P, x, y, rng.uniform(2.2, 3.0), 0, rng.uniform(5, 7), shade=True)
        elif kind == "haystack":
            x, y = args
            r = rng.uniform(4, 6)
            cone(pen, P, x, y, r, 0, r * rng.uniform(1.6, 2.2))
        elif kind == "bush":
            pen.bush(*P(*args), rng.uniform(5, 8))


# --------------------------------------------------------------------------- cliffs & canyons
# Open edge lines in the style of hand-drawn map escarpments: a heavy ink edge, a comb of
# tapered strokes hanging below it where the face is visible, and a fading wash. Pieces have
# open, tapered ends and a fixed face scale, so several can be overlapped into longer runs.

FACE_H = (9.0, 16.0)  # shared face-height range keeps pieces stackable


def _smooth_noise(rng, n, scale=8, lo=0.0, hi=1.0):
    k = max(1, int(scale))
    raw = rng.uniform(lo, hi, n + 2 * k)
    return np.convolve(raw, np.ones(k) / k, "same")[k:k + n]


def edge_path(rng, length, heading, wiggle=0.22, kinks=0.0, step=2.0):
    """Smooth open edge, centred on the origin."""
    n = max(4, int(length / step))
    s = np.arange(n) * step
    ph = rng.uniform(0, TAU, 3)
    ang = heading + wiggle * (0.6 * np.sin(s / rng.uniform(18, 30) + ph[0]) + 0.4 * np.sin(s / rng.uniform(7, 12) + ph[1]))
    for _ in range(rng.poisson(kinks * length / 60)):  # soft spurs (smoothed, never a hard corner)
        i = int(rng.integers(3, n - 4))
        ang[i - 2:i + 3] += rng.choice([-1, 1]) * rng.uniform(0.4, 0.8) * np.array([0.3, 0.7, 1, 0.7, 0.3])
    d = np.stack([np.cos(ang), np.sin(ang)], 1) * step
    pts = chaikin(np.vstack([[0, 0], np.cumsum(d, 0)]), 2)[::4]
    return pts - pts.mean(0)


def jagged(rng, pts, low, seg=(5.0, 10.0), amp=1.4, notch=0.15, step=1.5):
    """Re-draw a smooth edge as rocky straight segments between irregular vertices.
    Returns (jagged points, low-side normals carried over from the smooth curve)."""
    pts = np.asarray(pts, float)
    low = np.asarray(low, float)
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return pts, low
    marks = [0.0]
    while marks[-1] < s[-1]:
        marks.append(marks[-1] + rng.uniform(*seg))
    marks[-1] = s[-1]
    verts, vs = [], []
    for k, m in enumerate(marks):
        i = int(np.searchsorted(s, m).clip(0, len(pts) - 1))
        off = 0.0 if k in (0, len(marks) - 1) else rng.normal(0, amp)
        verts.append(pts[i] + low[i] * off)
        vs.append(m)
        if 0 < k < len(marks) - 1 and rng.random() < notch:  # small step/notch in the rock
            t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
            t /= max(np.hypot(*t), 1e-6)
            jog = rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
            verts.append(pts[i] + low[i] * (off + jog) + t * rng.uniform(0.5, 1.5))
            vs.append(m + 0.01)
    verts = np.array(verts)
    out, src = [verts[0]], [0.0]
    for a, b, va, vb in zip(verts[:-1], verts[1:], vs[:-1], vs[1:]):
        L = np.hypot(*(b - a))
        for f in np.linspace(0, 1, max(2, int(L / step)) + 1)[1:]:
            out.append(a + (b - a) * f)
            src.append(va + (vb - va) * f)
    idx = np.searchsorted(s, np.array(src)).clip(0, len(pts) - 1)
    return np.array(out), low[idx]


def _face(pen, pts, h, density=1.0):
    """Hanging comb strokes, rock teeth and wash under edge `pts`; `h` = face height per point."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    n = len(pts)
    h = np.asarray(h, float) * np.maximum(np.sin(np.linspace(0, np.pi, n)) ** 0.35, 0.15)
    for k, v in enumerate((235, 190, 140, 90, 45)):
        f = (k + 1) / 5 * 1.25
        cv.wash(list(pts) + [(x, y + hh * f) for (x, y), hh in zip(pts[::-1], h[::-1])], v)
    i = 0
    while i < n:
        x, y = pts[i]
        if h[i] > 2.0:
            L = h[i] * rng.uniform(0.45, 0.95)
            lean = rng.normal(0, 0.7)
            heavy = rng.random() < 0.4
            pen.stroke([(x, y), (x + lean * 0.3, y + L * 0.5), (x + lean, y + L)],
                       W0 * (1.15 if heavy else 0.8), taper=(0.0, 0.9), wobble=0, smooth=1)
            if rng.random() < 0.12 and h[i] > 6:
                dx = rng.uniform(1.2, 2.2)
                cv.fill([(x - dx, y), (x + dx, y), (x + rng.normal(0, 0.5), y + rng.uniform(2.5, 4))])
        i += max(1, int(rng.uniform(1.6, 2.8) / density))


def _spur(pen, pts, i):
    """Tiny overhang on an otherwise plain edge (little notch with a lick of wash)."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    x, y = pts[i]
    w = rng.uniform(4, 7)
    tri = [(x - w / 2, y), (x + w / 2, y + rng.uniform(-0.5, 0.5)), (x + rng.uniform(-1, 1), y + rng.uniform(3, 5))]
    cv.wash(tri, 200)
    for dx in np.linspace(-w * 0.3, w * 0.3, int(rng.integers(1, 3))):
        pen.stroke([(x + dx, y), (x + dx, y + rng.uniform(2, 4))], W0 * 0.7, taper=(0.05, 0.85), wobble=0)


def _edge_line(pen, pts, w=1.2):
    pen.stroke(pts, pen.width * w, taper=(0.15, 0.15), taper_max=16, wobble=0.1, pressure=0.3)


def right_normals(pts):
    """Unit normals on the right-hand side of travel (screen coords: L->R path gives 'down')."""
    d = np.gradient(np.asarray(pts, float), axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
    return np.stack([-d[:, 1], d[:, 0]], 1)


def escarp(pen, pts, low, hmax, mask=None, spurs=True, line=True, line_w=1.2, density=1.0, jag=1.4):
    """Edge whose low ground lies along `low` (unit vectors). The face shows where the drop faces
    the viewer (low side pointing down-screen) and fades out smoothly as the edge turns away."""
    rng = pen.rng
    pts = np.asarray(pts, float)
    if jag:
        pts, low = jagged(rng, pts, low, amp=jag)
    n = len(pts)
    strength = np.clip(np.asarray(low)[:, 1], 0, 1)
    strength = np.convolve(np.pad(strength, 6, mode="edge"), np.ones(13) / 13, "valid")
    if mask is not None:
        strength = strength * np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(mask)), mask)
    h = hmax * strength ** 0.8 * _smooth_noise(rng, n, 6, 0.55, 1.0)
    on = h > 1.5
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i > 3:
                _face(pen, pts[i:j], h[i:j], density)
            i = j
        else:
            i += 1
    if spurs:
        for _ in range(rng.integers(0, 3)):
            k = int(rng.integers(3, max(4, n - 3)))
            if not on[k] and low[k][1] > -0.3:
                _spur(pen, pts, k)
    if line:
        _edge_line(pen, pts, line_w)


def _section_mask(rng, n, cover=(0.55, 1.0)):
    """Smooth 0..1 mask switching the face on and off along a straight piece."""
    if rng.random() < 0.35:
        return np.ones(n)
    m = np.zeros(n)
    for _ in range(int(rng.integers(1, 3))):
        a = int(rng.uniform(-0.1, 0.6) * n)
        b = int(a + rng.uniform(*cover) * n)
        m[max(0, a):min(n, b)] = 1
    return np.clip(np.convolve(np.pad(m, 6, mode="edge"), np.ones(13) / 13, "valid"), 0, 1)


def cliff(pen, P, rng, heading=None):
    """A single escarpment edge piece, low ground below."""
    if heading is None:
        heading = float(rng.choice([0.0, 0.0, 0.35, -0.35]) + rng.normal(0, 0.1))
    pts = edge_path(rng, rng.uniform(70, 190), heading, kinks=0.3) + (P.ox, P.oy)
    escarp(pen, pts, right_normals(pts), rng.uniform(*FACE_H), _section_mask(rng, len(pts)))


def cliff_steep(pen, P, rng):
    cliff(pen, P, rng, heading=float(rng.choice([1, -1]) * rng.uniform(0.7, 1.1)))


# --- curves: arcs of an elliptical outline (perspective-squashed), convex or concave -------------

ARC_ROLES = {
    # name: (start, end) angle in screen coords (0 = right, pi/2 = bottom/front)
    "front": (0.15, 0.85), "back": (1.15, 1.85),
    "left_end": (0.5, 1.5), "right_end": (-0.5, 0.5),
    "front_left": (0.5, 1.0), "front_right": (0.0, 0.5),
    "back_left": (1.0, 1.5), "back_right": (1.5, 2.0),
}


def ring_pts(rng, rx, ry, t0, t1, wob=0.05, step=2.0):
    """Points on a gently irregular ellipse from angle t0 to t1, with outward unit normals."""
    span = abs(t1 - t0)
    n = max(8, int(span * (rx + ry) / 2 / step))
    t = np.linspace(t0, t1, n)
    ph = rng.uniform(0, TAU, 3)
    r = 1 + wob * (np.sin(3 * t + ph[0]) + 0.6 * np.sin(5 * t + ph[1]) + 0.3 * np.sin(9 * t + ph[2]))
    e = 2.0 / rng.uniform(2.4, 3.2)  # superellipse: flatter runs, the turning happens nearer the corners
    c, sn = np.cos(t), np.sin(t)
    pts = np.stack([rx * r * np.sign(c) * np.abs(c) ** e, ry * r * np.sign(sn) * np.abs(sn) ** e], 1)
    nrm = right_normals(pts) * (-1 if t1 > t0 else 1)  # increasing angle travels clockwise on screen
    return pts, nrm


def cliff_curve(pen, P, rng, role=None, inner=False):
    """Curved piece of a plateau outline. Convex: high ground inside. Inner: a bay (high ground outside)."""
    role = role or str(rng.choice(list(ARC_ROLES)))
    t0, t1 = (v * np.pi for v in ARC_ROLES[role])
    if inner:  # same arc, but on the far side of a bay: flip it to the opposite half
        t0, t1 = t0 + np.pi, t1 + np.pi
    rx = rng.uniform(60, 110)
    ry = rx * rng.uniform(0.5, 0.62)
    pts, out = ring_pts(rng, rx, ry, t0, t1)
    pts = pts - pts.mean(0) + (P.ox, P.oy)
    escarp(pen, pts, -out if inner else out, rng.uniform(*FACE_H))


def mesa(pen, P, rng):
    """A complete flat-topped plateau: face along the front, plain rim behind."""
    rx = rng.uniform(45, 85)
    ry = rx * rng.uniform(0.5, 0.65)
    pts, out = ring_pts(rng, rx, ry, -0.5 * np.pi, 1.5 * np.pi + 0.15, wob=0.07)
    escarp(pen, pts + (P.ox, P.oy), out, rng.uniform(*FACE_H))


def _inside(poly, x, y):
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def terraced_hill(pen, P, rng):
    """Two to four stacked plateaus: a whole hill or mountain in one piece."""
    tiers = int(rng.integers(2, 5))
    rx = rng.uniform(70, 100)
    ry = rx * rng.uniform(0.5, 0.62)
    cx, cy = P.ox, P.oy + 15
    rings = []
    for k in range(tiers):
        hmax = rng.uniform(*FACE_H)
        pts, out = ring_pts(rng, rx, ry, -0.5 * np.pi, 1.5 * np.pi + 0.12, wob=0.06)
        rings.append((pts + (cx, cy), out, hmax))
        # next tier: smaller, raised by its own face height, nudged sideways
        shrink = rng.uniform(0.55, 0.75)
        cx += rng.normal(0, rx * 0.12)
        cy -= ry * (1 - shrink) + hmax * 0.6
        rx, ry = rx * shrink, ry * shrink
        if rx < 16:
            break
    for k, (pts, out, hmax) in enumerate(rings):
        above = rings[k + 1][0] if k + 1 < len(rings) else None
        if above is None:
            escarp(pen, pts, out, hmax)
            continue
        # hide this tier's rim wherever the tier above sits on top of it
        vis = np.array([not _inside(above, x, y) for x, y in pts])
        i = 0
        while i < len(pts):
            if vis[i]:
                j = i
                while j < len(pts) and vis[j]:
                    j += 1
                if j - i > 2:
                    escarp(pen, pts[i:j], out[i:j], hmax, spurs=False)
                i = j
            else:
                i += 1


# --- canyons -----------------------------------------------------------------------------------

def _canyon_from_centre(pen, centre, width, rng):
    n = len(centre)
    nr = right_normals(centre)  # right of travel
    left = centre - nr * (width / 2)[:, None]
    right = centre + nr * (width / 2)[:, None]
    hmax = min(FACE_H[1] * 1.3, float(np.median(width)) * 0.85)
    # each rim drops toward the centre line
    escarp(pen, left, nr, hmax, density=1.2)
    escarp(pen, right, -nr, hmax, density=1.2, line_w=1.1)


def _width_profile(rng, n, lo=16, hi=30):
    t = np.linspace(0, 1, n)
    w = rng.uniform(lo, hi) * (1 + 0.25 * np.sin(TAU * t * rng.uniform(0.5, 1.5) + rng.uniform(0, TAU)))
    if rng.random() < 0.4:  # one end pinches shut
        w *= np.clip((t if rng.random() < 0.5 else 1 - t) * 3, 0.15, 1)
    return w


def canyon(pen, P, rng):
    """Straight-ish canyon run, left to right."""
    centre = edge_path(rng, rng.uniform(110, 210), rng.normal(0, 0.25), wiggle=0.45) + (P.ox, P.oy)
    _canyon_from_centre(pen, centre, _width_profile(rng, len(centre)), rng)


def canyon_bend(pen, P, rng):
    """Canyon turning through a curve (quarter or half turn), for joining runs without rotating."""
    role = str(rng.choice(["front", "back", "left_end", "right_end", "front_left", "front_right", "back_left", "back_right"]))
    t0, t1 = (v * np.pi for v in ARC_ROLES[role])
    rx = rng.uniform(65, 100)
    ry = rx * rng.uniform(0.5, 0.62)
    centre, _ = ring_pts(rng, rx, ry, t0, t1, wob=0.03)
    centre = centre - centre.mean(0) + (P.ox, P.oy)
    _canyon_from_centre(pen, centre, _width_profile(rng, len(centre), 16, 26), rng)
