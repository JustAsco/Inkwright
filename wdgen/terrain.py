"""Landscape symbols: farmland patchworks and stackable hand-inked cliffs, mesas, and canyons."""
import math

import numpy as np

from .ink import TAU, chaikin, clip_halfplane
from .solids import box, cone


def _bilerp(q, u, v):
    a, b, c, d = (np.asarray(p, float) for p in q)  # q = [(0,0), (1,0), (1,1), (0,1)] corners
    return tuple(a * (1 - u) * (1 - v) + b * u * (1 - v) + c * u * v + d * (1 - u) * v)


# --------------------------------------------------------------------------- fields

def _haystack(pen, P, x, y, r, h):
    """Properly shaded conical haystack/stook (lit left, hatched right, never a solid black blob)."""
    rng, W0 = pen.rng, pen.width
    a = P.az
    ts = np.linspace(0, TAU, 24, endpoint=False)
    base = [P(x + r * math.cos(t), y + r * math.sin(t), 0) for t in ts]
    front = [P(x + r * math.cos(t), y + r * math.sin(t), 0) for t in np.linspace(a + math.pi, a + TAU, 14)]
    apex = P(x + rng.normal(0, r * 0.08), y, h)
    # gentle rounded dome-cone silhouette
    l_p, r_p = front[0], front[-1]
    l_mid = (l_p[0] * 0.62 + apex[0] * 0.38 - r * 0.12, l_p[1] * 0.55 + apex[1] * 0.45)
    r_mid = (r_p[0] * 0.62 + apex[0] * 0.38 + r * 0.12, r_p[1] * 0.55 + apex[1] * 0.45)
    poly = chaikin([l_p, l_mid, apex, r_mid, r_p], 1)
    full = list(poly) + list(front[::-1])
    pen.cv.erase(full)
    # right-side straw shadow hatching
    sh = clip_halfplane(full, np.array([1.0, 0.05]), apex[0] + r * 0.05)
    if len(sh) >= 3:
        pen.hatch(sh, 76, 1.4, W0 * 0.45, coverage=(0.6, 0.95))
    # a few light straw lines descending from the crown
    for f in (0.25, 0.5):
        b_pt = front[int(f * (len(front) - 1))]
        pen.stroke([apex, (apex[0] * 0.3 + b_pt[0] * 0.7, apex[1] * 0.3 + b_pt[1] * 0.7)],
                   W0 * 0.36, taper=(0.2, 0.5), wobble=0.1)
    pen.stroke(poly, W0 * 0.78, taper=(0.05, 0.05), wobble=0.12)
    pen.stroke(front, W0 * 0.85, taper=(0.1, 0.1), wobble=0.1)


def _field_cell(pen, P, quad, kind, extras):
    rng, W0 = pen.rng, pen.width

    def S(u, v, h=0.0):
        x, y = _bilerp(quad, u, v)
        return P(x, y, h)

    along_u = rng.random() < 0.5

    def rows(spacing_world):
        size = math.dist(quad[0], quad[3] if along_u else quad[1])
        n = max(2, int(size / spacing_world))
        return np.linspace(0.08, 0.92, n)

    def row_line(v):
        return (S(0.05, v), S(0.95, v)) if along_u else (S(v, 0.05), S(v, 0.95))

    if kind == "furrow":
        for idx, v in enumerate(rows(3.6)):
            a, b = row_line(v)
            # alternate subtle furrow depth and occasional natural break
            if rng.random() < 0.18:
                m = rng.uniform(0.3, 0.7)
                mid = (a[0] + (b[0] - a[0]) * m, a[1] + (b[1] - a[1]) * m)
                pen.stroke([a, mid], W0 * 0.42, taper=(0.25, 0.35), wobble=0.3)
            else:
                pen.stroke([a, b], W0 * (0.48 if idx % 2 == 0 else 0.36), taper=(0.2, 0.2), wobble=0.32)
        if rng.random() < 0.35:
            extras.append("stooks")
    elif kind == "grain":
        for v in rows(4.4):
            a, b = row_line(v)
            n = max(3, int(math.dist(a, b) / 3.0))
            for i in range(n):
                if rng.random() < 0.18:
                    continue
                f = (i + rng.uniform(0.2, 0.8)) / n
                x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                hgt = rng.uniform(2.2, 3.6)
                lean = rng.uniform(0.3, 1.1)  # gentle breeze lean
                pen.stroke([(x, y), (x + lean * 0.4, y - hgt * 0.6), (x + lean, y - hgt)],
                           W0 * 0.40, taper=(0.05, 0.7), smooth=1, wobble=0)
    elif kind == "orchard":
        pts = []
        nu, nv = int(rng.integers(3, 5)), int(rng.integers(2, 4))
        for i in range(nu):
            for j in range(nv):
                u, v = (i + 0.5) / nu + rng.normal(0, 0.03), (j + 0.5) / nv + rng.normal(0, 0.03)
                x, y = _bilerp(quad, np.clip(u, 0.12, 0.88), np.clip(v, 0.12, 0.88))
                pts.append((P.depth(x, y), x, y))
        for _, x, y in sorted(pts, reverse=True):
            sx, sy = P(x, y)
            pen.stroke([(sx, sy), (sx, sy - 3.8)], W0 * 0.65, taper=(0.0, 0.3), wobble=0)
            pen.bush(sx, sy - 2.6, rng.uniform(3.2, 4.2), outline=True)
    elif kind == "vineyard":
        for v in rows(5.4):
            a, b = row_line(v)
            n = max(2, int(math.dist(a, b) / 6.0))
            tops = []
            for i in range(n + 1):
                x, y = a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n
                pen.stroke([(x, y), (x, y - 4.0)], W0 * 0.48, taper=(0, 0.1), wobble=0)
                tops.append((x, y - 3.3))
            pen.stroke(tops, W0 * 0.36, taper=(0.1, 0.1), wobble=0.18)
            for _ in range(n * 2):
                f = rng.uniform(0.05, 0.95)
                pen.dab(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f - 2.8, 1.25, w=W0 * 0.55)
    elif kind == "fallow":
        for _ in range(int(math.dist(quad[0], quad[2]) * 0.55)):
            pen.cv.disc(*S(rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)), 0.42)
        for _ in range(rng.integers(2, 5)):
            pen.tuft(*S(rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)), rng.uniform(2.8, 4.2))
    else:  # pasture
        for _ in range(max(2, int(math.dist(quad[0], quad[2]) * 0.10))):
            pen.tuft(*S(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)), rng.uniform(2.8, 4.2))
        if rng.random() < 0.5:
            extras.append("haystack")


def _boundary(pen, P, a, b, kind):
    rng, W0 = pen.rng, pen.width
    A, B = P(*a), P(*b)
    L = math.dist(A, B)
    if kind == "line":
        pen.stroke([A, B], W0 * 0.52, taper=(0.15, 0.15), wobble=0.45)
    elif kind == "hedge":
        # Low ink-drawn hedgerow (no glaring un-outlined white blobs on parchment)
        n = max(3, int(L / 2.8))
        for i in range(n + 1):
            f = i / n
            x, y = A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f
            pen.dab(x + rng.normal(0, 0.5), y - 1.0 + rng.normal(0, 0.35), rng.uniform(1.4, 2.1),
                    angle=rng.normal(0.1, 0.25), w=W0 * 0.85)
            if rng.random() < 0.7:
                pen.dab(x + rng.normal(0, 0.6), y - 2.6 + rng.normal(0, 0.4), rng.uniform(1.1, 1.7),
                        angle=rng.normal(-0.1, 0.25), w=W0 * 0.62)
    elif kind == "wall":
        # Low rustic dry-stone field wall
        poly = [A, B, (B[0], B[1] - 2.4), (A[0], A[1] - 2.4)]
        pen.cv.erase(poly)
        pen.stroke([A, B], W0 * 0.75, taper=(0.05, 0.05), wobble=0.35)
        pen.stroke([(A[0], A[1] - 2.4), (B[0], B[1] - 2.4)], W0 * 0.58, taper=(0.05, 0.05), wobble=0.35)
        n_st = max(2, int(L / 3.8))
        for i in range(n_st):
            f = (i + rng.uniform(0.2, 0.8)) / n_st
            x, y = A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f
            pen.stroke([(x, y), (x, y - 2.4)], W0 * 0.38, taper=(0, 0), wobble=0)
    elif kind == "fence":
        n = max(2, int(L / 6.5))
        tops = []
        for i in range(n + 1):
            x, y = A[0] + (B[0] - A[0]) * i / n, A[1] + (B[1] - A[1]) * i / n
            pen.stroke([(x, y), (x, y - 4.4)], W0 * 0.55, taper=(0, 0.1), wobble=0)
            tops.append((x, y - 3.6))
        pen.stroke(tops, W0 * 0.42, taper=(0.05, 0.05), wobble=0.2)
        pen.stroke([(p[0], p[1] + 1.8) for p in tops], W0 * 0.42, taper=(0.05, 0.05), wobble=0.2)
    elif kind == "track":
        for off in (-1.7, 1.7):
            d = np.array([B[1] - A[1], A[0] - B[0]]) / max(L, 1e-6) * off
            pen.stroke([np.add(A, d), np.add(B, d)], W0 * 0.45, taper=(0.15, 0.15), wobble=0.55)


def fields(pen, P, rng):
    """Organic farmland patchwork: plowed furrows, grain, orchards, vineyards, hedges, stone walls."""
    nx, ny = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    W, D = rng.uniform(140, 195), rng.uniform(90, 128)
    cx = np.concatenate([[0], np.cumsum(rng.uniform(0.7, 1.35, nx))])
    cy = np.concatenate([[0], np.cumsum(rng.uniform(0.7, 1.35, ny))])
    cx, cy = cx / cx[-1] * W - W / 2, cy / cy[-1] * D - D / 2
    # gentle organic vertex jitter across interior & perimeter so plots feel hand-surveyed
    V = [[(cx[i] + rng.normal(0, 4.5 if 0 < i < nx else 2.2),
           cy[j] + rng.normal(0, 4.5 if 0 < j < ny else 2.2))
          for j in range(ny + 1)] for i in range(nx + 1)]
    skip = set()
    if rng.random() < 0.45:
        skip.add((0, ny - 1))
    if rng.random() < 0.45:
        skip.add((nx - 1, 0))
    if nx * ny >= 6 and rng.random() < 0.35:
        skip.add((0, 0) if rng.random() < 0.5 else (nx - 1, ny - 1))

    extras, cells = [], []
    for i in range(nx):
        for j in range(ny):
            if (i, j) in skip:
                continue
            quad = [V[i][j], V[i + 1][j], V[i + 1][j + 1], V[i][j + 1]]
            kind = str(rng.choice(["furrow", "furrow", "grain", "grain", "orchard", "vineyard", "fallow", "pasture"],
                                  p=[0.22, 0.10, 0.18, 0.10, 0.11, 0.10, 0.09, 0.10]))
            cells.append((quad, kind))
    for quad, kind in sorted(cells, key=lambda c: -P.depth(*np.mean(c[0], axis=0))):
        ex = []
        _field_cell(pen, P, quad, kind, ex)
        for e in ex:
            for _ in range(rng.integers(2, 5) if e == "stooks" else rng.integers(1, 3)):
                u, v = rng.uniform(0.18, 0.82, 2)
                extras.append((quad, e, u, v))
    edges = set()
    for i in range(nx):
        for j in range(ny):
            if (i, j) in skip:
                continue
            for e in (((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)),
                      ((i, j + 1), (i + 1, j + 1)), ((i, j), (i, j + 1))):
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
    # Perimeter bushes only next to non-skipped front cells (never floating in empty corner)
    front_cells = [V[i][0] for i in range(nx) if (i, 0) not in skip]
    for _ in range(rng.integers(0, 3)):
        if not front_cells:
            break
        vx, vy = front_cells[int(rng.integers(0, len(front_cells)))]
        x, y = vx + rng.uniform(4, W / nx - 4), vy - rng.uniform(3, 8)
        items.append((P.depth(x, y), "bush", (x, y)))
    for _, kind, args in sorted(items, key=lambda it: -it[0]):
        if kind == "edge":
            _boundary(pen, P, *args)
        elif kind == "stooks":
            _haystack(pen, P, args[0], args[1], rng.uniform(2.2, 3.0), rng.uniform(5.2, 7.2))
        elif kind == "haystack":
            r = rng.uniform(4.2, 5.8)
            _haystack(pen, P, args[0], args[1], r, r * rng.uniform(1.45, 1.85))
        elif kind == "bush":
            pen.bush(*P(*args), rng.uniform(4.5, 7.2), outline=True)


# --------------------------------------------------------------------------- cliffs & canyons
# Hand-inked map escarpments designed to STACK SEAMLESSLY without rotating:
# - Shared FACE_H scale across all cliffs, steep cliffs, curves, bays, mesas, terraced hills, canyons.
# - Open-ended runs damp wiggle & face height smoothly near endpoints so overlapping joins disappear.
# - Faceted rock buttresses, directional crevice hatching & horizontal strata replace comb needles.

FACE_H = (11.0, 16.5)  # shared escarpment height range for seamless stacking


def _smooth_noise(rng, n, scale=8, lo=0.0, hi=1.0):
    k = max(1, int(scale))
    raw = rng.uniform(lo, hi, n + 2 * k)
    return np.convolve(raw, np.ones(k) / k, "same")[k:k + n]


def right_normals(pts):
    """Unit normals on the right-hand side of travel (screen coords: L->R path gives 'down')."""
    d = np.gradient(np.asarray(pts, float), axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
    return np.stack([-d[:, 1], d[:, 0]], 1)


def edge_path(rng, length, heading, wiggle=0.20, kinks=0.0, step=2.0):
    """Smooth open edge centred on origin whose endpoint tangents return cleanly to `heading`."""
    n = max(6, int(length / step))
    s = np.arange(n) * step
    u = np.linspace(0.0, 1.0, n)
    # smooth window ensures clean, predictable tangent headings at both stackable open ends
    win = np.sin(np.pi * u) ** 0.75
    ph = rng.uniform(0, TAU, 2)
    ang = heading + win * wiggle * (
        0.65 * np.sin(s / rng.uniform(20, 34) + ph[0])
        + 0.35 * np.sin(s / rng.uniform(9, 15) + ph[1])
    )
    for _ in range(rng.poisson(kinks * length / 70)):
        if n > 10:
            i = int(rng.integers(4, n - 5))
            ang[i - 2:i + 3] += win[i] * rng.choice([-1, 1]) * rng.uniform(0.28, 0.55) * np.array([0.3, 0.7, 1, 0.7, 0.3])
    d = np.stack([np.cos(ang), np.sin(ang)], 1) * step
    pts = chaikin(np.vstack([[0, 0], np.cumsum(d, 0)]), 2)[::4]
    return pts - pts.mean(0)


def jagged(rng, pts, low, seg=(6.0, 11.0), amp=1.25, notch=0.14, step=1.5, closed=False):
    """Re-draw a smooth edge as crisp rocky straight facets without self-intersections or end knots."""
    pts = np.asarray(pts, float)
    low = np.asarray(low, float)
    if closed and np.hypot(*(pts[0] - pts[-1])) > 0.5:
        pts = np.vstack([pts, pts[:1]])
        low = np.vstack([low, low[:1]])
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-6:
        return pts, low
    marks = [0.0]
    while marks[-1] < total:
        marks.append(marks[-1] + rng.uniform(*seg))
    marks[-1] = total

    verts, vs = [], []
    n_m = len(marks)
    for k, m in enumerate(marks):
        i = int(np.searchsorted(s, m).clip(0, len(pts) - 1))
        if closed and k == n_m - 1:
            verts.append(verts[0].copy())
            vs.append(m)
            continue
        if not closed:
            # smooth endpoint damping guarantees crisp, stackable open ends
            end_w = math.sin(math.pi * (m / total)) ** 0.55
            off = 0.0 if k in (0, n_m - 1) else rng.normal(0, amp) * end_w
        else:
            off = rng.normal(0, amp)
        p_base = pts[i] + low[i] * off
        verts.append(p_base)
        vs.append(m)
        # Forward-stepping geological crag notch (strictly monotone along tangent, never hooks back!)
        if 1 < k < n_m - 2 and rng.random() < notch:
            m_next = min(total - 1.0, m + rng.uniform(1.6, 2.8))
            if m_next < marks[k + 1] - 1.5:
                j = int(np.searchsorted(s, m_next).clip(0, len(pts) - 1))
                jog = rng.choice([-1, 1]) * rng.uniform(1.3, 2.4)
                verts.append(pts[j] + low[j] * (off + jog))
                vs.append(m_next)

    verts = np.array(verts)
    out, src = [verts[0]], [0.0]
    for a, b, va, vb in zip(verts[:-1], verts[1:], vs[:-1], vs[1:]):
        L = float(np.hypot(*(b - a)))
        if L < 1e-6:
            continue
        for f in np.linspace(0, 1, max(2, int(L / step)) + 1)[1:]:
            out.append(a + (b - a) * f)
            src.append(va + (vb - va) * f)
    idx = np.searchsorted(s, np.array(src)).clip(0, len(pts) - 1)
    return np.array(out), low[idx]


def _clamp_floor(x, y_bot, floor_clip):
    if floor_clip is None:
        return y_bot
    fy = floor_clip(x)
    return min(y_bot, fy) if fy is not None else y_bot


def _face(pen, pts, h, density=1.0, floor_clip=None, open_ends=(True, True)):
    """Sculpted rocky escarpment face: stepped wash, primary rock ribs, facet hatching & strata."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 3:
        return
    h = np.asarray(h, float).copy()
    # Gentle taper at open stackable ends so overlapping pieces blend seamlessly
    u = np.linspace(0.0, 1.0, n)
    if open_ends[0]:
        h *= np.clip(u / 0.16, 0.08, 1.0) ** 0.55
    if open_ends[1]:
        h *= np.clip((1.0 - u) / 0.16, 0.08, 1.0) ** 0.55

    # Compute effective bottom y per vertex (respecting optional canyon near-rim occlusion!)
    h_eff = np.array([max(0.0, _clamp_floor(pts[i, 0], pts[i, 1] + h[i], floor_clip) - pts[i, 1]) for i in range(n)])
    if h_eff.max() < 1.8:
        return

    # Layered rock wash following natural craggy talus contour (never a uniform flat airbrush blob)
    talus_mod = _smooth_noise(rng, n, 4, 0.78, 1.18)
    for k, v in enumerate((215, 170, 125, 80, 38)):
        f = (k + 1) / 5.0 * 1.18
        bot = [(pts[i, 0], _clamp_floor(pts[i, 0], pts[i, 1] + h_eff[i] * f * talus_mod[i], floor_clip))
               for i in range(n - 1, -1, -1)]
        cv.wash(list(pts) + bot, v)

    # Compute local rim slope & convexity to cluster hatching into 3D sunlit vs shadowed rock facets!
    d = np.gradient(pts, axis=0)
    # East-facing / downward-right slope catches shadow (light from west/top-left)
    slope_shadow = np.clip(d[:, 1] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6), -1.0, 1.0)
    # Natural alternating geological facet bands along the cliff face
    facet_wave = np.sin(np.linspace(0, max(2.0, n / 7.5) * math.pi, n) + rng.uniform(0, TAU))
    shadow_field = np.clip(0.55 + 0.35 * facet_wave + 0.30 * slope_shadow, 0.12, 1.0)

    # Pass 1: Primary bold rock buttress ribs + clustered shadowed facet crevices
    i = 1
    rib_positions = []
    while i < n - 1:
        hi = h_eff[i]
        if hi > 2.2:
            x, y = pts[i]
            sh_w = shadow_field[i]
            is_rib = rng.random() < (0.36 if sh_w > 0.52 else 0.24)
            if is_rib:
                # Primary vertical/slightly-splayed rock rib descending from a cliff promontory
                L = hi * rng.uniform(0.72, 1.05)
                y_end = _clamp_floor(x, y + L, floor_clip)
                L = max(0.0, y_end - y)
                if L > 2.2:
                    lean = rng.normal(0.55, 0.65)
                    mid_x = x + lean * 0.45 + rng.normal(0, 0.35)
                    pen.stroke([(x, y), (mid_x, y + L * 0.52), (x + lean, y + L)],
                               W0 * rng.uniform(0.95, 1.22), taper=(0.0, 0.92), smooth=1, wobble=0.1)
                    rib_positions.append((x, y, L))
                    # Subtle dark under-lip shadow accent just to the right of a strong rib
                    if hi > 5.5 and rng.random() < 0.35:
                        dx = rng.uniform(1.2, 2.2)
                        cv.fill([(x - 0.3, y), (x + dx, y), (x + dx * 0.4, min(y_end, y + rng.uniform(2.2, 4.0)))])
            else:
                # Shorter secondary crevice / facet hatching (skip occasionally on sunlit facets for crisp rock relief!)
                if sh_w < 0.34 and rng.random() < 0.45:
                    i += max(1, int(rng.uniform(1.6, 2.4) / density))
                    continue
                L = hi * rng.uniform(0.32, 0.72) * (0.65 + 0.45 * sh_w)
                y_end = _clamp_floor(x, y + L, floor_clip)
                L = max(0.0, y_end - y)
                if L > 1.8:
                    lean = rng.normal(0.3, 0.5)
                    pen.stroke([(x, y), (x + lean * 0.5, y + L * 0.5), (x + lean, y + L)],
                               W0 * (0.76 if sh_w > 0.6 else 0.54), taper=(0.0, 0.9), smooth=1, wobble=0.05)
        step_i = rng.uniform(1.4, 2.3) if shadow_field[min(i, n - 1)] > 0.55 else rng.uniform(2.1, 3.2)
        i += max(1, int(step_i / density))

    # Pass 2: Subtle rock-ledge / strata breaks tucked cleanly inside broad cliff facets
    for _ in range(max(0, int(n / 18))):
        k = int(rng.integers(4, max(5, n - 4)))
        if h_eff[k] > 10.0 and abs(d[k, 1]) < abs(d[k, 0]) * 0.55 and rng.random() < 0.55:
            x, y = pts[k]
            dy = h_eff[k] * rng.uniform(0.32, 0.55)
            y_s = _clamp_floor(x, y + dy, floor_clip)
            if y_s - y > 3.5:
                sw = rng.uniform(2.8, 5.0)
                pen.stroke([(x - sw * 0.45, y_s), (x + sw * 0.55, y_s + rng.normal(0, 0.35))],
                           W0 * 0.38, taper=(0.35, 0.35), wobble=0.1)


def _spur(pen, pts, i):
    """Subtle rock ledge notch along a turning/side escarpment rim."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    x, y = pts[i]
    w = rng.uniform(3.5, 6.0)
    tri = [(x - w / 2, y), (x + w / 2, y + rng.uniform(-0.4, 0.4)), (x + rng.uniform(-0.8, 0.8), y + rng.uniform(2.5, 4.2))]
    cv.wash(tri, 165)
    for dx in np.linspace(-w * 0.25, w * 0.25, int(rng.integers(1, 3))):
        pen.stroke([(x + dx, y), (x + dx + 0.3, y + rng.uniform(2.0, 3.6))], W0 * 0.62, taper=(0.05, 0.85), wobble=0)


def _edge_line(pen, pts, w=1.18, closed=False):
    if closed:
        pen.closed(pts, pen.width * w, wobble=0.12, pressure=0.22)
    else:
        # Cleanly tapered open ends so overlapping pieces merge without blunt ink stubs
        pen.stroke(pts, pen.width * w, taper=(0.15, 0.15), taper_max=18, wobble=0.1, pressure=0.25, min_w=0.06)


def escarp(pen, pts, low, hmax, mask=None, spurs=True, line=True, line_w=1.18,
           density=1.0, jag=1.25, closed=False, floor_clip=None, open_ends=(True, True)):
    """Edge whose low ground lies along `low` (unit vectors). The face shows where the drop faces
    the viewer (low side pointing down-screen) and fades smoothly as the edge turns away."""
    rng = pen.rng
    pts = np.asarray(pts, float)
    low = np.asarray(low, float)
    if jag:
        pts, low = jagged(rng, pts, low, amp=jag, closed=closed)
    n = len(pts)
    strength = np.clip(low[:, 1], 0.0, 1.0)
    pad_mode = "wrap" if closed else "edge"
    strength = np.convolve(np.pad(strength, 6, mode=pad_mode), np.ones(13) / 13, "valid")
    if mask is not None:
        strength = strength * np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(mask)), mask)
    h = hmax * (strength ** 0.78) * _smooth_noise(rng, n, 6, 0.62, 1.0)
    on = h > 1.6
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i > 3:
                oe = (open_ends[0] and i <= 2, open_ends[1] and j >= n - 2)
                _face(pen, pts[i:j], h[i:j], density, floor_clip=floor_clip, open_ends=oe)
            i = j
        else:
            i += 1
    if spurs:
        for _ in range(rng.integers(0, 3)):
            if n > 8:
                k = int(rng.integers(3, n - 3))
                if not on[k] and low[k][1] > -0.25:
                    _spur(pen, pts, k)
    if line:
        _edge_line(pen, pts, line_w, closed=closed)
    return pts


def _relief_mask(rng, n):
    """Organic geological height modulation along a cliff run (keeps the cliff continuous & stackable)."""
    if rng.random() < 0.45:
        return np.ones(n)
    t = np.linspace(0, 1, n)
    # gentle swells & occasional shallow saddle, never turning off half the cliff into bare wire
    sw = 0.78 + 0.22 * np.sin(TAU * t * rng.uniform(0.8, 1.6) + rng.uniform(0, TAU))
    if rng.random() < 0.35:
        c = rng.uniform(0.25, 0.75)
        sw *= 1.0 - 0.55 * np.exp(-((t - c) / rng.uniform(0.10, 0.18)) ** 2)
    return np.clip(sw, 0.25, 1.0)


def cliff(pen, P, rng, heading=None):
    """Open stackable escarpment run, low ground below; gently horizontal so pieces chain cleanly."""
    if heading is None:
        heading = float(rng.normal(0.0, 0.11))
    pts = edge_path(rng, rng.uniform(100, 185), heading, wiggle=0.16, kinks=0.22) + (P.ox, P.oy)
    escarp(pen, pts, right_normals(pts), rng.uniform(*FACE_H), _relief_mask(rng, len(pts)))


def cliff_steep(pen, P, rng):
    """Open stackable steep diagonal escarpment run matching corner & canyon slopes."""
    cliff(pen, P, rng, heading=float(rng.choice([1, -1]) * rng.uniform(0.72, 1.02)))


# --- curves: arcs of an elliptical outline (perspective-squashed), convex or concave -------------

ARC_ROLES = {
    # name: (start, end) angle in screen coords (0 = right, pi/2 = bottom/front)
    "front": (0.15, 0.85), "back": (1.15, 1.85),
    "left_end": (0.5, 1.5), "right_end": (-0.5, 0.5),
    "front_left": (0.5, 1.0), "front_right": (0.0, 0.5),
    "back_left": (1.0, 1.5), "back_right": (1.5, 2.0),
}


def ring_pts(rng, rx, ry, t0, t1, wob=0.05, step=2.0, closed=False):
    """Points on a gently irregular superellipse from t0 to t1, with outward unit normals."""
    span = abs(t1 - t0)
    n = max(10, int(span * (rx + ry) / 2 / step))
    t = np.linspace(t0, t1, n, endpoint=not closed)
    ph = rng.uniform(0, TAU, 3)
    raw_wob = np.sin(3 * t + ph[0]) + 0.55 * np.sin(5 * t + ph[1]) + 0.25 * np.sin(9 * t + ph[2])
    if not closed:
        # damp wobble smoothly at open arc endpoints so adjacent pieces join cleanly
        win = np.sin(np.linspace(0, np.pi, n)) ** 0.65
        raw_wob = raw_wob * win
    r = 1.0 + wob * raw_wob
    e = 2.0 / rng.uniform(2.5, 3.1)  # superellipse: clean straightish runs, turning near corners
    c, sn = np.cos(t), np.sin(t)
    pts = np.stack([rx * r * np.sign(c) * np.abs(c) ** e, ry * r * np.sign(sn) * np.abs(sn) ** e], 1)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    nrm = right_normals(pts) * (-1 if t1 > t0 else 1)  # increasing angle travels clockwise on screen
    if closed:
        nrm[-1] = nrm[0]
    return pts, nrm


def cliff_curve(pen, P, rng, role=None, inner=False):
    """Stackable curved plateau outline piece. Convex: high ground inside. Inner: concave bay."""
    role = role or str(rng.choice(list(ARC_ROLES)))
    t0, t1 = (v * np.pi for v in ARC_ROLES[role])
    if inner:
        t0, t1 = t0 + np.pi, t1 + np.pi
    rx = rng.uniform(68, 96)
    ry = rx * rng.uniform(0.52, 0.60)
    pts, out = ring_pts(rng, rx, ry, t0, t1, wob=0.045, closed=False)
    pts = pts - pts.mean(0) + (P.ox, P.oy)
    escarp(pen, pts, -out if inner else out, rng.uniform(*FACE_H), jag=1.15)


def mesa(pen, P, rng):
    """Complete flat-topped sandstone/limestone mesa plateau with crisp closed rim (ready for landmarks on top)."""
    rx = rng.uniform(48, 84)
    ry = rx * rng.uniform(0.50, 0.62)
    pts, out = ring_pts(rng, rx, ry, -0.5 * np.pi, 1.5 * np.pi, wob=0.065, closed=True)
    pts = pts + (P.ox, P.oy)
    escarp(pen, pts, out, rng.uniform(*FACE_H), jag=1.25, closed=True, open_ends=(False, False))


def _inside(poly, x, y):
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def terraced_hill(pen, P, rng):
    """Two to four stacked craggy plateau terraces forming a stepped hill or mountain butte."""
    tiers = int(rng.integers(2, 5))
    rx = rng.uniform(72, 98)
    ry = rx * rng.uniform(0.50, 0.60)
    cx, cy = P.ox, P.oy + 18
    drift = rng.choice([-1, 1]) * rng.uniform(0.06, 0.14)  # cohesive geological lean
    rings = []
    for k in range(tiers):
        hmax = rng.uniform(FACE_H[0] * 0.92, FACE_H[1] * 0.95)
        pts, out = ring_pts(rng, rx, ry, -0.5 * np.pi, 1.5 * np.pi, wob=0.065, closed=True)
        # Pre-jag each closed ring once so occlusion tests match the drawn rim exactly!
        jpts, jout = jagged(rng, pts + (cx, cy), out, amp=1.2, closed=True)
        rings.append((jpts, jout, hmax, cx, cy, rx, ry))
        shrink = rng.uniform(0.58, 0.74)
        cx += drift * rx + rng.normal(0, rx * 0.05)
        cy -= ry * (1.0 - shrink) * 0.85 + hmax * 0.55
        rx, ry = rx * shrink, ry * shrink
        if rx < 17:
            break

    # Draw bottom-to-top, clearing each upper plateau's footprint so lower cliff washes stay crisp!
    for k, (jpts, jout, hmax, rcx, rcy, rrx, rry) in enumerate(rings):
        if k > 0:
            pen.cv.clear(jpts)  # cleanly hide lower tier rim & wash behind the upper terrace!
        escarp(pen, jpts, jout, hmax, jag=0.0, closed=True, open_ends=(False, False), spurs=(k == len(rings) - 1))


# --- stackable canyons -------------------------------------------------------------------------

def _safe_offset_rim(centre, nr, offset):
    """Offset a polyline smoothly along `nr` while preventing inner-bend loops / needle spikes."""
    raw = centre + nr * offset[:, None]
    n = len(raw)
    if n < 5:
        return raw
    # Detect & relax any backward fold-overs where rim travel reverses against centreline tangent
    tc = np.gradient(centre, axis=0)
    tc /= np.maximum(np.hypot(tc[:, 0], tc[:, 1]), 1e-6)[:, None]
    out = raw.copy()
    for _ in range(8):
        changed = False
        for i in range(1, n - 1):
            d_prev = out[i] - out[i - 1]
            d_next = out[i + 1] - out[i]
            if float(d_prev @ tc[i]) < 0.35 or float(d_next @ tc[i]) < 0.35:
                out[i] = 0.5 * (out[i - 1] + out[i + 1])
                changed = True
        if not changed:
            break
    return out


def _canyon_from_centre(pen, centre, width, rng):
    """Draw a two-rimmed canyon with near-rim floor occlusion and cleanly tapered stackable ends."""
    W0 = pen.width
    centre = np.asarray(centre, float)
    nr = right_normals(centre)  # points toward the right-of-travel (typically south/down-screen)
    hw = np.asarray(width, float) * 0.5
    left = _safe_offset_rim(centre, nr, -hw)
    right = _safe_offset_rim(centre, nr, hw)

    # Identify which rim sits lower on screen (nearer to the viewer) to clip the far canyon wall!
    # Also handle bends where either rim can be the far/facing wall depending on local orientation.
    left_j, nr_l = jagged(rng, left, nr, amp=1.15, closed=False)
    right_j, nr_r = jagged(rng, right, -nr, amp=1.15, closed=False)

    def make_clipper(rim_pts):
        """Return y-upper-bound lookup for points falling directly above a front-facing rim."""
        rx, ry = rim_pts[:, 0], rim_pts[:, 1]
        # Build piecewise-min y envelope over x bins only where the rim spans horizontally
        xmin, xmax = float(rx.min()), float(rx.max())
        if xmax - xmin < 4.0:
            return None

        def clip_y(x):
            if x < xmin or x > xmax:
                return None
            # Find segments crossing x and take the highest (smallest y) rim crossing below the wall
            best = None
            for i in range(len(rx) - 1):
                x0, x1 = rx[i], rx[i + 1]
                if (x0 <= x <= x1) or (x1 <= x <= x0):
                    if abs(x1 - x0) < 1e-6:
                        yy = min(ry[i], ry[i + 1])
                    else:
                        f = (x - x0) / (x1 - x0)
                        yy = ry[i] + (ry[i + 1] - ry[i]) * f
                    if best is None or yy > best:
                        best = yy
            return None if best is None else best - 0.6

        return clip_y

    # Determine which rim has the lower mean y (higher on screen = far rim facing viewer)
    hmax = min(FACE_H[1] * 1.15, float(np.median(width)) * 0.82)
    clip_for_left = make_clipper(right_j) if left_j[:, 1].mean() < right_j[:, 1].mean() else None
    clip_for_right = make_clipper(left_j) if right_j[:, 1].mean() < left_j[:, 1].mean() else None

    # Draw far wall first (clipped cleanly at near rim!), then smooth canyon floor accent, then near rim!
    escarp(pen, left_j, nr_l, hmax, density=1.15, jag=0.0, floor_clip=clip_for_left)
    escarp(pen, right_j, nr_r, hmax, density=1.15, line_w=1.12, jag=0.0, floor_clip=clip_for_right)

    # Delicate smooth dry-bed meander nestled softly inside the canyon middle (faded well before open ends)
    n = len(centre)
    if n > 14 and rng.random() < 0.55:
        i0, i1 = max(3, int(n * 0.22)), min(n - 3, int(n * 0.78))
        u_f = np.linspace(0, 1, i1 - i0)
        meander = 1.6 * np.sin(TAU * u_f * rng.uniform(0.7, 1.4) + rng.uniform(0, TAU))
        floor_pts = centre[i0:i1] + nr[i0:i1] * meander[:, None]
        pen.stroke(floor_pts, W0 * 0.34, taper=(0.38, 0.38), smooth=2, wobble=0.15)


def _width_profile(rng, n, base=22.0, spread=3.0):
    """Stackable canyon width: organic swell in the middle, consistent open mouth at BOTH ends."""
    t = np.linspace(0, 1, n)
    win = np.sin(np.pi * t) ** 0.8
    w0 = rng.uniform(base - spread * 0.5, base + spread * 0.5)
    swell = win * rng.uniform(-0.16, 0.22) * w0 * np.sin(TAU * t * rng.uniform(0.6, 1.3) + rng.uniform(0, TAU))
    return np.clip(w0 + swell, 16.0, 28.0)


def canyon(pen, P, rng):
    """Stackable straight-ish canyon run with gently horizontal open ends matching Canyon Bends."""
    heading = float(rng.normal(0.0, 0.08))
    centre = edge_path(rng, rng.uniform(120, 200), heading, wiggle=0.25, kinks=0.10) + (P.ox, P.oy)
    _canyon_from_centre(pen, centre, _width_profile(rng, len(centre), base=22.0), rng)


def canyon_bend(pen, P, rng):
    """Stackable canyon curve (quarter or half turn) matching straight canyon widths & ends."""
    role = str(rng.choice(["front", "back", "left_end", "right_end",
                           "front_left", "front_right", "back_left", "back_right"]))
    t0, t1 = (v * np.pi for v in ARC_ROLES[role])
    rx = rng.uniform(72, 102)
    ry = rx * rng.uniform(0.54, 0.62)
    centre, _ = ring_pts(rng, rx, ry, t0, t1, wob=0.025, closed=False)
    centre = centre - centre.mean(0) + (P.ox, P.oy)
    _canyon_from_centre(pen, centre, _width_profile(rng, len(centre), base=21.5), rng)
