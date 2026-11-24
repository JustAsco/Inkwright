"""Projected 3D primitives drawn in ink: boxes, cylinders, cones, pyramids, earth mounds."""
import math

import numpy as np

from .ink import TAU


def hull(pts):
    pts = sorted({(round(float(x), 3), round(float(y), 3)) for x, y in pts})
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def box(pen, P, x0, y0, x1, y1, h0, h1, shade=True, top=True, inset=0.0, w=None, lean=(0.0, 0.0)):
    """Axis-aligned box, optionally tapered (`inset`) and leaning (`lean` shifts the top). Returns the top face."""
    W0 = pen.width
    w = w or W0 * 0.85
    i = inset
    lx, ly = lean

    def T(x, y):
        return P(x + lx, y + ly, h1)

    south = [P(x0, y0, h0), P(x1, y0, h0), T(x1 - i, y0 + i), T(x0 + i, y0 + i)]
    east = [P(x1, y0, h0), P(x1, y1, h0), T(x1 - i, y1 - i), T(x1 - i, y0 + i)]
    topf = [T(x0 + i, y0 + i), T(x1 - i, y0 + i), T(x1 - i, y1 - i), T(x0 + i, y1 - i)]
    faces = [south, east] + ([topf] if top else [])
    for f in faces:
        pen.cv.erase(f)
    if shade:
        pen.hatch(east, 82, 1.6, W0 * 0.7, coverage=(0.75, 1.0))
    for f in faces:
        pen.closed(f, w, overshoot=0.4, wobble=0.2)
    return topf


def cylinder(pen, P, X, Y, r, h0, h1, shade=True, top=True, n=40, w=None):
    W0 = pen.width
    w = w or W0
    a = P.az

    def C(t, h):
        return P(X + r * math.cos(t), Y + r * math.sin(t), h)

    front = np.linspace(a + math.pi, a + TAU, n)
    body = [C(t, h0) for t in front] + [C(t, h1) for t in front[::-1]]
    pen.cv.erase(body)
    ring = [C(t, h1) for t in np.linspace(0, TAU, n, endpoint=False)]
    if top:
        pen.cv.erase(ring)
    if shade:
        ts = np.linspace(a + TAU - 1.1, a + TAU, 14)
        pen.hatch([C(t, h0) for t in ts] + [C(t, h1) for t in ts[::-1]], 84, 1.5, W0 * 0.7, coverage=(0.75, 1.0))
    pen.stroke([C(t, h0) for t in front], w * 1.1, taper=(0.05, 0.05))
    pen.line(C(a + math.pi, h0), C(a + math.pi, h1), w, sketch=0)
    pen.line(C(a, h0), C(a, h1), w, sketch=0)
    if top:
        pen.closed(ring, w * 0.9)
    else:
        pen.stroke([C(t, h1) for t in front], w, taper=(0.05, 0.05))
    return C


def cone(pen, P, X, Y, r, h0, apex_h, shade=True, broken=0.0):
    """Conical roof. `broken` in (0,1): truncated at that fraction with a ragged hole and rafters."""
    rng, W0 = pen.rng, pen.width
    a = P.az
    ring_t = np.linspace(0, TAU, 48, endpoint=False)
    base = [P(X + r * math.cos(t), Y + r * math.sin(t), h0) for t in ring_t]
    apex = P(X, Y, apex_h)
    if broken:
        f = broken
        rt = r * (1 - f)
        ht = h0 + (apex_h - h0) * f
        jag = rng.uniform(-0.12, 0.12, len(ring_t)) * (apex_h - h0)
        top = [P(X + rt * math.cos(t), Y + rt * math.sin(t), ht + j) for t, j in zip(ring_t, jag)]
        shape = hull(base + top)
    else:
        top = None
        shape = hull(base + [apex])
    pen.cv.erase(shape)
    front = np.arange(a + math.pi + 0.08, a + TAU - 0.04, 0.16)
    for t in front:
        b = P(X + r * math.cos(t), Y + r * math.sin(t), h0)
        if top is not None:
            k = int(np.argmin(np.abs((ring_t - t + math.pi) % TAU - math.pi)))
            e = top[k]
        else:
            e = apex
        L = rng.uniform(0.75, 1.0)
        pen.stroke([b, (b[0] + (e[0] - b[0]) * L, b[1] + (e[1] - b[1]) * L)], W0 * 0.55, taper=(0.05, 0.5), wobble=0.1)
    ts = np.linspace(a + TAU - 0.9, a + TAU, 10)
    sh = hull([P(X + r * math.cos(t), Y + r * math.sin(t), h0) for t in ts] + ([apex] if top is None else [top[0]]))
    pen.hatch(sh, 70, 1.4, W0 * 0.6, coverage=(0.7, 1.0)) if shade else None
    if top is not None:
        pen.blot(top, jitter=0.2)
        for k in rng.choice(len(top), size=int(rng.integers(3, 7)), replace=False):
            p = top[k]
            L = rng.uniform(0.3, 0.8)
            pen.beam(p, (p[0] + (apex[0] - p[0]) * L, p[1] + (apex[1] - p[1]) * L), 1.4)
    pen.closed(shape, W0 * 1.0)
    pen.stroke([P(X + r * math.cos(t), Y + r * math.sin(t), h0) for t in np.linspace(a + math.pi, a + TAU, 30)],
               W0 * 1.2, taper=(0.05, 0.05))


def pyramid(pen, P, x0, y0, x1, y1, h0, apex_h, shade=True):
    rng, W0 = pen.rng, pen.width
    apex = P((x0 + x1) / 2, (y0 + y1) / 2, apex_h)
    south = [P(x0, y0, h0), P(x1, y0, h0), apex]
    east = [P(x1, y0, h0), P(x1, y1, h0), apex]
    for f in (south, east):
        pen.cv.erase(f)
    for t in np.linspace(0.1, 0.9, 7):
        b = P(x0 + (x1 - x0) * t, y0, h0)
        pen.stroke([b, (b[0] + (apex[0] - b[0]) * 0.9, b[1] + (apex[1] - b[1]) * 0.9)], W0 * 0.5, taper=(0.05, 0.5))
    if shade:
        pen.hatch(east, 75, 1.5, W0 * 0.65, coverage=(0.7, 1.0))
    for f in (south, east):
        pen.closed(f, W0 * 0.9)


def mound(pen, P, X, Y, R, Ry, Hh, texture=True):
    """Lumpy earth dome on an oval footprint. Returns (silhouette, front_rim, top_env_fn)."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    ph = rng.uniform(0, TAU, 3)
    amp = rng.uniform(0.03, 0.1, 3)

    def rad(t):
        return 1 + sum(amp[i] * math.sin((i + 2) * t + ph[i]) for i in range(3))

    def hgt(r, t):
        return Hh * max(0.0, 1 - r * r) ** 0.7 * (1 + 0.12 * math.sin(3 * t + ph[0]))

    def S(r, t):
        rr = r * rad(t)
        return P(X + R * rr * math.cos(t), Y + Ry * rr * math.sin(t), hgt(r, t))

    ts = np.linspace(0, TAU, 72, endpoint=False)
    pts = np.array([S(r, t) for r in np.linspace(0, 1, 14) for t in ts])
    a = P.az
    rim = sorted([S(1, t) for t in np.linspace(a + math.pi, a + TAU, 40)], key=lambda p: p[0])
    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 60)
    bw = (xs[1] - xs[0]) * 0.8
    top = [(x, pts[np.abs(pts[:, 0] - x) < bw, 1].min()) for x in xs]
    top[0] = rim[0]
    top[-1] = rim[-1]
    poly = top + rim[::-1]
    cv.erase(poly)
    if texture:
        for _ in range(int(R * 0.9)):
            t = rng.uniform(a + math.pi - 0.3, a + TAU + 0.3)
            r = rng.uniform(0.25, 0.95)
            p0, p1 = S(r, t - 0.09), S(r, t + 0.09)
            right = math.cos(t - a) > 0.3
            if right or rng.random() < 0.5:
                pen.stroke([p0, S(r, t), p1], W0 * (0.6 if right else 0.4), taper=(0.4, 0.4), wobble=0)
            else:
                cv.disc(*S(r, t), 0.45)
        ts2 = np.linspace(a + TAU - 0.9, a + TAU + 0.2, 10)
        shade = [S(1, t) for t in ts2] + [S(0.55, t) for t in ts2[::-1]]
        pen.hatch(shade, 70, 2.0, W0 * 0.5, coverage=(0.4, 0.9))
    pen.stroke(top, W0 * (0.9 if texture else 1.0), taper=(0.05, 0.05), wobble=0.5)
    pen.stroke(rim, W0 * 0.8, taper=(0.25, 0.25), wobble=0.5)
    txs = np.array([p[0] for p in top])
    tys = np.array([p[1] for p in top])
    return poly, rim, (lambda x: float(np.interp(x, txs, tys)))


def broken_cylinder(pen, P, X, Y, r, h0, h1, jag=6.0, shade=True, n=48, r_top=None):
    """Column / limb / tower stump whose top is snapped off unevenly (radius may taper). Returns C(t, h)."""
    rng, W0 = pen.rng, pen.width
    a = P.az
    ph = rng.uniform(0, TAU, 2)

    def hat(t):
        return h1 - jag * (0.5 + 0.3 * math.sin(2 * t + ph[0]) + 0.2 * math.sin(5 * t + ph[1]))

    rt = r if r_top is None else r_top

    def C(t, h):
        rr = r + (rt - r) * (h - h0) / max(1e-6, h1 - h0)
        return P(X + rr * math.cos(t), Y + rr * math.sin(t), h)

    front = np.linspace(a + math.pi, a + TAU, n // 2)
    ring = [C(t, hat(t)) for t in np.linspace(0, TAU, n, endpoint=False)]
    body = [C(t, h0) for t in front] + [C(t, hat(t)) for t in front[::-1]]
    pen.cv.erase(body)
    pen.cv.erase(ring)
    if shade:
        ts = np.linspace(a + TAU - 1.1, a + TAU, 12)
        pen.hatch([C(t, h0) for t in ts] + [C(t, hat(t)) for t in ts[::-1]], 84, 1.5, W0 * 0.7, coverage=(0.75, 1.0))
    pen.closed(ring, W0 * 0.8, wobble=0.15)
    for _ in range(rng.integers(0, 3)):  # fracture lines on the break
        t = rng.uniform(0, TAU)
        pen.stroke([P(X, Y, hat(t)), C(t, hat(t))], W0 * 0.45, taper=(0.3, 0.3))
    pen.stroke([C(t, h0) for t in front], W0 * 1.1, taper=(0.05, 0.05))
    pen.line(C(a + math.pi, h0), C(a + math.pi, hat(a + math.pi)), W0, sketch=0)
    pen.line(C(a, h0), C(a, hat(a)), W0, sketch=0)
    return C


def lying_cylinder(pen, P, xa, xb, y, r, h0=0.0):
    """Fallen drum/column lying east-west on the ground."""
    W0 = pen.width
    ts = np.linspace(0, TAU, 28, endpoint=False)

    def end(x):
        return [P(x, y + r * math.cos(t), h0 + r + r * math.sin(t)) for t in ts]

    shape = hull(end(xa) + end(xb))
    pen.cv.erase(shape)
    face = end(xb)
    pen.cv.erase(face)
    lower = [P(xa + (xb - xa) * f, y - r * 0.3, h0 + r * 0.25) for f in (0.05, 0.95)]
    pen.hatch(hull(end(xa)[18:] + end(xb)[18:] + lower), 10, 1.6, W0 * 0.55, coverage=(0.5, 0.9))
    pen.closed(shape, W0 * 0.9)
    pen.closed(face, W0 * 0.8)
