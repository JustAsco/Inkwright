"""Beast lairs: sculpted crag caves, cliff caverns, sunken dens, grassy burrows, and woven eyries."""
import math

import numpy as np

from . import flora
from .ink import TAU, blob_pts, chaikin, clip_halfplane, ellipse_pts
from .solids import box, mound


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
    """Craggy rock silhouette from left base to right base (screen coords, y down)."""
    m = m or int(rng.integers(7, 11))
    xs = np.linspace(-Bw / 2, Bw / 2, m)
    xs[1:-1] += rng.normal(0, Bw / m * 0.14, m - 2)
    env = np.clip(np.sin(np.pi * (xs / Bw + 0.5)), 0, None) ** rng.uniform(0.52, 0.95)
    hs = Hh * env * (1 + rng.normal(0, 0.14 if rocky else 0.05, m))
    hs[0] = hs[-1] = 0
    pts = np.stack([cx + xs, by - hs], 1)
    if rocky:
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            if rng.random() < 0.65:
                mid = (a + b) / 2 + rng.normal(0, 2.4, 2)
                out.append(mid)
            out.append(b)
        return np.array(out)
    return chaikin(pts, 3)


def base_curve(rng, x0, x1, by, sag=3.0, n=9):
    """Gently bowed ground contour from x1 back to x0 (closes a silhouette)."""
    xs = np.linspace(x1, x0, n)
    t = (xs - x0) / max(1e-6, x1 - x0)
    ys = by + sag * np.sin(np.pi * t) + rng.normal(0, 0.55, n)
    ys[0] = ys[-1] = by
    return np.stack([xs, ys], 1)


def ground_line(pen, base):
    """Broken, tapered strokes along a base contour plus subtle stray ground marks."""
    rng, W0 = pen.rng, pen.width
    n = len(base)
    i = 0
    while i < n - 1:
        j = min(n, i + int(rng.integers(3, 6)))
        if rng.random() < 0.82:
            pen.stroke(base[i:j], W0 * 0.88, taper=(0.3, 0.3), wobble=0.28)
        i = j - 1 if rng.random() < 0.7 else j
    for x0, sgn in ((base[-1][0], -1), (base[0][0], 1)):
        for _ in range(rng.integers(0, 3)):
            x = x0 + sgn * rng.uniform(2, 14)
            y = base[0][1] + rng.uniform(-1, 3)
            pen.stroke([(x, y), (x + sgn * rng.uniform(4, 9), y + rng.normal(0, 0.3))], W0 * 0.52, taper=(0.4, 0.4))


def rock(pen, pts, shade=True, facets=True, base_sag=None, strata=True):
    """Sculpted fantasy rock massif with geological ridge spines, faceted shadow planes & rock strata."""
    rng, W0 = pen.rng, pen.width
    pts = np.asarray(pts, float)
    by = max(pts[0][1], pts[-1][1])
    base = base_curve(rng, pts[0][0], pts[-1][0], by, sag=base_sag or rng.uniform(2.5, 5.0), n=15)
    poly = list(pts) + list(base[1:-1])
    pen.cv.erase(poly)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin = float(pts[:, 1].min())
    W = max(10.0, xmax - xmin)
    H = max(10.0, by - ymin)

    # 1. Major descending ridge spines from upper peaks (carving the massif into 3D rock facets!)
    ridges = []
    if facets and len(pts) >= 5:
        # Pick 2-3 well-spaced high crest vertices along the silhouette
        cand = [i for i in range(1, len(pts) - 1) if pts[i][1] < by - H * 0.35]
        cand.sort(key=lambda i: pts[i][1])  # highest first
        chosen = []
        for idx in cand:
            if all(abs(pts[idx][0] - pts[c][0]) > W * 0.19 for c in chosen):
                chosen.append(idx)
            if len(chosen) >= 3:
                break
        for idx in sorted(chosen):
            x0, y0 = pts[idx]
            frac_x = (x0 - xmin) / W
            # Ridge descends down and slightly outward/east, breaking into 3-4 angular crag segments
            segs = int(rng.integers(3, 5))
            rpath = [(x0, y0)]
            cx, cy = x0, y0
            drift = (frac_x - 0.45) * W * 0.16 + rng.normal(0, 3.0)
            drop = (by - y0) * rng.uniform(0.68, 0.94)
            for k in range(1, segs + 1):
                f = k / segs
                nx = x0 + drift * f + rng.normal(0, 2.2)
                ny = y0 + drop * f
                if _inside(poly, nx, ny):
                    rpath.append((nx, ny))
                    cx, cy = nx, ny
            if len(rpath) >= 2:
                ridges.append(rpath)

    # 2. Primary east-flank shadow hatching + secondary shadow planes tucked right of each ridge!
    if shade:
        # Main right-side shadow plane bounded by the rightmost major ridge (or natural 58% split)
        split = xmin + W * rng.uniform(0.54, 0.66)
        sh_main = clip_halfplane(poly, (0.97, 0.24), split * 0.97 + (ymin + H * 0.5) * 0.24)
        if len(sh_main) >= 3:
            pen.hatch(sh_main, rng.uniform(62, 74), rng.uniform(1.85, 2.4), W0 * 0.58, coverage=(0.65, 1.0))
        # Lighter secondary shadow hatching tucked right beside every interior ridge spine!
        for rpath in ridges:
            if len(rpath) < 2:
                continue
            rx_top = rpath[0][0]
            if rx_top > split - W * 0.08:
                continue
            r_right = [(px + rng.uniform(6.0, 12.0), py + 2.0) for px, py in rpath]
            facet_poly = rpath + r_right[::-1]
            # Clip facet to stay inside the rock polygon roughly
            facet_in = [p for p in facet_poly if _inside(poly, p[0], p[1]) or p in rpath]
            if len(facet_in) >= 3:
                pen.hatch(facet_poly, rng.uniform(58, 72), rng.uniform(2.5, 3.2), W0 * 0.44, coverage=(0.45, 0.85))

    # 3. Draw the bold sculpted ridge spines + secondary crag fractures
    for rpath in ridges:
        pen.stroke(rpath, W0 * 0.82, taper=(0.05, 0.78), wobble=0.2)

    if facets:
        # Additional shorter crag cracks descending from silhouette angles
        for i in range(1, len(pts) - 1):
            if rng.random() < 0.28:
                x, y = pts[i]
                L = rng.uniform(7, min(20, H * 0.35))
                ang = math.pi * 0.5 + rng.normal(0.15, 0.45)
                p_end = (x + math.cos(ang) * L, y + math.sin(ang) * L)
                if _inside(poly, *p_end):
                    mid = ((x + p_end[0]) * 0.5 + rng.normal(0, 1.4), (y + p_end[1]) * 0.5)
                    pen.stroke([(x, y + 1.0), mid, p_end], W0 * 0.56, taper=(0.1, 0.82), wobble=0.15)

    if strata:
        # Subtle horizontal/tilted geological rock bedding strata across the rock face
        tilt = rng.uniform(-0.18, 0.18)
        for _ in range(int(rng.integers(3, 7))):
            x = rng.uniform(xmin + W * 0.12, xmax - W * 0.22)
            y_top = float(np.interp(x, pts[:, 0], pts[:, 1]))
            if by - y_top < 14:
                continue
            y = y_top + rng.uniform(6, (by - y_top) * 0.82)
            L = rng.uniform(8, min(22, W * 0.28))
            p2 = (x + L, y + L * tilt + rng.normal(0, 1.0))
            if _inside(poly, x, y) and _inside(poly, *p2):
                pen.stroke([(x, y), ((x + p2[0]) * 0.5, (y + p2[1]) * 0.5 + rng.normal(0, 0.8)), p2],
                           W0 * 0.52, taper=(0.25, 0.55), wobble=0.18)

    # Crisp outer rock silhouette + grounded base contour
    pen.stroke(pts, W0 * 1.22, taper=(0.05, 0.05), wobble=0.26)
    ground_line(pen, base)
    if rng.random() < 0.45:
        i0 = int(rng.integers(0, max(1, len(pts) // 2)))
        pen.stroke(pts[i0:i0 + len(pts) // 2] + rng.normal(0, 0.5, 2), W0 * 0.44, taper=(0.3, 0.3))


def boulder(pen, x, y, r):
    rng, W0 = pen.rng, pen.width
    pts = blob_pts(rng, x, y - r * 0.58, r, n=9, irregular=0.18, squash=0.68)
    pen.cv.erase(pts)
    pen.hatch(clip_halfplane(list(pts), (0.92, 0.38), x * 0.92 + (y - r * 0.58) * 0.38 + r * 0.12),
              64, 1.9, W0 * 0.52, coverage=(0.55, 1.0))
    pen.closed(pts, W0 * 0.9, overshoot=0)
    # facet ridge line across the boulder
    pen.stroke([(x - r * 0.28, y - r * 1.05), (x - r * 0.05, y - r * 0.55), (x - r * 0.22, y - r * 0.15)],
               W0 * 0.5, taper=(0.2, 0.65), wobble=0.12)


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
            pen.stroke([base, mid, tip], W0 * 1.5, taper=(0.0, 0.9), smooth=2, wobble=0)
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
        pen.stroke(pts, W0 * 0.68, taper=(0.05, 0.6), smooth=2, wobble=0.1)
    pen.stroke(spine, W0 * 0.95, taper=(0.2, 0.2), smooth=1)
    for i in range(int(L / 3)):
        p = np.array(spine[0]) + (np.array(spine[-1]) - np.array(spine[0])) * i / (L / 3)
        pen.stroke([p + (0, -1.2), p + (0, 1.2)], W0 * 0.42, taper=(0, 0))


def remains(pen, cx, cy, spread, n=None):
    rng = pen.rng
    n = n or int(rng.integers(3, 7))
    items = []
    for _ in range(n):
        x, y = cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread * 0.2, spread * 0.25)
        r = rng.random()
        kind = "skull" if r < 0.18 else "ribs" if r < 0.28 else "bone"
        items.append((y, x, kind))
    for y, x, kind in sorted(items):
        if kind == "bone":
            bone(pen, x, y, rng.uniform(7, 12.5), rng.uniform(0, math.pi))
        elif kind == "skull":
            skull(pen, x, y, rng.uniform(5, 6.8))
        else:
            ribcage(pen, x, y, rng.uniform(13, 20), flip=rng.random() < 0.5)


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


def claw_marks(pen, x, y, size=15):
    rng, W0 = pen.rng, pen.width
    a = rng.uniform(1.0, 1.4)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    for i in (-1, 0, 1):
        c = np.array([x, y]) + n * i * 2.8
        pts = [c - d * size / 2, c + n * 0.7, c + d * size / 2]
        pen.stroke(pts, W0 * 0.92, taper=(0.4, 0.5), smooth=2, wobble=0)


# --------------------------------------------------------------------------- caves

def crack(pen, x, y, ang, L, w=None, branch=True, inside=None):
    rng = pen.rng
    pts = [(x, y)]
    segs = int(rng.integers(3, 6))
    for _ in range(segs):
        ang += rng.normal(0, 0.45)
        x, y = x + math.cos(ang) * L / segs, y + math.sin(ang) * L / segs * 0.8
        if inside is not None and not _inside(inside, x, y):
            break
        pts.append((x, y))
    if len(pts) < 2:
        return
    pen.stroke(pts, w or pen.width * 0.85, taper=(0.0, 0.85), wobble=0)
    if branch and len(pts) > 2 and rng.random() < 0.55:
        bx, by = pts[int(rng.integers(1, len(pts) - 1))]
        crack(pen, bx, by, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.0), L * 0.45, w, False, inside)


# --------------------------------------------------------------------------- 3D (az=28°, el=34°) caves, burrows & warrens

_SUN_3D = np.array([-0.82, 0.24, 0.52])
_SUN_3D /= float(np.linalg.norm(_SUN_3D))


def _cam_vec(P):
    v = np.array([P.sa * P.ce, -P.ca * P.ce, P.se], float)
    return v / float(np.linalg.norm(v))


def _place_tree(pen, x, y, h, kind=None):
    kind = kind or str(pen.rng.choice(["spruce", "spruce", "round"]))
    if kind == "spruce":
        flora.spruce(pen, x, y, h)
    else:
        flora.round_tree(pen, x, y, h)


def _trail_3d(pen, P, rng, x0, y0, dir_xy, width, length, z_fn=None, stones=True):
    """Winding hand-inked 3D ground footpath projected through P(x, y, z) along the oblique 3/4 ground plane."""
    W0 = pen.width
    d = np.asarray(dir_xy, float)
    d /= max(1e-6, float(np.hypot(*d)))
    nrm = np.array([-d[1], d[0]])
    bend = rng.uniform(-0.35, 0.35) * length
    sway = rng.uniform(-0.18, 0.18) * length
    ts = np.linspace(0, 1, 10)
    l_pts, r_pts, c_pts = [], [], []
    for t in ts:
        c = np.array([x0, y0]) + d * (length * t) + nrm * (bend * (t ** 1.3) + sway * math.sin(math.pi * t))
        z = float(z_fn(c[0], c[1])) if z_fn else 0.0
        hw = width * (0.46 + 0.24 * (1 - t * 0.4))
        c_pts.append((c[0], c[1], z))
        l_pts.append(P(c[0] - nrm[0] * hw, c[1] - nrm[1] * hw, z))
        r_pts.append(P(c[0] + nrm[0] * hw, c[1] + nrm[1] * hw, z))
    pen.stroke(chaikin(l_pts, 1), W0 * 0.54, taper=(0.08, 0.88), wobble=0.20)
    pen.stroke(chaikin(r_pts, 1), W0 * 0.54, taper=(0.08, 0.88), wobble=0.20)
    # Subtle worn dirt stipples along the 3D track
    for _ in range(int(length * 0.35)):
        idx = int(rng.integers(0, len(c_pts)))
        cx, cy, cz = c_pts[idx]
        off = rng.uniform(-width * 0.38, width * 0.38)
        sx, sy = P(cx + nrm[0] * off, cy + nrm[1] * off, cz)
        pen.cv.disc(sx + rng.normal(0, 0.8), sy + rng.normal(0, 0.6), rng.uniform(0.35, 0.62))
    if stones:
        for _ in range(rng.integers(2, 5)):
            idx = int(rng.integers(1, len(c_pts) - 1))
            cx, cy, cz = c_pts[idx]
            side = rng.choice([-1, 1])
            off = side * (width * 0.52 + rng.uniform(2.0, 6.5))
            sx, sy = P(cx + nrm[0] * off, cy + nrm[1] * off, cz)
            if rng.random() < 0.55:
                pen.stone(sx, sy, rng.uniform(1.2, 2.4))
            else:
                pen.tuft(sx, sy, rng.uniform(3.0, 4.8))


def _hill_3d(pen, P, rng, h_fn, rx, ry, cx=0.0, cy=0.0, rocky=True, shade_scale=1.0, peaks=(), fill_body=None):
    """True 3D heightfield hill/crag projected through P(x, y, h) — viewed from 34° above & sideways!"""
    W0 = pen.width
    cam = _cam_vec(P)
    eps = 1.5
    fill_body = rocky if fill_body is None else fill_body

    def grad(x, y):
        gx = (h_fn(x + eps, y) - h_fn(x - eps, y)) / (2 * eps)
        gy = (h_fn(x, y + eps) - h_fn(x, y - eps)) / (2 * eps)
        return gx, gy

    def visible(x, y, z):
        for step in (7.0, 16.0, 28.0):
            qx, qy, qz = x + cam[0] * step, y + cam[1] * step, z + cam[2] * step
            if h_fn(qx, qy) > qz + 1.2:
                return False
        return True

    # 1. Sample 3D surface grid to build exact projected 3/4 upper silhouette + tilted ground footprint!
    surf_s = []
    for r in np.linspace(0.0, 1.0, 15):
        nt = 1 if r == 0 else 44
        for t in np.linspace(0, TAU, nt, endpoint=False):
            wx, wy = cx + rx * r * math.cos(t), cy + ry * r * math.sin(t)
            wz = h_fn(wx, wy)
            surf_s.append(P(wx, wy, wz))
    surf_arr = np.asarray(surf_s, float)
    sxmin, sxmax = float(surf_arr[:, 0].min()), float(surf_arr[:, 0].max())

    nb = 32
    edges = np.linspace(sxmin, sxmax, nb + 1)
    horizon = []
    for i in range(nb):
        sel = (surf_arr[:, 0] >= edges[i] - 1.5) & (surf_arr[:, 0] <= edges[i + 1] + 1.5)
        if np.any(sel):
            sub = surf_arr[sel]
            k = int(np.argmin(sub[:, 1]))
            horizon.append((float(sub[k, 0]), float(sub[k, 1])))
    if len(horizon) >= 4:
        horizon = list(chaikin(horizon, 1 if rocky else 2))

    # Exact 3/4 tilted front footprint rim (in world ground plane z=0!)
    t_front = np.linspace(P.az + math.pi, P.az + TAU, 26)
    front_rim = [P(cx + rx * math.cos(t), cy + ry * math.sin(t), 0.0) for t in t_front]
    if front_rim[0][0] > front_rim[-1][0]:
        front_rim = front_rim[::-1]
    if fill_body:
        # Rocky mountain crags mask background map features cleanly (and draw grounded rocky foothill ink along front_rim!)
        pen.cv.erase(horizon + front_rim[::-1])

    # 2. True 3D downhill fall-line slope hachures on shadowed east/south-east slopes!
    nr = 12 if rocky else 8
    for ir in range(1, nr):
        r = 0.12 + 0.82 * (ir / nr)
        nt = max(14, int((34 if rocky else 26) * r * shade_scale))
        for it in range(nt):
            t = (it + rng.uniform(-0.26, 0.26)) * TAU / nt
            wx, wy = cx + rx * r * math.cos(t), cy + ry * r * math.sin(t)
            wz = h_fn(wx, wy)
            if wz < (1.6 if rocky else 2.2):
                continue
            gx, gy = grad(wx, wy)
            sl = math.hypot(gx, gy)
            if sl < (0.12 if rocky else 0.09):
                continue
            N = np.array([-gx, -gy, 1.0])
            N /= float(np.linalg.norm(N))
            if float(N @ cam) < 0.06 or not visible(wx, wy, wz):
                continue
            lit = float(N @ _SUN_3D)
            if lit < (0.38 if rocky else 0.30):
                darkness = max(0.0, (0.38 - lit) / 0.65)
                if rng.random() > (0.50 + 0.50 * darkness) * shade_scale:
                    continue
                step_l = rng.uniform(5.0, 9.5) * (0.85 + 0.45 * darkness) * (1.0 if rocky else 0.80)
                pts = [P(wx, wy, wz)]
                px, py = wx, wy
                for _ in range(3):
                    ggx, ggy = grad(px, py)
                    gsl = math.hypot(ggx, ggy)
                    if gsl < 0.05:
                        break
                    px -= (ggx / gsl) * (step_l / 3.0)
                    py -= (ggy / gsl) * (step_l / 3.0)
                    pz = h_fn(px, py)
                    if pz <= 0.3 or not visible(px, py, pz):
                        break
                    pts.append(P(px, py, pz))
                if len(pts) >= 2:
                    w_h = W0 * ((0.58 if rocky else 0.40) * (0.78 + 0.35 * darkness))
                    pen.stroke(pts, w_h, taper=(0.12, 0.84), smooth=1, wobble=0.12)
            elif not rocky and lit > 0.48 and rng.random() < 0.18:
                tx, ty = -gy / sl, gx / sl
                cl = rng.uniform(5.0, 10.5)
                p0 = P(wx - tx * cl * 0.5, wy - ty * cl * 0.5, wz)
                p1 = P(wx + tx * cl * 0.5, wy + ty * cl * 0.5, h_fn(wx + tx * cl * 0.5, wy + ty * cl * 0.5))
                pen.stroke([p0, p1], W0 * 0.34, taper=(0.45, 0.45), wobble=0.10)

    # 3. Rocky watershed ridges & 3D strata contours OR grassy top-down contour arcs!
    if rocky:
        for pkx, pky in peaks:
            pkz = h_fn(pkx, pky)
            if pkz < 12 or not visible(pkx, pky, pkz):
                continue
            for ang_off in (rng.uniform(-1.45, -0.95), rng.uniform(-0.45, 0.05)):
                rpts = [P(pkx, pky, pkz)]
                px, py = pkx, pky
                for _ in range(1, 7):
                    ang = ang_off + rng.normal(0, 0.16)
                    px += math.cos(ang) * (rx * 0.14)
                    py += math.sin(ang) * (ry * 0.14)
                    pz = h_fn(px, py)
                    if pz < 1.5 or not visible(px, py, pz):
                        break
                    rpts.append(P(px, py, pz))
                if len(rpts) >= 3:
                    pen.stroke(rpts, W0 * 0.84, taper=(0.08, 0.86), wobble=0.22)
        for _ in range(rng.integers(4, 7)):
            t0 = rng.uniform(P.az + math.pi * 0.9, P.az + math.pi * 1.65)
            r0 = rng.uniform(0.32, 0.82)
            wx, wy = cx + rx * r0 * math.cos(t0), cy + ry * r0 * math.sin(t0)
            wz = h_fn(wx, wy)
            if wz < 4 or not visible(wx, wy, wz):
                continue
            gx, gy = grad(wx, wy)
            sl = max(0.1, math.hypot(gx, gy))
            tx, ty = -gy / sl, gx / sl
            pts = []
            for s in (-7.5, 0.0, 7.5):
                qx, qy = wx + tx * s, wy + ty * s
                qz = h_fn(qx, qy)
                if qz > 1.2 and visible(qx, qy, qz):
                    pts.append(P(qx, qy, qz))
            if len(pts) >= 2:
                pen.stroke(pts, W0 * 0.48, taper=(0.3, 0.55), wobble=0.14)
    else:
        for r_c in (0.34, 0.60):
            t_a = P.az + math.pi * rng.uniform(0.85, 1.08)
            t_b = t_a + rng.uniform(0.85, 1.45)
            arc = []
            for t in np.linspace(t_a, t_b, 8):
                wx, wy = cx + rx * r_c * math.cos(t), cy + ry * r_c * math.sin(t)
                wz = h_fn(wx, wy)
                if wz > 2.0 and visible(wx, wy, wz):
                    arc.append(P(wx, wy, wz))
            if len(arc) >= 3 and rng.random() < 0.78:
                pen.stroke(chaikin(arc, 1), W0 * 0.36, taper=(0.45, 0.45), wobble=0.14)

    # 4. Crisp upper horizon crest with open fading 3/4 ground toes
    if len(horizon) >= 3:
        tl = (horizon[0][0] - rng.uniform(5, 10), horizon[0][1] + rng.uniform(1.5, 3.5))
        tr = (horizon[-1][0] + rng.uniform(5, 10), horizon[-1][1] + rng.uniform(1.5, 3.5))
        pen.stroke([tl] + horizon + [tr], W0 * (1.12 if rocky else 0.90), taper=(0.68, 0.68), wobble=0.22)
    if fill_body:
        # Grounded broken rocky foothill contour strokes hugging the tilted front rim (eliminates un-inked white pancake edge!)
        i = 0
        while i < len(front_rim) - 2:
            j = min(len(front_rim), i + int(rng.integers(4, 8)))
            if rng.random() < 0.85:
                pen.stroke(front_rim[i:j], W0 * 0.72, taper=(0.32, 0.32), wobble=0.22)
            i = j - 1 if rng.random() < 0.4 else j
    else:
        for seg in (front_rim[1:5], front_rim[-5:-1]):
            if len(seg) >= 3 and rng.random() < 0.65:
                pen.stroke(seg, W0 * 0.40, taper=(0.5, 0.5), wobble=0.16)

    return visible


def _cave_portal_3d(pen, P, rng, cx, cy, z0, mw, mh, facing="south", depth=20.0,
                    timber=False, stone_lintel=False, roots=False, trail=True, rocky_brow=True, brow=True):
    """True 3D oblique sideways cave portal projected through P(x, y, h) (facing 'south' down-left OR 'east' down-right)."""
    W0, cv = pen.width, pen.cv
    if facing == "south":
        u = np.array([1.0, 0.0])       # mouth width runs along +x (tilted 16.5° down-right on screen!)
        v_in = np.array([0.0, 1.0])    # tunnel plunges north (+y, up-right on screen!)
    else:
        u = np.array([0.0, 1.0])       # mouth width runs along +y (tilted 46° up-right on screen!)
        v_in = np.array([-1.0, 0.0])   # tunnel plunges west (-x, left-up on screen!)

    n = 16
    ts = np.linspace(math.pi, 0.0, n)
    arch_w = []  # 3D world coords (wx, wy, wz) along outer arch
    arch_s = []  # projected screen coords
    for i, t in enumerate(ts):
        j = 1.0 if i in (0, n - 1) else (rng.uniform(0.92, 1.08) if rocky_brow else rng.uniform(0.96, 1.04))
        du = math.cos(t) * (mw * 0.5) * j
        dh = (math.sin(t) ** 0.76) * mh * j
        wx, wy, wz = cx + u[0] * du, cy + u[1] * du, z0 + dh
        arch_w.append((wx, wy, wz))
        arch_s.append(P(wx, wy, wz))

    # 1. Overhanging 3D rock brow hood jutting slightly outward (-v_in) over the mouth!
    if brow:
        brow_s = []
        for i, (wx, wy, wz) in enumerate(arch_w):
            f_top = math.sin(math.pi * i / max(1, n - 1)) ** 1.15
            du_s = math.cos(ts[i]) * (2.2 + 2.4 * f_top)
            dh_s = (1.6 + 3.6 * f_top)
            out_s = 2.8 * (0.4 + 0.6 * f_top)
            brow_s.append(P(wx + u[0] * du_s - v_in[0] * out_s,
                            wy + u[1] * du_s - v_in[1] * out_s,
                            wz + dh_s))
        brow_poly = brow_s + arch_s[::-1]
        cv.erase(brow_poly)
        pen.hatch(brow_poly, 78 if facing == "south" else 84, 1.75, W0 * 0.52, coverage=(0.65, 1.0))
        pen.stroke(chaikin(brow_s, 1), W0 * 0.92, taper=(0.35, 0.35), wobble=0.18)

    # 2. Recessed 3D inner tunnel vault receding along +v_in!
    inner_s = []
    for i, (wx, wy, wz) in enumerate(arch_w):
        # Shrink slightly toward center as it goes deeper inside
        du_in = (wx - cx) * 0.78 + v_in[0] * depth
        dy_in = (wy - cy) * 0.78 + v_in[1] * depth
        dz_in = (wz - z0) * 0.76
        inner_s.append(P(cx + du_in, cy + dy_in, z0 + dz_in))

    # Erase full mouth silhouette first
    cv.erase(arch_s)
    # Fill pitch-black cavernous interior
    pen.blot(arch_s, jitter=0.22)

    # Visible receding 3D interior side wall & sloping floor wedge (clipped strictly inside arch_s!)
    # For 'south' (v_in = +y, goes up-right): LEFT (west, i=0..n//2) inner tunnel wall is visible receding up-right!
    # For 'east'  (v_in = -x, goes up-left): RIGHT/NORTH (i=n//2..n-1) inner tunnel wall is visible receding up-left!
    half = n // 2
    if facing == "south":
        wall_pts = arch_s[:half + 1] + inner_s[:half + 1][::-1]
        floor_pts = [arch_s[0], arch_s[-1], inner_s[-1], inner_s[0]]
    else:
        wall_pts = arch_s[half:] + inner_s[half:][::-1]
        floor_pts = [arch_s[0], arch_s[-1], inner_s[-1], inner_s[0]]

    # Only reveal a narrow dark-hatched outer ~24% jamb so the deep cavern stays brooding pitch black (even on east faces!)
    shallow_s = [P(cx + (wx - cx) * 0.92 + v_in[0] * (depth * 0.22),
                   cy + (wy - cy) * 0.92 + v_in[1] * (depth * 0.22),
                   z0 + (wz - z0) * 0.90) for wx, wy, wz in arch_w]
    if facing == "south":
        jamb_vis = arch_s[:half] + shallow_s[:half][::-1]
    else:
        jamb_vis = arch_s[half + 1:] + shallow_s[half + 1:][::-1]
    cv.erase(jamb_vis)
    pen.hatch(jamb_vis, 82, 1.25, W0 * 0.68, coverage=(0.82, 1.0))
    pen.hatch(jamb_vis, 24, 1.8, W0 * 0.46, coverage=(0.55, 0.90))
    # Subtle dark-shaded floor lip right at the threshold (never a bright white triangle!)
    fl_vis = [arch_s[0], arch_s[-1], shallow_s[-1], shallow_s[0]]
    pen.hatch(fl_vis, 16 if facing == "south" else -26, 1.35, W0 * 0.62, coverage=(0.75, 1.0))

    # Crisp 3D outer arch outline & threshold edge
    pen.stroke(arch_s, W0 * 1.04, taper=(0.06, 0.06), wobble=0.16)
    pen.stroke([arch_s[0], arch_s[-1]], W0 * 0.58, taper=(0.25, 0.25), wobble=0.15)

    # 3. Optional TRUE 3D architectural mine-timber frame or megalithic stone lintel!
    if timber:
        off_u = mw * 0.42
        ht = mh * 0.84
        for sgn in (-1, 1):
            p0 = P(cx + u[0] * sgn * off_u - v_in[0] * 1.5, cy + u[1] * sgn * off_u - v_in[1] * 1.5, z0)
            p1 = P(cx + u[0] * sgn * off_u - v_in[0] * 1.5, cy + u[1] * sgn * off_u - v_in[1] * 1.5, z0 + ht)
            pen.beam(p0, p1, 2.4)
        l0 = P(cx - u[0] * (off_u + 4.0) - v_in[0] * 2.0, cy - u[1] * (off_u + 4.0) - v_in[1] * 2.0, z0 + ht)
        l1 = P(cx + u[0] * (off_u + 4.0) - v_in[0] * 2.0, cy + u[1] * (off_u + 4.0) - v_in[1] * 2.0, z0 + ht)
        pen.beam(l0, l1, 2.8)
    elif stone_lintel:
        # True 3D stone jamb boxes & heavy overhanging stone lintel aligned with `facing`!
        off_u = mw * 0.44
        ht = mh * 0.84
        t_st = 4.2
        if facing == "south":
            box(pen, P, cx - off_u - 2.5, cy - t_st, cx - off_u + 2.5, cy + 1.5, z0, z0 + ht, w=W0 * 0.78)
            box(pen, P, cx + off_u - 2.5, cy - t_st, cx + off_u + 2.5, cy + 1.5, z0, z0 + ht, w=W0 * 0.78)
            box(pen, P, cx - off_u - 5.0, cy - t_st - 1.0, cx + off_u + 5.0, cy + 2.0, z0 + ht, z0 + ht + 4.8, w=W0 * 0.84)
        else:
            box(pen, P, cx - 1.5, cy + off_u - 2.5, cx + t_st, cy + off_u + 2.5, z0, z0 + ht, w=W0 * 0.78)
            box(pen, P, cx - 1.5, cy - off_u - 2.5, cx + t_st, cy - off_u + 2.5, z0, z0 + ht, w=W0 * 0.78)
            box(pen, P, cx - 2.0, cy - off_u - 5.0, cx + t_st + 1.0, cy + off_u + 5.0, z0 + ht, z0 + ht + 4.8, w=W0 * 0.84)

    if roots:
        for _ in range(rng.integers(2, 5)):
            i = int(rng.integers(3, n - 3))
            wx, wy, wz = arch_w[i]
            p_top = P(wx - v_in[0] * 1.5, wy - v_in[1] * 1.5, wz + rng.uniform(4.5, 9.0))
            p_lip = arch_s[i]
            p_dn = P(wx, wy, max(z0 + 3.0, wz - rng.uniform(3.5, mh * 0.36)))
            pen.stroke([p_top, p_lip], W0 * 0.52, taper=(0.55, 0.05), smooth=1, wobble=0.14)
            cv.erase([(p_lip[0] - 0.6, p_lip[1]), (p_lip[0] + 0.6, p_lip[1]), p_dn])
            pen.stroke([p_lip, p_dn], W0 * 0.38, taper=(0.08, 0.78), wobble=0.12)

    if trail:
        _trail_3d(pen, P, rng, cx - v_in[0] * 2.0, cy - v_in[1] * 2.0, -v_in, mw * 0.62, rng.uniform(24, 36))


def _boulder_3d(pen, P, rng, cx, cy, z0, rx, ry, h, tilt=(0.0, 0.0), flat_top=False, n=6):
    """Organic 3D weathered rock monolith projected through P(x, y, h) — NO cut-diamond wireframe grid!"""
    W0 = pen.width
    cam = _cam_vec(P)
    tx, ty = tilt
    ang0 = rng.uniform(0, TAU / n)
    angs = [ang0 + k * TAU / n + rng.normal(0, 0.18) for k in range(n)]
    r_jit = rng.uniform(0.80, 1.18, n)

    bot = [np.array([cx + rx * r_jit[k] * math.cos(a), cy + ry * r_jit[k] * math.sin(a), z0]) for k, a in enumerate(angs)]
    top_f = rng.uniform(0.68, 0.86) if flat_top else rng.uniform(0.45, 0.64)
    top_sx, top_sy = rng.normal(0, 0.16), rng.normal(0, 0.16)
    top = []
    for k, a in enumerate(angs):
        dx = rx * r_jit[k] * top_f * math.cos(a)
        dy = ry * r_jit[k] * top_f * math.sin(a)
        hz = z0 + h + dx * top_sx + dy * top_sy + rng.normal(0, h * 0.07)
        top.append(np.array([cx + tx + dx, cy + ty + dy, max(z0 + 3.0, hz)]))

    # Project all vertices and erase full boulder convex silhouette FIRST (clean unified rock mass!)
    all_s = [P(p[0], p[1], p[2]) for p in bot + top]
    from .solids import hull as _hull
    sil = _hull(all_s)
    pen.cv.erase(sil)

    faces = []
    for k in range(n):
        kn = (k + 1) % n
        q = [bot[k], bot[kn], top[kn], top[k]]
        e1, e2 = q[1] - q[0], q[3] - q[0]
        N = np.cross(e1, e2)
        cen = sum(q) / 4.0
        if float(N @ (cen - np.array([cx, cy, z0 + h * 0.4]))) < 0:
            N = -N
        nl = float(np.linalg.norm(N))
        if nl > 1e-6:
            N /= nl
            if float(N @ cam) > 0.04:
                faces.append((P.depth(cen[0], cen[1]) - cen[2] * 0.02, N, q, False))
    tcen = sum(top) / n
    N_top = np.array([-top_sx, -top_sy, 1.0])
    N_top /= float(np.linalg.norm(N_top))
    if float(N_top @ cam) > 0.05:
        faces.append((P.depth(tcen[0], tcen[1]) - tcen[2] * 0.15, N_top, top, True))

    # Shade visible rock faces organically & draw ONLY crisp natural fracture ridges (NOT a uniform wireframe!)
    for _, N, pts3, is_top in sorted(faces, key=lambda it: -it[0]):
        spts = [P(p[0], p[1], p[2]) for p in pts3]
        lit = float(N @ _SUN_3D)
        if not is_top:
            if lit < -0.10:
                pen.hatch(spts, 80, 1.75, W0 * 0.60, coverage=(0.68, 1.0))
            elif lit < 0.24:
                pen.hatch(spts, 74, 2.55, W0 * 0.42, coverage=(0.38, 0.78))
            # Draw vertical ridge edge cleanly where lighting turns (tapered, organic rock crease!)
            pen.stroke([spts[3], spts[0]], W0 * 0.62, taper=(0.12, 0.55), wobble=0.18)
        else:
            if lit < 0.35:
                pen.hatch(spts, 20, 2.8, W0 * 0.34, coverage=(0.25, 0.55))
            # Soft overhanging top brow rim (organic chiseled stone edge)
            pen.closed(spts, W0 * 0.78, overshoot=0.1, wobble=0.22)
            # Subtle fracture crack across top rock plane
            if rx > 12 and rng.random() < 0.65:
                c0 = P(tcen[0] - rx * 0.25, tcen[1] + rng.uniform(-ry * 0.2, ry * 0.2), tcen[2])
                c1 = P(tcen[0] + rx * 0.22, tcen[1] + rng.uniform(-ry * 0.2, ry * 0.2), tcen[2])
                pen.stroke([c0, c1], W0 * 0.42, taper=(0.35, 0.55), wobble=0.18)

    # Bold outer weathered stone silhouette
    pen.closed(sil, W0 * 1.02, overshoot=0.1, wobble=0.22)
    return tcen


def _burrow_pit_3d(pen, P, rng, cx, cy, z0, r_m, facing_ang, spoil=True, roots=True):
    """True 3/4 top-down oblique earth burrow pit dug DOWN into the ground/slope (ZERO train-tunnel semicircles!)."""
    W0, cv = pen.width, pen.cv
    # Outward exit direction along ground plane & perpendicular width axis
    d = np.array([math.cos(facing_ang), math.sin(facing_ang)])
    p = np.array([-d[1], d[0]])

    # 1. Freshly kicked-out crescent dirt spoil heap just outside & slightly beside the exit!
    if spoil:
        sp_c = np.array([cx, cy]) + d * (r_m * 1.15) + p * rng.uniform(-r_m * 0.45, r_m * 0.45)
        srx, sry, sh = r_m * rng.uniform(0.68, 0.95), r_m * rng.uniform(0.50, 0.72), r_m * rng.uniform(0.28, 0.42)
        ts_sp = np.linspace(0, TAU, 14, endpoint=False)
        sp_rim = [P(sp_c[0] + srx * math.cos(t), sp_c[1] + sry * math.sin(t), z0) for t in ts_sp]
        # Subtle low stippled earth mound (open & soft — NOT a white egg sticker!)
        top_sp = P(sp_c[0], sp_c[1], z0 + sh)
        for _ in range(int(r_m * 1.4)):
            rr = rng.uniform(0.0, 0.92)
            tt = rng.uniform(0, TAU)
            sx, sy = P(sp_c[0] + srx * rr * math.cos(tt), sp_c[1] + sry * rr * math.sin(tt), z0 + sh * (1 - rr * rr))
            cv.disc(sx, sy, rng.uniform(0.35, 0.62))
        # Delicate curved earth-heap brow stroke on the far/top side & 1-2 tiny pebbles
        brow_sp = [P(sp_c[0] + srx * math.cos(t), sp_c[1] + sry * math.sin(t), z0 + sh * max(0.0, math.sin(t - P.az)))
                   for t in np.linspace(P.az + 0.15, P.az + math.pi - 0.15, 7)]
        pen.stroke(chaikin(brow_sp, 1), W0 * 0.44, taper=(0.5, 0.5), wobble=0.18)
        pen.stone(top_sp[0] + rng.normal(0, 3.0), top_sp[1] + rng.uniform(1, 3.5), rng.uniform(1.1, 2.0))

    # 2. Sunken oblique ground-plane burrow opening (elongated along exit direction `d`, plunging under back lip!)
    # Outer hollow rim sits on the ground plane z0; back half rises gently into a low overhanging turf lip!
    n = 18
    ts = np.linspace(0, TAU, n, endpoint=False)
    pit_pts = []
    lip_pts = []
    h_lip = r_m * rng.uniform(0.42, 0.62)
    for t in ts:
        # cos(t) along -d (back/uphill), sin(t) along p (width)
        ca_t, sa_t = math.cos(t), math.sin(t)
        # Deeper along back/front axis so it reads clearly as a sloping burrow dug into the ground!
        dl = r_m * (0.85 if ca_t > 0 else 1.05) * (-ca_t)
        dw = r_m * 0.92 * sa_t
        wx, wy = cx + d[0] * dl + p[0] * dw, cy + d[1] * dl + p[1] * dw
        # Back lip (-d side, ca_t > 0) rises slightly above z0 to form an overhanging turf/earth hood
        dz = h_lip * (ca_t ** 1.25) if ca_t > 0 else 0.0
        pit_pts.append(P(wx, wy, z0 + dz))
        if ca_t > -0.15:
            # Slightly thicker outer crescent hood curving behind the lip
            ox = wx - d[0] * (2.6 * max(0.0, ca_t)) + p[0] * (1.2 * sa_t)
            oy = wy - d[1] * (2.6 * max(0.0, ca_t)) + p[1] * (1.2 * sa_t)
            lip_pts.append((t, P(ox, oy, z0 + dz + 1.2 * max(0.0, ca_t)), P(wx, wy, z0 + dz)))

    # Do NOT white-fill the open sloping dirt floor (keeps burrow holes grounded & earthy in filled style!)
    # Deep pitch-black tunnel throat tucked under the back/uphill ~68% of the pit!
    throat = []
    for t in ts:
        ca_t, sa_t = math.cos(t), math.sin(t)
        if ca_t >= -0.22:
            dl = r_m * 0.80 * (-ca_t)
            dw = r_m * 0.86 * sa_t
            dz = h_lip * 0.85 * (max(0.0, ca_t) ** 1.3)
        else:
            dl = r_m * 0.22
            dw = r_m * 0.78 * sa_t
            dz = 0.0
        throat.append(P(cx + d[0] * dl + p[0] * dw, cy + d[1] * dl + p[1] * dw, z0 + dz))

    # Front sloping bare-earth entry trough (subtle shaded cut-slope & dirt stipples leading down into darkness!)
    trough = [P(cx + d[0] * (r_m * (0.18 + 0.88 * max(0.0, -math.cos(t)))) + p[0] * (r_m * 0.88 * math.sin(t)),
                cy + d[1] * (r_m * (0.18 + 0.88 * max(0.0, -math.cos(t)))) + p[1] * (r_m * 0.88 * math.sin(t)), z0)
              for t in np.linspace(math.pi * 0.45, math.pi * 1.55, 10)]
    pen.hatch(pit_pts, 74, 1.85, W0 * 0.44, coverage=(0.48, 0.88))
    pen.blot(throat, jitter=0.22)
    # Earthy stipple dots scattered across the sloping bare-earth threshold
    for _ in range(int(r_m * 0.9)):
        f_d = rng.uniform(0.2, 1.05)
        f_w = rng.uniform(-0.75, 0.75)
        sx, sy = P(cx + d[0] * r_m * f_d + p[0] * r_m * f_w, cy + d[1] * r_m * f_d + p[1] * r_m * f_w, z0)
        cv.disc(sx, sy, rng.uniform(0.35, 0.58))

    # Crisp overhanging turf/earth crescent lip curving behind the black throat
    back_arc = [P(cx - d[0] * (r_m * 0.85 * math.cos(t)) + p[0] * (r_m * 0.94 * math.sin(t)),
                  cy - d[1] * (r_m * 0.85 * math.cos(t)) + p[1] * (r_m * 0.94 * math.sin(t)),
                  z0 + h_lip * (max(0.0, math.cos(t)) ** 1.2))
                for t in np.linspace(-math.pi * 0.56, math.pi * 0.56, 11)]
    outer_arc = [P(cx - d[0] * (r_m * (0.85 + 0.34 * max(0.0, math.cos(t)))) + p[0] * (r_m * 1.04 * math.sin(t)),
                   cy - d[1] * (r_m * (0.85 + 0.34 * max(0.0, math.cos(t)))) + p[1] * (r_m * 1.04 * math.sin(t)),
                   z0 + h_lip * 1.12 * (max(0.0, math.cos(t)) ** 1.15))
                 for t in np.linspace(-math.pi * 0.56, math.pi * 0.56, 11)]
    hood = outer_arc + back_arc[::-1]
    cv.erase(hood)
    pen.hatch(hood, 80, 1.85, W0 * 0.46, coverage=(0.55, 0.95))
    pen.stroke(chaikin(outer_arc, 1), W0 * 0.82, taper=(0.45, 0.45), wobble=0.16)
    pen.stroke(chaikin(back_arc, 1), W0 * 0.98, taper=(0.20, 0.20), wobble=0.14)
    # Soft broken front trough lip fading cleanly into the surrounding dirt/grass
    pen.stroke(chaikin(trough[1:-1], 1), W0 * 0.46, taper=(0.5, 0.5), wobble=0.16)

    if roots:
        for _ in range(rng.integers(2, 4)):
            t = rng.uniform(-0.85, 0.85)
            wx = cx - d[0] * (r_m * 0.82 * math.cos(t)) + p[0] * (r_m * 0.85 * math.sin(t))
            wy = cy - d[1] * (r_m * 0.82 * math.cos(t)) + p[1] * (r_m * 0.85 * math.sin(t))
            wz = z0 + h_lip * (max(0.0, math.cos(t)) ** 1.2)
            p_up = P(wx - d[0] * 3.5, wy - d[1] * 3.5, wz + 2.2)
            p_lip = P(wx, wy, wz)
            p_dn = P(wx + d[0] * 1.8, wy + d[1] * 1.8, max(z0, wz - 3.8))
            pen.stroke([p_up, p_lip], W0 * 0.48, taper=(0.55, 0.08), smooth=1)
            pen.stroke([p_lip, p_dn], W0 * 0.34, taper=(0.08, 0.78))
    # Tiny grass tuft hugging the back turf brow
    t_g = rng.uniform(-0.7, 0.7)
    gx, gy = P(cx - d[0] * (r_m * 1.15 * math.cos(t_g)) + p[0] * (r_m * 0.9 * math.sin(t_g)),
               cy - d[1] * (r_m * 1.15 * math.cos(t_g)) + p[1] * (r_m * 0.9 * math.sin(t_g)),
               z0 + h_lip * (max(0.0, math.cos(t_g)) ** 1.2))
    pen.tuft(gx, gy, rng.uniform(2.8, 4.4))


# --------------------------------------------------------------------------- 1. Hill Caves (3D 3/4)

def hill_cave(pen, P, rng):
    """True 3D 3/4 forested craggy hillside cave: irregular alpine massif, oblique south/east portal, trees & trail."""
    facing = "south" if rng.random() < 0.58 else "east"
    rx = rng.uniform(56, 72)
    ry = rng.uniform(44, 58)
    H = rng.uniform(52, 72)
    # Build a craggy asymmetric 2-peak 3D rocky ridge with steep rugged slopes (NO wide flat saucer rim!)
    pk1 = (rng.uniform(-rx * 0.18, rx * 0.08), rng.uniform(-ry * 0.06, ry * 0.16))
    side = int(rng.choice([-1, 1]))
    pk2 = (pk1[0] + side * rx * rng.uniform(0.32, 0.46), pk1[1] + rng.uniform(-ry * 0.18, ry * 0.22))
    H2 = H * rng.uniform(0.58, 0.80)
    s1x, s1y = rx * rng.uniform(0.58, 0.72), ry * rng.uniform(0.58, 0.72)
    s2x, s2y = rx * rng.uniform(0.38, 0.52), ry * rng.uniform(0.40, 0.54)
    ph1, ph2, ph3 = rng.uniform(0, TAU, 3)

    def h_fn(x, y):
        ang = math.atan2(y, x)
        # Irregular angular rocky mountain footprint (NOT a circular saucer!)
        rmod = 1.0 + 0.14 * math.sin(2 * ang + ph1) + 0.09 * math.cos(3 * ang + ph3)
        r2 = (x / (rx * rmod)) ** 2 + (y / (ry * rmod)) ** 2
        if r2 >= 1.0:
            return 0.0
        env = max(0.0, 1.0 - r2) ** 0.82
        g1 = H * math.exp(-((x - pk1[0]) / s1x) ** 2 - ((y - pk1[1]) / s1y) ** 2)
        g2 = H2 * math.exp(-((x - pk2[0]) / s2x) ** 2 - ((y - pk2[1]) / s2y) ** 2)
        crag = 1.0 + 0.09 * math.sin(x * 0.10 + ph1) * math.cos(y * 0.11 + ph2)
        return max(0.0, max(g1, g2) * env * crag * 1.15)

    # 1. Back woodland trees strictly behind the massif (depth > 0)
    tree_kind = str(rng.choice(["spruce", "spruce", "round"]))
    back_items = []
    for _ in range(rng.integers(1, 3)):
        t = rng.uniform(P.az + 0.15, P.az + math.pi - 0.15)
        wx, wy = rx * rng.uniform(0.85, 1.08) * math.cos(t), ry * rng.uniform(0.85, 1.08) * math.sin(t)
        back_items.append((P.depth(wx, wy), wx, wy, h_fn(wx, wy), rng.uniform(22, 34)))
    for _, wx, wy, wz, th in sorted(back_items, key=lambda it: -it[0]):
        _place_tree(pen, *P(wx, wy, wz), th, kind=tree_kind)

    # 2. True 3D rocky hill massif projected through P(x, y, h)!
    vis_fn = _hill_3d(pen, P, rng, h_fn, rx, ry, rocky=True, peaks=[pk1, pk2])

    # 3. Depth-sorted hillside trees, bushes, boulders & the 3D oblique cave portal!
    if facing == "south":
        mx, my = rng.uniform(-rx * 0.16, rx * 0.12), -ry * 0.68
    else:
        mx, my = rx * 0.56, rng.uniform(-ry * 0.28, -ry * 0.04)
    mw, mh = rng.uniform(24, 34), rng.uniform(20, 28)
    timber = rng.random() < 0.28
    roots = (not timber) and (rng.random() < 0.55)

    props = [(P.depth(mx, my), "cave", (mx, my))]
    for _ in range(rng.integers(2, 5)):
        t = rng.uniform(0, TAU)
        rr = rng.uniform(0.28, 0.86)
        wx, wy = rx * rr * math.cos(t), ry * rr * math.sin(t)
        wz = h_fn(wx, wy)
        if math.hypot(wx - mx, wy - my) < mw * 1.15 or not vis_fn(wx, wy, wz):
            continue
        kind = "tree" if rng.random() < 0.65 else "bush"
        props.append((P.depth(wx, wy), kind, (wx, wy, wz)))

    for sgn in (-1, 1):
        if facing == "south":
            bx, by = mx + sgn * (mw * 0.68 + rng.uniform(4, 12)), my - rng.uniform(4, 10)
        else:
            bx, by = mx + rng.uniform(4, 10), my + sgn * (mw * 0.68 + rng.uniform(4, 12))
        props.append((P.depth(bx, by), "boulder", (bx, by)))

    for _, kind, args in sorted(props, key=lambda it: -it[0]):
        if kind == "cave":
            _cave_portal_3d(pen, P, rng, args[0], args[1], 0.0, mw, mh, facing=facing,
                            depth=20.0, timber=timber, roots=roots, trail=True)
        elif kind == "tree":
            _place_tree(pen, *P(*args), rng.uniform(18, 30), kind=tree_kind if rng.random() < 0.8 else None)
        elif kind == "bush":
            pen.bush(*P(*args), rng.uniform(4.8, 7.6), outline=True)
        else:
            boulder(pen, *P(args[0], args[1], 0.0), rng.uniform(4.8, 8.2))

    # Ground grass tufts around the tilted 3/4 foothills
    for _ in range(rng.integers(4, 8)):
        t = rng.uniform(P.az + math.pi * 0.9, P.az + math.pi * 2.1)
        wx, wy = rx * rng.uniform(0.95, 1.22) * math.cos(t), ry * rng.uniform(0.95, 1.22) * math.sin(t)
        pen.tuft(*P(wx, wy, 0.0), rng.uniform(3.2, 5.2))


# --------------------------------------------------------------------------- 2. Cliff Caves (3D 3/4 Mesa-Headland)

def cliff_cave(pen, P, rng):
    """True 3D 3/4 stepped rocky headland bluff: tilted grassy plateau top, sheer wrapped cliff walls & oblique cavern."""
    W0 = pen.width
    facing = "south" if rng.random() < 0.56 else "east"
    W = rng.uniform(106, 138)
    D = rng.uniform(76, 102)
    H = rng.uniform(44, 62)

    # 1. Build an irregular jagged 3D rocky promontory polygon in world (x, y) — seen from 34° above & sideways!
    n = 18
    angs = np.linspace(0, TAU, n, endpoint=False)
    # Slight asymmetric jutting headland shape
    ph1, ph2 = rng.uniform(0, TAU, 2)
    ring_xy = []
    for a in angs:
        mod = 1.0 + 0.14 * math.sin(2 * a + ph1) + 0.09 * math.cos(3 * a + ph2) + rng.normal(0, 0.04)
        ring_xy.append((W * 0.5 * mod * math.cos(a), D * 0.5 * mod * math.sin(a)))

    # Optional secondary lower rocky shelf/spur tucked on the back-west or flanking side!
    if rng.random() < 0.72:
        sgn = int(rng.choice([-1, 1]))
        _boulder_3d(pen, P, rng, sgn * W * 0.44, rng.uniform(-D * 0.15, D * 0.25), 0.0,
                    W * 0.26, D * 0.28, H * rng.uniform(0.45, 0.68), flat_top=True)

    # Compute exact visible front cliff wall segments (where outward 2D edge normal faces viewer!)
    cam2 = np.array([P.sa, -P.ca])
    bot_s = [P(x * 1.08, y * 1.08, 0.0) for x, y in ring_xy]
    top_s = [P(x, y, H + rng.normal(0, 1.4)) for x, y in ring_xy]

    # Build cliff wall quads & sort back-to-front by edge midpoint depth!
    walls = []
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = ring_xy[i]
        x1, y1 = ring_xy[j]
        ex, ey = x1 - x0, y1 - y0
        # CCW polygon -> outward normal is (ey, -ex)
        nx, ny = ey, -ex
        nl = max(1e-6, math.hypot(nx, ny))
        nx, ny = nx / nl, ny / nl
        if nx * cam2[0] + ny * cam2[1] > -0.02:
            mx_e, my_e = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            walls.append((P.depth(mx_e, my_e), nx, ny, i, j))

    # Erase full headland silhouette (top plateau + all visible cliff faces) cleanly
    for _, _, _, i, j in sorted(walls, key=lambda it: -it[0]):
        quad = [bot_s[i], bot_s[j], top_s[j], top_s[i]]
        pen.cv.erase(quad)
    pen.cv.erase(top_s)

    # 2. Shade sheer 3D cliff faces with directional east shadow hatching + clustered vertical crag buttresses!
    for _, nx, ny, i, j in sorted(walls, key=lambda it: -it[0]):
        quad = [bot_s[i], bot_s[j], top_s[j], top_s[i]]
        eastness = nx * 0.92 - ny * 0.28
        if eastness > 0.35:
            pen.hatch(quad, 84, 2.35, W0 * 0.54, coverage=(0.62, 1.0))
        elif eastness > 0.08:
            pen.hatch(quad, 82, 3.2, W0 * 0.38, coverage=(0.35, 0.72))
        # Bold vertical crag buttress rib descending from each cliff rim corner + tucked shadow accent!
        drop = rng.uniform(0.68, 0.96)
        b_pt = (top_s[i][0] + (bot_s[i][0] - top_s[i][0]) * drop, top_s[i][1] + (bot_s[i][1] - top_s[i][1]) * drop)
        pen.stroke([top_s[i], b_pt], W0 * (0.88 if eastness > 0.2 else 0.74), taper=(0.05, 0.78), wobble=0.18)
        if rng.random() < 0.70:
            xm = (top_s[i][0] + top_s[j][0]) * 0.5
            ym = (top_s[i][1] + top_s[j][1]) * 0.5
            ym_b = (bot_s[i][1] + bot_s[j][1]) * 0.5
            pen.stroke([(xm, ym + 1.5), (xm + rng.normal(0, 1.2), ym + (ym_b - ym) * rng.uniform(0.35, 0.75))],
                       W0 * 0.50, taper=(0.08, 0.85), wobble=0.14)
        if rng.random() < 0.45:
            f_h = rng.uniform(0.32, 0.68)
            p_a = (top_s[i][0] + (bot_s[i][0] - top_s[i][0]) * f_h, top_s[i][1] + (bot_s[i][1] - top_s[i][1]) * f_h)
            p_b = (top_s[j][0] + (bot_s[j][0] - top_s[j][0]) * f_h, top_s[j][1] + (bot_s[j][1] - top_s[j][1]) * f_h)
            pen.stroke([p_a, p_b], W0 * 0.44, taper=(0.25, 0.45), wobble=0.16)
        if rng.random() < 0.78:
            pen.stroke([bot_s[i], bot_s[j]], W0 * 0.64, taper=(0.35, 0.35), wobble=0.18)

    # 3. Clean sunlit 3/4 grassy plateau top (seen clearly from 34° above!) + optional upper rocky shelf + pines!
    pen.closed(top_s, W0 * 1.12, wobble=0.22)
    if rng.random() < 0.48:
        # Subtle stepped secondary rock outcrop on the back-west plateau corner for richer headland geology
        _boulder_3d(pen, P, rng, rng.uniform(-W * 0.16, W * 0.12), rng.uniform(D * 0.05, D * 0.18), H - 1.0,
                    W * 0.18, D * 0.16, rng.uniform(8, 14), flat_top=True, n=5)
    top_items = []
    for _ in range(rng.integers(2, 4)):
        rr = rng.uniform(0.15, 0.62)
        t = rng.uniform(0, TAU)
        tx, ty = W * 0.42 * rr * math.cos(t), D * 0.42 * rr * math.sin(t)
        top_items.append((P.depth(tx, ty), "tree" if rng.random() < 0.68 else "bush", tx, ty))
    for _ in range(rng.integers(4, 8)):
        rr = rng.uniform(0.1, 0.72)
        t = rng.uniform(0, TAU)
        tx, ty = W * 0.42 * rr * math.cos(t), D * 0.42 * rr * math.sin(t)
        top_items.append((P.depth(tx, ty), "tuft", tx, ty))
    for _, kind, tx, ty in sorted(top_items, key=lambda it: -it[0]):
        if kind == "tree":
            flora.spruce(pen, *P(tx, ty, H), rng.uniform(18, 30))
        elif kind == "bush":
            pen.bush(*P(tx, ty, H), rng.uniform(5.0, 8.0), outline=True)
        else:
            pen.tuft(*P(tx, ty, H), rng.uniform(3.0, 4.8))

    # 4. Carve the 3D oblique cliff cavern right into the south OR east sheer cliff face!
    if facing == "south":
        c_pts = [(x, y) for x, y in ring_xy if y < -D * 0.25 and abs(x) < W * 0.28]
        mx, my = c_pts[len(c_pts) // 2] if c_pts else (0.0, -D * 0.46)
        mx *= 1.04
        my *= 1.04
    else:
        c_pts = [(x, y) for x, y in ring_xy if x > W * 0.25 and abs(y) < D * 0.28]
        mx, my = c_pts[len(c_pts) // 2] if c_pts else (W * 0.46, 0.0)
        mx *= 1.04
        my *= 1.04
    mw, mh = rng.uniform(24, 34), min(H * 0.66, rng.uniform(22, 32))
    _cave_portal_3d(pen, P, rng, mx, my, 0.0, mw, mh, facing=facing, depth=22.0,
                    timber=(rng.random() < 0.22), stone_lintel=(rng.random() < 0.25),
                    roots=(rng.random() < 0.30), trail=True)

    # Organic weathered talus boulders flanking the cliff foot
    for sgn in (-1, 1):
        if facing == "south":
            bx, by = mx + sgn * (mw * 0.72 + rng.uniform(6, 16)), my - rng.uniform(4, 10)
        else:
            bx, by = mx + rng.uniform(4, 10), my + sgn * (mw * 0.72 + rng.uniform(6, 16))
        boulder(pen, *P(bx, by, 0.0), rng.uniform(5.0, 8.5))


# --------------------------------------------------------------------------- 3. Boulder Caves (3D 3/4 Megalithic Dolmen)

def boulder_cave(pen, P, rng):
    """True 3D tumbled megalithic boulder grotto in P(x, y, h): organic 3D monoliths + tilted capstone slab + woodland flora."""
    facing = "south" if rng.random() < 0.56 else "east"
    gap = rng.uniform(24, 32)
    h_side = rng.uniform(26, 36)
    if facing == "south":
        u, v_in = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    else:
        u, v_in = np.array([0.0, 1.0]), np.array([-1.0, 0.0])

    # 1. Back woodland trees & rear anchor boulder
    back_b = v_in * rng.uniform(20, 30)
    items_bg = [(P.depth(back_b[0], back_b[1]), "rock", (back_b[0], back_b[1], rng.uniform(22, 32), rng.uniform(18, 28), h_side * rng.uniform(0.95, 1.20)))]
    for sgn in (-1, 1):
        if rng.random() < 0.72:
            tp = u * (sgn * (gap * 0.9 + rng.uniform(22, 36))) + v_in * rng.uniform(6, 22)
            items_bg.append((P.depth(tp[0], tp[1]), "tree", (tp[0], tp[1], rng.uniform(26, 38))))
    for _, kind, args in sorted(items_bg, key=lambda it: -it[0]):
        if kind == "rock":
            _boulder_3d(pen, P, rng, args[0], args[1], 0.0, args[2], args[3], args[4], flat_top=False, n=6)
        else:
            _place_tree(pen, *P(args[0], args[1], 0.0), args[2], kind=str(rng.choice(["round", "spruce"])))

    # 2. Two heavy organic 3D flanking monoliths — placed cleanly apart so the dark grotto ALWAYS reads boldly!
    b_l = -u * (gap * 0.5 + rng.uniform(14, 19))
    b_r = u * (gap * 0.5 + rng.uniform(14, 19))
    flanks = [
        (P.depth(b_l[0], b_l[1]), b_l, u * 3.0, rng.uniform(14, 20), rng.uniform(14, 20), h_side * rng.uniform(0.92, 1.08)),
        (P.depth(b_r[0], b_r[1]), b_r, -u * 3.0, rng.uniform(14, 20), rng.uniform(14, 20), h_side * rng.uniform(0.92, 1.08)),
    ]
    # Draw farther monolith -> dark 3D grotto hollow -> closer monolith -> overhanging capstone slab!
    flanks.sort(key=lambda it: -it[0])
    _, pos0, lean0, rx0, ry0, h0 = flanks[0]
    _boulder_3d(pen, P, rng, pos0[0], pos0[1], 0.0, rx0, ry0, h0, tilt=(lean0[0], lean0[1]), flat_top=True, n=6)

    mouth_c = -v_in * 4.0
    _cave_portal_3d(pen, P, rng, mouth_c[0], mouth_c[1], 0.0, gap * 1.12, h_side * 0.90, facing=facing,
                    depth=24.0, roots=(rng.random() < 0.45), trail=True, brow=False)

    _, pos1, lean1, rx1, ry1, h1 = flanks[1]
    _boulder_3d(pen, P, rng, pos1[0], pos1[1], 0.0, rx1, ry1, h1, tilt=(lean1[0], lean1[1]), flat_top=True, n=6)

    # 3. Massive tilted 3D capstone boulder slab spanning across BOTH monoliths!
    cap_rx = (gap * 0.88 + 22.0) if facing == "south" else rng.uniform(20, 28)
    cap_ry = rng.uniform(20, 28) if facing == "south" else (gap * 0.88 + 22.0)
    cap_c = v_in * 2.0 + u * rng.normal(0, 3.0)
    top_c = _boulder_3d(pen, P, rng, cap_c[0], cap_c[1], h_side * 0.80, cap_rx, cap_ry,
                        rng.uniform(13, 19), tilt=(rng.normal(0, 3), rng.normal(0, 3)), flat_top=True, n=6)
    if rng.random() < 0.65:
        off_t = u * rng.choice([-1, 1]) * rng.uniform(6, 14)
        pen.tuft(*P(top_c[0] + off_t[0], top_c[1] + off_t[1], top_c[2]), rng.uniform(3.2, 5.0))

    # 4. Foreground organic scree boulders, scalloped bushes & tufts in 3D `(x, y)`
    fg = []
    for sgn in (-1, 1):
        p_f = u * (sgn * (gap * 0.72 + rng.uniform(14, 24))) - v_in * rng.uniform(10, 18)
        fg.append((P.depth(p_f[0], p_f[1]), "bush" if rng.random() < 0.55 else "rock", p_f))
    for _ in range(rng.integers(4, 7)):
        p_t = u * rng.uniform(-58, 58) - v_in * rng.uniform(12, 30)
        fg.append((P.depth(p_t[0], p_t[1]), "tuft", p_t))
    for _, kind, pos in sorted(fg, key=lambda it: -it[0]):
        if kind == "bush":
            pen.bush(*P(pos[0], pos[1], 0.0), rng.uniform(5.2, 8.2), outline=True)
        elif kind == "rock":
            boulder(pen, *P(pos[0], pos[1], 0.0), rng.uniform(4.8, 8.2))
        else:
            pen.tuft(*P(pos[0], pos[1], 0.0), rng.uniform(3.2, 5.2))


# --------------------------------------------------------------------------- 4. Sunken Entrances (3D 3/4 Barrow Stairway)

def low_entrance(pen, P, rng):
    """True 3D 3/4 sunken barrow / stone-stairway descent cutting obliquely down below ground level!"""
    W0, cv = pen.width, pen.cv
    facing = "south" if rng.random() < 0.58 else "east"
    # 1. Low 3D rounded grassy barrow mound behind the sunken cut (viewed from 34° above!)
    rx, ry, H = rng.uniform(58, 76), rng.uniform(42, 56), rng.uniform(22, 32)
    off_m = (0.0, 16.0) if facing == "south" else (-16.0, 0.0)

    def h_fn(x, y):
        r2 = ((x - off_m[0]) / rx) ** 2 + ((y - off_m[1]) / ry) ** 2
        if r2 >= 1.0:
            return 0.0
        return H * ((1.0 - r2) ** 1.35)

    # Rear tree/shrubs anchored on the 3D barrow mound
    vis_fn = _hill_3d(pen, P, rng, h_fn, rx, ry, cx=off_m[0], cy=off_m[1], rocky=False, shade_scale=0.92)
    if rng.random() < 0.65:
        tx = off_m[0] + rng.choice([-1, 1]) * rx * rng.uniform(0.28, 0.52)
        ty = off_m[1] + ry * rng.uniform(0.10, 0.42)
        _place_tree(pen, *P(tx, ty, h_fn(tx, ty)), rng.uniform(22, 32))
    for _ in range(rng.integers(1, 3)):
        bx = off_m[0] + rng.uniform(-rx * 0.45, rx * 0.45)
        by = off_m[1] + rng.uniform(ry * 0.05, ry * 0.48)
        pen.bush(*P(bx, by, h_fn(bx, by)), rng.uniform(5.0, 7.8), outline=True)

    # 2. True 3D sunken trench / stone-walled stairway cutting DOWN from z=0 to z=-depth_z!
    tw = rng.uniform(22, 28)   # trench width
    tL = rng.uniform(34, 46)   # trench length emerging along -y (south) or +x (east)
    dz = rng.uniform(13, 18)   # sunken depth below ground!
    n_st = int(rng.integers(5, 7))
    has_steps = rng.random() < 0.72

    if facing == "south":
        # Trench runs from y = y_m (deep vault) southward to y = y_m - tL (surface level z=0)
        y_m = off_m[1] - ry * 0.56
        x0, x1 = -tw * 0.5, tw * 0.5
        # Visible sunken inner WEST cut-wall (at x=x0, faces east -> deeply shadowed!)
        # Top edge stays at z=0 (ground), bottom floor slopes from z=-dz at y_m up to z=0 at y_m - tL!
        w_west = [P(x0, y_m, 0.0), P(x0, y_m - tL, 0.0), P(x0, y_m, -dz)]
        # Dark vault mouth tucked at the bottom of the sunken cut (y = y_m, z = -dz .. +6)
        _cave_portal_3d(pen, P, rng, 0.0, y_m, -dz * 0.78, tw * 0.96, dz * 0.92 + 7.5,
                        facing="south", depth=18.0, stone_lintel=(rng.random() < 0.65),
                        timber=(rng.random() < 0.25), roots=(rng.random() < 0.45), trail=False)
        # Draw sunken shadowed west embankment wall & stone steps descending down into the vault!
        cv.erase(w_west)
        pen.hatch(w_west, 84, 1.55, W0 * 0.62, coverage=(0.75, 1.0))
        pen.stroke([P(x0, y_m, 0.0), P(x0, y_m - tL, 0.0)], W0 * 1.05, taper=(0.06, 0.75), wobble=0.18)
        pen.stroke([P(x0, y_m, -dz), P(x0, y_m - tL, 0.0)], W0 * 0.62, taper=(0.15, 0.55), wobble=0.14)
        # Descending 3D stone stairs (drawn deepest -> shallowest so front steps cleanly step up!)
        if has_steps:
            for k in range(n_st):
                f0, f1 = k / n_st, (k + 1) / n_st
                ya, yb = y_m - tL * f0, y_m - tL * f1
                za, zb = -dz * (1 - f0), -dz * (1 - f1)
                tread = [P(x0 + 0.6, ya, za), P(x1 - 0.6, ya, za), P(x1 - 0.6, yb, za), P(x0 + 0.6, yb, za)]
                riser = [P(x0 + 0.6, yb, za), P(x1 - 0.6, yb, za), P(x1 - 0.6, yb, zb), P(x0 + 0.6, yb, zb)]
                cv.erase(tread)
                pen.hatch(tread, 16, 2.5, W0 * 0.34, coverage=(0.25, 0.55))
                pen.stroke([tread[0], tread[1]], W0 * 0.62, taper=(0.12, 0.12), wobble=0.12)
                cv.erase(riser)
                pen.stroke([riser[2], riser[3]], W0 * 0.66, taper=(0.12, 0.12), wobble=0.12)
        else:
            _trail_3d(pen, P, rng, 0.0, y_m - 4.0, (0.0, -1.0), tw * 0.55, tL * 1.15,
                      z_fn=lambda x, y: -dz * max(0.0, min(1.0, (y - (y_m - tL)) / tL)))
        # Right (east) surface rim lip curving out along x = x1
        pen.stroke([P(x1, y_m, 0.0), P(x1, y_m - tL, 0.0)], W0 * 1.02, taper=(0.06, 0.78), wobble=0.18)
        # Mossy stones & tufts hugging the sunken rims
        for sgn in (-1, 1):
            for f in (0.28, 0.72):
                sx, sy = (x0 if sgn < 0 else x1) + sgn * rng.uniform(3.5, 7.5), y_m - tL * f
                if rng.random() < 0.55:
                    boulder(pen, *P(sx, sy, 0.0), rng.uniform(3.8, 6.2))
                else:
                    pen.tuft(*P(sx, sy, 0.0), rng.uniform(3.2, 5.0))
        if has_steps:
            _trail_3d(pen, P, rng, 0.0, y_m - tL - 1.0, (0.0, -1.0), tw * 0.58, rng.uniform(14, 22))
    else:
        # East-facing sunken descent running from x = x_m eastward to x = x_m + tL!
        x_m = off_m[0] + rx * 0.56
        y0, y1 = -tw * 0.5, tw * 0.5
        # Visible sunken inner NORTH cut-wall (at y = y1, faces south -> lightly hatched!)
        w_north = [P(x_m, y1, 0.0), P(x_m + tL, y1, 0.0), P(x_m, y1, -dz)]
        _cave_portal_3d(pen, P, rng, x_m, 0.0, -dz * 0.78, tw * 0.96, dz * 0.92 + 7.5,
                        facing="east", depth=18.0, stone_lintel=(rng.random() < 0.65),
                        timber=(rng.random() < 0.25), roots=(rng.random() < 0.45), trail=False)
        cv.erase(w_north)
        pen.hatch(w_north, 82, 2.2, W0 * 0.48, coverage=(0.55, 0.92))
        pen.stroke([P(x_m, y1, 0.0), P(x_m + tL, y1, 0.0)], W0 * 1.04, taper=(0.06, 0.75), wobble=0.18)
        pen.stroke([P(x_m, y1, -dz), P(x_m + tL, y1, 0.0)], W0 * 0.58, taper=(0.15, 0.55), wobble=0.14)
        if has_steps:
            for k in range(n_st):
                f0, f1 = k / n_st, (k + 1) / n_st
                xa, xb = x_m + tL * f0, x_m + tL * f1
                za, zb = -dz * (1 - f0), -dz * (1 - f1)
                tread = [P(xa, y0 + 0.6, za), P(xa, y1 - 0.6, za), P(xb, y1 - 0.6, za), P(xb, y0 + 0.6, za)]
                cv.erase(tread)
                pen.hatch(tread, -22, 2.4, W0 * 0.36, coverage=(0.25, 0.55))
                pen.stroke([tread[0], tread[1]], W0 * 0.64, taper=(0.12, 0.12), wobble=0.12)
                pen.stroke([P(xb, y0 + 0.6, zb), P(xb, y1 - 0.6, zb)], W0 * 0.64, taper=(0.12, 0.12), wobble=0.12)
        else:
            _trail_3d(pen, P, rng, x_m + 4.0, 0.0, (1.0, 0.0), tw * 0.55, tL * 1.15,
                      z_fn=lambda x, y: -dz * max(0.0, min(1.0, ((x_m + tL) - x) / tL)))
        pen.stroke([P(x_m, y0, 0.0), P(x_m + tL, y0, 0.0)], W0 * 1.04, taper=(0.06, 0.78), wobble=0.18)
        for sgn in (-1, 1):
            for f in (0.28, 0.72):
                sx, sy = x_m + tL * f, (y0 if sgn < 0 else y1) + sgn * rng.uniform(3.5, 7.5)
                if rng.random() < 0.55:
                    boulder(pen, *P(sx, sy, 0.0), rng.uniform(3.8, 6.2))
                else:
                    pen.tuft(*P(sx, sy, 0.0), rng.uniform(3.2, 5.0))
        if has_steps:
            _trail_3d(pen, P, rng, x_m + tL + 1.0, 0.0, (1.0, 0.0), tw * 0.58, rng.uniform(14, 22))


CAVE_KINDS = [(hill_cave, 0.35), (cliff_cave, 0.25), (boulder_cave, 0.25), (low_entrance, 0.15)]


def cave_lair(pen, P, rng):
    fns, wts = zip(*CAVE_KINDS)
    fns[int(rng.choice(len(fns), p=np.array(wts) / sum(wts)))](pen, P, rng)


# --------------------------------------------------------------------------- 5 & 6. Burrows & Warrens (3D 3/4 Aerial View!)

def _gnarled_oak_3d(pen, P, rng, wx, wy, wz, h, exit_ang):
    """3D woodland oak perched atop a burrow bank with gnarled surface roots radiating across the 3/4 ground plane."""
    W0 = pen.width
    flora.round_tree(pen, *P(wx, wy, wz), h)
    # 3-4 twisting roots snaking outward across the 3D ground plane & arching around the burrow!
    for off in (-1.05, -0.45, 0.45, 1.05):
        ang = exit_ang + off + rng.normal(0, 0.15)
        L = rng.uniform(12, 20)
        p0 = P(wx, wy, wz + 2.0)
        p1 = P(wx + math.cos(ang) * L * 0.45, wy + math.sin(ang) * L * 0.45, max(0.0, wz * 0.55))
        p2 = P(wx + math.cos(ang) * L, wy + math.sin(ang) * L, 0.0)
        pen.stroke([p0, p1, p2], W0 * (0.78 if abs(off) < 0.7 else 0.56), taper=(0.05, 0.88), smooth=1, wobble=0.14)


def burrow_lair(pen, P, rng, warren=None):
    """True 3D 3/4 top-down Burrow (root-hollow / earth-bank pit) OR sprawling honeycombed Warren colony (ZERO train tunnels!)."""
    if warren is None:
        warren = rng.random() >= 0.6

    if not warren:
        # SINGLE BURROW: gentle 3D grassy knoll seen from 34° above with an oblique dug earth-pit under roots/turf!
        rx, ry, H = rng.uniform(44, 58), rng.uniform(34, 46), rng.uniform(16, 25)
        cx_w, cy_w = 0.0, 8.0

        def h_fn(x, y):
            r2 = ((x - cx_w) / rx) ** 2 + ((y - cy_w) / ry) ** 2
            if r2 >= 1.0:
                return 0.0
            return H * ((1.0 - r2) ** 1.35)

        _hill_3d(pen, P, rng, h_fn, rx, ry, cx=cx_w, cy=cy_w, rocky=False, shade_scale=0.92)

        # Pick an oblique 3/4 exit angle facing either south-west (-y) OR south-east (+x)!
        exit_ang = rng.uniform(-math.pi * 0.68, -math.pi * 0.35) if rng.random() < 0.58 else rng.uniform(-0.28, 0.22)
        bx = cx_w + math.cos(exit_ang) * rx * 0.52
        by = cy_w + math.sin(exit_ang) * ry * 0.52
        bz = h_fn(bx, by)
        rm = rng.uniform(9.5, 13.5)

        # Place woodland tree(s), scalloped bushes & optional mossy stone cleanly off to the knoll shoulders
        # (strictly clear of the burrow lip & each other — NEVER a lollipop trunk centered atop the hole!)
        props = [(P.depth(bx, by), "pit", (bx, by, bz))]
        pit_sx, _ = P(bx, by, bz)
        trees_xy = []
        bushes_xy = []

        if rng.random() < 0.78:
            side = int(rng.choice([-1, 1]))
            n_trees = 2 if rng.random() < 0.25 else 1
            for k in range(n_trees):
                sgn = side if k == 0 else -side
                for _ in range(30):
                    # Upper/rear left or right shoulder of the 3D grassy knoll
                    ang_t = rng.uniform(0.32, 1.35) if sgn > 0 else rng.uniform(1.75, 2.82)
                    rr = rng.uniform(0.42, 0.84)
                    tx, ty = cx_w + rx * rr * math.cos(ang_t), cy_w + ry * rr * math.sin(ang_t)
                    tz = h_fn(tx, ty)
                    tsx, _ = P(tx, ty, tz)
                    if (math.hypot(tx - bx, ty - by) > rm * 2.15
                            and abs(tsx - pit_sx) > rm * 1.25
                            and all(math.hypot(tx - ox, ty - oy) > 20.0 for ox, oy in trees_xy)):
                        trees_xy.append((tx, ty))
                        props.append((P.depth(tx, ty), "tree", (tx, ty, tz, rng.uniform(22, 34))))
                        break

        for _ in range(int(rng.integers(2, 4))):
            for _ in range(25):
                ang_b = rng.uniform(0.0, TAU)
                rr = rng.uniform(0.35, 0.92)
                px, py = cx_w + rx * rr * math.cos(ang_b), cy_w + ry * rr * math.sin(ang_b)
                if (math.hypot(px - bx, py - by) > rm * 1.75
                        and all(math.hypot(px - tx, py - ty) > 16.0 for tx, ty in trees_xy)
                        and all(math.hypot(px - ox, py - oy) > 14.0 for ox, oy in bushes_xy)):
                    bushes_xy.append((px, py))
                    props.append((P.depth(px, py), "bush", (px, py, h_fn(px, py))))
                    break

        if rng.random() < 0.35:
            sgn = int(rng.choice([-1, 1]))
            ang_r = exit_ang + sgn * rng.uniform(0.85, 1.35)
            rx_p, ry_p = bx + math.cos(ang_r) * rm * 1.65, by + math.sin(ang_r) * rm * 1.65
            if all(math.hypot(rx_p - tx, ry_p - ty) > 14.0 for tx, ty in trees_xy):
                props.append((P.depth(rx_p, ry_p), "stone_b", (rx_p, ry_p, h_fn(rx_p, ry_p))))

        for _ in range(rng.integers(6, 10)):
            t = rng.uniform(0, TAU)
            rr = rng.uniform(0.25, 1.18)
            gx, gy = cx_w + rx * rr * math.cos(t), cy_w + ry * rr * math.sin(t)
            if math.hypot(gx - bx, gy - by) > rm * 1.45:
                props.append((P.depth(gx, gy), "tuft", (gx, gy, h_fn(gx, gy))))

        for _, kind, args in sorted(props, key=lambda it: -it[0]):
            if kind == "pit":
                _burrow_pit_3d(pen, P, rng, args[0], args[1], args[2], rm, exit_ang, spoil=True, roots=True)
                _trail_3d(pen, P, rng, args[0] + math.cos(exit_ang) * rm * 0.9,
                          args[1] + math.sin(exit_ang) * rm * 0.9,
                          (math.cos(exit_ang), math.sin(exit_ang)), rm * 0.85, rng.uniform(22, 32), z_fn=h_fn)
            elif kind == "tree":
                _place_tree(pen, *P(args[0], args[1], args[2]), args[3])
            elif kind == "bush":
                pen.bush(*P(*args), rng.uniform(5.0, 7.8), outline=True)
            elif kind == "stone_b":
                boulder(pen, *P(*args), rng.uniform(4.2, 6.8))
            else:
                pen.tuft(*P(*args), rng.uniform(3.0, 4.8))
        return

    # WARREN: True 3D 3/4 TOP-DOWN sprawling honeycombed burrow colony!
    # Broad low rolling grassy knoll + 4-6 oblique ground-dug burrow pits scattered across the 3D hillside & meadow!
    rx, ry, H = rng.uniform(68, 86), rng.uniform(50, 64), rng.uniform(20, 30)
    cx_w, cy_w = 0.0, 12.0
    lobe_x = rng.choice([-1, 1]) * rx * rng.uniform(0.25, 0.36)
    lobe_y = cy_w + rng.uniform(-ry * 0.12, ry * 0.18)

    def h_fn(x, y):
        r1 = ((x - cx_w) / rx) ** 2 + ((y - cy_w) / ry) ** 2
        r2 = ((x - lobe_x) / (rx * 0.62)) ** 2 + ((y - lobe_y) / (ry * 0.62)) ** 2
        h1 = H * (max(0.0, 1.0 - r1) ** 1.35) if r1 < 1.0 else 0.0
        h2 = H * 0.72 * (max(0.0, 1.0 - r2) ** 1.35) if r2 < 1.0 else 0.0
        return max(h1, h2)

    _hill_3d(pen, P, rng, h_fn, rx * 1.05, ry * 1.05, cx=cx_w, cy=cy_w, rocky=False, shade_scale=0.95)

    # Scatter 4-6 oblique 3D burrow holes across the visible south/east slopes AND surrounding front meadow!
    holes = []  # (x, y, z, r_m, exit_ang)
    n_target = int(rng.integers(4, 7))
    # Ensure a mix of hillside holes (on the south/east flanks) and sunken meadow apron holes!
    for attempt in range(140):
        if len(holes) >= n_target:
            break
        if len(holes) < 2:
            # Upper/mid hillside holes facing radially downhill (south-west to east!)
            ang = rng.uniform(-math.pi * 0.78, 0.22)
            rr = rng.uniform(0.32, 0.66)
        else:
            # Lower flank & meadow holes scattered naturally across the 3/4 clearing!
            ang = rng.uniform(-math.pi * 0.85, 0.30)
            rr = rng.uniform(0.65, 1.15)
        hx = cx_w + rx * rr * math.cos(ang)
        hy = cy_w + ry * rr * math.sin(ang)
        rm = rng.uniform(8.8, 12.2) if len(holes) == 0 else rng.uniform(6.8, 10.2)
        if all(math.hypot(hx - ox, hy - oy) > (rm + orm) * 2.25 for ox, oy, _, orm, _ in holes):
            # Face radially outward/downhill with slight organic variation
            ex_ang = math.atan2(hy - cy_w, hx - cx_w) + rng.normal(0, 0.25)
            # Clamp exit angle so holes always face generally toward the viewer (south/east hemispheres)
            ex_ang = float(np.clip(ex_ang, -math.pi * 0.85, 0.35))
            holes.append((hx, hy, h_fn(hx, hy), rm, ex_ang))

    # Subtle broken stippled animal-track wisps meandering between nearby warren holes (never a single rigid wire!)
    for i in range(len(holes)):
        hx, hy, hz, rm, ex_ang = holes[i]
        others = [j for j in range(len(holes)) if j != i]
        if others:
            j = min(others, key=lambda k: math.hypot(holes[k][0] - hx, holes[k][1] - hy))
            if i < j:
                nx, ny, _, _, _ = holes[j]
                dist = math.hypot(nx - hx, ny - hy)
                if dist < 68:
                    mx_p = (hx + nx) * 0.5 + rng.normal(0, 7.0)
                    my_p = (hy + ny) * 0.5 + rng.normal(0, 6.0)
                    for t in np.linspace(0.20, 0.80, int(dist / 5.5)):
                        qx = (1 - t) ** 2 * hx + 2 * (1 - t) * t * mx_p + t ** 2 * nx
                        qy = (1 - t) ** 2 * hy + 2 * (1 - t) * t * my_p + t ** 2 * ny
                        sx, sy = P(qx, qy, h_fn(qx, qy))
                        pen.cv.disc(sx + rng.normal(0, 1.6), sy + rng.normal(0, 1.0), rng.uniform(0.45, 0.72))

    # Depth-sort ALL 3D warren elements (trees, bushes, burrow pits, grass tufts) strictly back-to-front!
    props = [(P.depth(hx, hy), "pit", (hx, hy, hz, rm, ex_ang, idx == 0))
             for idx, (hx, hy, hz, rm, ex_ang) in enumerate(holes)]

    # 1-2 woodland trees anchoring the rear/flanking knoll ridges (well-spaced from holes & each other)
    w_trees = []
    for _ in range(int(rng.integers(1, 3))):
        for _ in range(20):
            ang_t = rng.uniform(0.35, math.pi * 0.95)
            rr = rng.uniform(0.38, 0.85)
            tx, ty = cx_w + rx * rr * math.cos(ang_t), cy_w + ry * rr * math.sin(ang_t)
            if (all(math.hypot(tx - hx, ty - hy) > 22 for hx, hy, _, _, _ in holes)
                    and all(math.hypot(tx - ox, ty - oy) > 18 for ox, oy in w_trees)):
                w_trees.append((tx, ty))
                props.append((P.depth(tx, ty), "tree", (tx, ty, h_fn(tx, ty))))
                break

    # Scalloped bramble bushes tucked naturally around the warren knoll (clear of tree trunks & holes!)
    w_bushes = []
    for _ in range(rng.integers(2, 5)):
        for _ in range(20):
            ang_b = rng.uniform(0, TAU)
            rr = rng.uniform(0.25, 1.05)
            bx, by = cx_w + rx * rr * math.cos(ang_b), cy_w + ry * rr * math.sin(ang_b)
            if (all(math.hypot(bx - hx, by - hy) > 18 for hx, hy, _, _, _ in holes)
                    and all(math.hypot(bx - tx, by - ty) > 15 for tx, ty in w_trees)
                    and all(math.hypot(bx - ox, by - oy) > 13 for ox, oy in w_bushes)):
                w_bushes.append((bx, by))
                props.append((P.depth(bx, by), "bush", (bx, by, h_fn(bx, by))))
                break

    # Grass tufts across the grassy hillside & clearing
    for _ in range(rng.integers(10, 16)):
        ang_g = rng.uniform(0, TAU)
        rr = rng.uniform(0.15, 1.25)
        gx, gy = cx_w + rx * rr * math.cos(ang_g), cy_w + ry * rr * math.sin(ang_g)
        if all(math.hypot(gx - hx, gy - hy) > 13 for hx, hy, _, _, _ in holes):
            props.append((P.depth(gx, gy), "tuft", (gx, gy, h_fn(gx, gy))))

    for _, kind, args in sorted(props, key=lambda it: -it[0]):
        if kind == "pit":
            hx, hy, hz, rm, ex_ang, is_main = args
            _burrow_pit_3d(pen, P, rng, hx, hy, hz, rm, ex_ang, spoil=True, roots=True)
            if is_main or rng.random() < 0.45:
                _trail_3d(pen, P, rng, hx + math.cos(ex_ang) * rm * 0.85,
                          hy + math.sin(ex_ang) * rm * 0.85,
                          (math.cos(ex_ang), math.sin(ex_ang)), rm * 0.72, rng.uniform(14, 24), z_fn=h_fn, stones=False)
        elif kind == "tree":
            _place_tree(pen, *P(*args), rng.uniform(22, 34), kind=str(rng.choice(["round", "spruce"])))
        elif kind == "bush":
            pen.bush(*P(*args), rng.uniform(4.8, 7.6), outline=True)
        else:
            pen.tuft(*P(*args), rng.uniform(2.8, 4.6))


# --------------------------------------------------------------------------- nests & crag eyries

def _twig(pen, x, y, ang, L, w_scale=0.6):
    rng = pen.rng
    d = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-d[1], d[0]])
    c = np.array([x, y])
    pen.stroke([c - d * L / 2, c + n * rng.normal(0, L * 0.09), c + d * L / 2], pen.width * w_scale,
               taper=(0.25, 0.25), smooth=1, wobble=0.1)


def nest(pen, cx, cy, rx, ry, contents=None):
    """Densely woven braided-twig wyvern/roc eyrie cradling speckled eggs or bones (no white donut gap!)."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    contents = contents or str(rng.choice(["eggs", "eggs", "bones"]))
    bh = min(ry * 1.55, rx * 0.48)
    body = list(ellipse_pts(cx, cy, rx * 1.02, ry * 1.02, 40, math.pi, TAU)) + \
        list(ellipse_pts(cx, cy + ry * 0.22, rx * 1.0, bh, 30, 0, math.pi))
    cv.erase(body)

    def rim_twig(theta, f_lo, f_hi, w_s=0.6):
        f = rng.uniform(f_lo, f_hi)
        x, y = cx + rx * f * math.cos(theta), cy + ry * f * math.sin(theta)
        tang = math.atan2(ry * math.cos(theta), -rx * math.sin(theta))
        _twig(pen, x, y, tang + rng.normal(0, 0.35), rng.uniform(8, 16), w_scale=w_s)

    # Back braided rim twigs
    for _ in range(int(rx * 1.1)):
        rim_twig(rng.uniform(math.pi, TAU), 0.82, 1.12, w_s=rng.uniform(0.5, 0.72))

    # Recessed shadowed interior nest cup
    inner = ellipse_pts(cx, cy - ry * 0.04, rx * 0.76, ry * 0.56, 30)
    cv.erase(inner)
    pen.hatch(inner, rng.uniform(4, 16), 1.45, W0 * 0.62, coverage=(0.75, 1.0))
    pen.hatch(clip_halfplane(list(inner), (0.0, -1.0), -(cy - ry * 0.02)), rng.uniform(-14, -4), 1.8, W0 * 0.5, coverage=(0.6, 0.95))

    # Eggs or picked-clean prey remains nestled inside the bowl
    items = []
    if contents == "eggs":
        ex = cx + rng.uniform(-rx * 0.20, rx * 0.20)
        for _ in range(rng.integers(3, 6)):
            items.append(("egg", ex + rng.uniform(-rx * 0.28, rx * 0.28), cy + rng.uniform(-0.10, 0.22) * ry))
    else:
        for _ in range(rng.integers(2, 5)):
            items.append(("bone", cx + rng.uniform(-rx * 0.45, rx * 0.45), cy + rng.uniform(-ry * 0.15, ry * 0.28)))
        if rng.random() < 0.65:
            items.append(("skull", cx + rng.uniform(-rx * 0.35, rx * 0.35), cy + ry * 0.22))

    for kind, x, y in sorted(items, key=lambda it: it[2]):
        if kind == "egg":
            ew, eh = rng.uniform(3.8, 5.0), rng.uniform(5.2, 6.8)
            e = ellipse_pts(x, y - eh, ew, eh, 18)
            tilt = rng.normal(0, 0.24)
            e = np.stack([x + (e[:, 0] - x) + (e[:, 1] - y) * tilt * -1, e[:, 1]], 1)
            cv.erase(e)
            sh = clip_halfplane(list(e), (0.85, 0.52), x * 0.85 + (y - eh) * 0.52 + ew * 0.15)
            if len(sh) >= 3:
                pen.hatch(sh, 75, 1.35, W0 * 0.38)
            pen.closed(e, W0 * 0.65, wobble=0)
            for _ in range(rng.integers(3, 7)):
                cv.disc(x + rng.uniform(-ew * 0.55, ew * 0.4), y - eh + rng.uniform(-eh * 0.55, eh * 0.4), 0.45)
        elif kind == "bone":
            bone(pen, x, y, rng.uniform(7, 11), rng.uniform(0, math.pi))
        else:
            skull(pen, x, y, 5.2, horns=False)

    # Erase ONLY the narrow front bowl rim over the lower eggs, then weave rich layered branches across ALL of it!
    front_wall = list(ellipse_pts(cx, cy - ry * 0.04, rx * 0.76, ry * 0.56, 22, 0, math.pi)) + \
        list(ellipse_pts(cx, cy + ry * 0.22, rx * 1.0, bh, 24, math.pi, 0))
    cv.erase(front_wall)

    # Subtle east/underside shading on the woven cup bowl
    bowl_sh = clip_halfplane(front_wall, (0.55, 0.83), cx * 0.55 + (cy + bh * 0.3) * 0.83 + 1.0)
    if len(bowl_sh) >= 3:
        pen.hatch(bowl_sh, 20, 2.2, W0 * 0.45, coverage=(0.5, 0.9))

    # Rich dense woven twigs wrapping continuously over the ENTIRE front lip & cupped underside!
    for _ in range(int(rx * 2.1)):
        th = rng.uniform(-0.08, math.pi + 0.08)
        rim_twig(th, 0.74, 1.10, w_s=rng.uniform(0.52, 0.78))
    for _ in range(int(rx * 1.9)):  # cupped underside woven wall
        th = rng.uniform(0.08, math.pi - 0.08)
        f = rng.uniform(0.45, 1.02)
        x = cx + rx * 1.04 * f * math.cos(th)
        y = cy + ry * 0.24 + bh * 1.02 * f * math.sin(th)
        tang = math.atan2(bh * 1.02 * math.cos(th), -rx * 1.04 * math.sin(th))
        dark = (x > cx - rx * 0.15) or (y > cy + bh * 0.55)
        _twig(pen, x, y, tang + rng.normal(0, 0.4), rng.uniform(7, 14), w_scale=0.72 if dark else 0.52)


def nest_lair(pen, P, rng, crag=None):
    cx, by = P.ox, P.oy
    rx = rng.uniform(32, 44)
    ry = rx * rng.uniform(0.48, 0.56)
    on_crag = rng.random() < 0.6 if crag is None else crag
    foot = rng.uniform(74, 102) / 2 if on_crag else rx * 1.05
    ring = []
    for _ in range(rng.integers(4, 9)):
        t = rng.uniform(0, math.tau)
        d = foot * rng.uniform(1.05, 1.55)
        x, y = cx + d * math.cos(t), by - (0 if on_crag else ry * 0.9) + d * 0.5 * math.sin(t)
        ring.append((y, x, str(rng.choice(["tuft", "tuft", "boulder", "stone", "bone"]))))

    def draw_ring(sel):
        for y, x, kind in sorted(r for r in ring if sel(r)):
            if kind == "tuft":
                pen.tuft(x, y, rng.uniform(3.5, 6))
            elif kind == "boulder":
                boulder(pen, x, y, rng.uniform(4.5, 8.5))
            elif kind == "stone":
                pen.stone(x, y, rng.uniform(1.5, 3))
            else:
                bone(pen, x, y, rng.uniform(7, 12), rng.uniform(0, math.pi))

    mid = by - (0 if on_crag else ry * 0.9)
    draw_ring(lambda r: r[0] < mid)
    if on_crag:
        # Dramatic faceted sea-stack / crag pinnacle supporting the giant eyrie (NO blank white sock!)
        ch, bw, tw = rng.uniform(66, 98), foot * 2, rx * rng.uniform(0.75, 1.02)
        flare = rng.uniform(0.42, 0.65)
        lean = rng.normal(0, 7)

        def edge(sgn):
            out = []
            for f in np.linspace(0, 1, 10):
                # stepped rocky ledges widening cleanly at the base & subtly flaring into shelf under nest
                w = bw / 2 - (bw - tw) / 2 * (f ** flare) + (2.5 * max(0, f - 0.8) / 0.2)
                jit = rng.normal(0, 3.8) if (0 < f < 1) else 0.0
                out.append((cx + lean * f + sgn * (w + jit), by - ch * f))
            return out

        left, right = edge(-1), edge(1)[::-1]
        top = [(cx + lean + rng.uniform(-tw * 0.2, tw * 0.2), by - ch - rng.uniform(0, 4))]
        crag_pts = left + top + right
        rock(pen, crag_pts, base_sag=bw * 0.22, facets=True, strata=True)
        # Bold central vertical cliff ridge + ledge tufts for unmistakable monumental scale
        r_mid = [(cx + lean * 0.85, by - ch * 0.85),
                 (cx + lean * 0.45 + rng.normal(0, 4), by - ch * 0.48),
                 (cx + rng.normal(0, 6), by - ch * 0.12)]
        pen.stroke(r_mid, pen.width * 0.85, taper=(0.1, 0.7), wobble=0.25)
        for k in (2, 5):
            if k < len(left):
                pen.tuft(left[k][0] + 2, left[k][1], rng.uniform(3.0, 4.8))
        nest(pen, cx + lean, by - ch - ry * 0.08, rx, ry)
    else:
        nest(pen, cx, by - ry * 0.9, rx, ry)
    draw_ring(lambda r: r[0] >= mid)
    if rng.random() < 0.72:
        remains(pen, cx, by + (foot * 0.42 if on_crag else 9), 40, n=int(rng.integers(2, 6)))
