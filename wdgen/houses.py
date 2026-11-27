"""Intact settlements: single houses, hamlets and villages.

Houses are convex solids (walls + gable roof) drawn with back-face culling, so any
orientation shades correctly: east-facing walls/roofs are in shadow. Houses and hamlets
share one scale and crop tightly without stray clutter so they compose into towns.
"""
import math

import numpy as np

from .flora import tree
from .ink import chaikin, ellipse_pts
from .proj import sort_boxes
from .solids import box, cylinder, pyramid


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _to_viewer(P):
    return np.array([P.sa * P.ce, -P.ca * P.ce, P.se])


def _lerp3(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


# --------------------------------------------------------------------------- geometry

def house_faces(x0, y0, W, D, H, R, ridge_x, o=1.6):
    x1, y1 = x0 + W, y0 + D

    def wall(a, b, n, gable):
        pts = [(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], H)]
        if gable:
            pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, H + R))
        pts.append((a[0], a[1], H))
        return dict(kind="wall", pts=pts, n=_unit(n), a=a, b=b, H=H, R=R, gable=gable)

    if ridge_x:
        faces = [wall((x0, y0), (x1, y0), (0, -1, 0), False), wall((x1, y1), (x0, y1), (0, 1, 0), False),
                 wall((x1, y0), (x1, y1), (1, 0, 0), True), wall((x0, y1), (x0, y0), (-1, 0, 0), True)]
        eh, ym = H - o * R / (D / 2), y0 + D / 2
        faces += [dict(kind="roof", n=_unit((0, -R, D / 2)), south=True,
                       quad=[(x0 - o, y0 - o, eh), (x1 + o, y0 - o, eh), (x1 + o, ym, H + R), (x0 - o, ym, H + R)]),
                  dict(kind="roof", n=_unit((0, R, D / 2)), south=False,
                       quad=[(x1 + o, y1 + o, eh), (x0 - o, y1 + o, eh), (x0 - o, ym, H + R), (x1 + o, ym, H + R)])]
    else:
        faces = [wall((x0, y0), (x1, y0), (0, -1, 0), True), wall((x1, y1), (x0, y1), (0, 1, 0), True),
                 wall((x1, y0), (x1, y1), (1, 0, 0), False), wall((x0, y1), (x0, y0), (-1, 0, 0), False)]
        eh, xm = H - o * R / (W / 2), x0 + W / 2
        faces += [dict(kind="roof", n=_unit((R, 0, W / 2)), south=False,
                       quad=[(x1 + o, y0 - o, eh), (x1 + o, y1 + o, eh), (xm, y1 + o, H + R), (xm, y0 - o, H + R)]),
                  dict(kind="roof", n=_unit((-R, 0, W / 2)), south=False,
                       quad=[(x0 - o, y1 + o, eh), (x0 - o, y0 - o, eh), (xm, y0 - o, H + R), (xm, y1 + o, H + R)])]
    return faces


def leanto_faces(x1, ya, yb, Wa, Hh, Hl, o=1.2):
    """Shed annex against the east wall at x = x1, roof sloping down to the east."""
    xe = x1 + Wa
    return [
        dict(kind="wall", pts=[(x1, ya, 0), (xe, ya, 0), (xe, ya, Hl), (x1, ya, Hh)], n=_unit((0, -1, 0)),
             a=(x1, ya), b=(xe, ya), H=Hl, R=0, gable=False, plain=True),
        dict(kind="wall", pts=[(xe, ya, 0), (xe, yb, 0), (xe, yb, Hl), (xe, ya, Hl)], n=_unit((1, 0, 0)),
             a=(xe, ya), b=(xe, yb), H=Hl, R=0, gable=False, plain=True),
        dict(kind="roof", n=_unit((Hh - Hl, 0, Wa)), south=False,
             quad=[(xe + o, ya - o, Hl - 0.6), (xe + o, yb + o, Hl - 0.6), (x1, yb + o, Hh), (x1, ya - o, Hh)]),
    ]


# --------------------------------------------------------------------------- drawing

def _wall(pen, P, f, st):
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    poly = [P(*p) for p in f["pts"]]
    cv.erase(poly)
    shadow = f["n"][0] > 0.3
    a, b = np.array(f["a"], float), np.array(f["b"], float)
    L = float(np.hypot(*(b - a)))
    u = (b - a) / max(1e-6, L)
    H = f["H"]

    def Wp(s, h):
        g = a + u * s
        return P(g[0], g[1], h)

    if st.get("timber") and not shadow and not f.get("plain"):
        posts = np.linspace(0, L, max(2, int(L / 7.5)) + 1)
        mid = H * (0.52 if st["floors"] == 2 else 0.58)
        pen.stroke([Wp(0, mid), Wp(L, mid)], W0 * 0.58, taper=(0.05, 0.05), wobble=0.1)
        for s in posts[1:-1]:
            pen.stroke([Wp(s, 0), Wp(s, H)], W0 * 0.52, taper=(0.05, 0.05), wobble=0.1)
        for s0, s1 in zip(posts[:-1], posts[1:]):
            if rng.random() < 0.45:
                lo, hi = (0, mid) if rng.random() < 0.5 else (mid, H)
                p, q = (s0, s1) if rng.random() < 0.5 else (s1, s0)
                pen.stroke([Wp(p, lo), Wp(q, hi)], W0 * 0.46, taper=(0.05, 0.05), wobble=0.08)
        if f["gable"]:
            pen.stroke([Wp(L / 2, H), Wp(L / 2, H + f["R"] * 0.85)], W0 * 0.48, taper=(0.05, 0.3), wobble=0)
            pen.stroke([Wp(L * 0.24, H), Wp(L * 0.5, H + f["R"] * 0.45)], W0 * 0.4, taper=(0.05, 0.2), wobble=0)
            pen.stroke([Wp(L * 0.76, H), Wp(L * 0.5, H + f["R"] * 0.45)], W0 * 0.4, taper=(0.05, 0.2), wobble=0)
    elif not shadow and not f.get("plain"):
        # Subtle stone plinth foundation & corner quoins so plaster/stone walls feel crafted
        if st.get("plinth"):
            pen.stroke([Wp(0.4, 2.2), Wp(L - 0.4, 2.2)], W0 * 0.42, taper=(0.08, 0.08), wobble=0.15)
        if st.get("quoins"):
            for hh in np.linspace(2.5, H - 2.0, max(2, int(H / 4.8))):
                ql = rng.uniform(1.8, 3.0)
                pen.stroke([Wp(0, hh), Wp(ql, hh)], W0 * 0.4, taper=(0.05, 0.3), wobble=0.05)
                pen.stroke([Wp(L - ql, hh), Wp(L, hh)], W0 * 0.4, taper=(0.3, 0.05), wobble=0.05)

    if shadow:
        # Clean east-wall vertical shadow hatching (readable woodcut spacing!)
        pen.hatch(poly, 82 + rng.normal(0, 2), rng.uniform(1.65, 2.0), W0 * 0.68, coverage=(0.78, 1.0))

    if not f.get("plain"):
        door = None
        if np.allclose(f["n"], (0, -1, 0)) and st["floors"] >= 0:
            ds = L * rng.uniform(0.32, 0.68) if st["kind"] != "barn" else L / 2
            dw = 5.0 if st["kind"] != "barn" else min(12.5, L * 0.42)
            door = (ds - dw / 2, ds + dw / 2)
            dh = min(8.8, H * 0.72) if st["kind"] != "barn" else H * 0.78
            dq = [Wp(door[0], 0), Wp(door[1], 0), Wp(door[1], dh), Wp(door[0], dh)]
            if st["kind"] == "barn":
                cv.erase(dq)
                pen.closed(dq, W0 * 0.72)
                pen.stroke([dq[0], dq[2]], W0 * 0.48, taper=(0, 0), wobble=0)
                pen.stroke([dq[1], dq[3]], W0 * 0.48, taper=(0, 0), wobble=0)
                pen.stroke([Wp(ds, 0), Wp(ds, dh)], W0 * 0.5, taper=(0, 0), wobble=0)
            else:
                if st.get("arch_door"):
                    arch = [Wp(door[0], 0), Wp(door[0], dh * 0.72), Wp(ds, dh * 1.06),
                            Wp(door[1], dh * 0.72), Wp(door[1], 0)]
                    pen.blot(arch, jitter=0.06)
                else:
                    pen.blot(dq, jitter=0.08)
                    # crisp timber lintel beam above door
                    pen.stroke([Wp(door[0] - 0.8, dh + 0.5), Wp(door[1] + 0.8, dh + 0.5)],
                               W0 * 0.55, taper=(0.1, 0.1), wobble=0.05)

        rows = {0: [], 1: [H * 0.36], 2: [H * 0.16, H * 0.60]}.get(min(2, st["floors"]), [])
        n = max(1, int((L - 5) / 11))
        ww, wh = st["win"]
        for hh in rows:
            for s in np.linspace(0, L, n + 2)[1:-1]:
                s0, s1 = s - ww / 2, s + ww / 2
                if door and not (s1 < door[0] - 1.8 or s0 > door[1] + 1.8) and hh < 10:
                    continue
                if st.get("lancet"):
                    q = [Wp(s0, hh), Wp(s1, hh), Wp(s1, hh + wh * 0.78), Wp(s, hh + wh * 1.08), Wp(s0, hh + wh * 0.78)]
                else:
                    q = [Wp(s0, hh), Wp(s1, hh), Wp(s1, hh + wh), Wp(s0, hh + wh)]
                pen.blot(q, jitter=0.06)
                if not shadow:
                    # subtle stone window sill
                    pen.stroke([Wp(s0 - 0.5, hh - 0.3), Wp(s1 + 0.5, hh - 0.3)], W0 * 0.42, taper=(0.1, 0.1), wobble=0)
                if st["cross"] and not shadow and not st.get("lancet"):
                    c1, c2 = Wp(s, hh + 0.35), Wp(s, hh + wh - 0.35)
                    cv.erase([(c1[0] - 0.35, c1[1]), (c1[0] + 0.35, c1[1]), (c2[0] + 0.35, c2[1]), (c2[0] - 0.35, c2[1])])
                    l1, l2 = Wp(s0 + 0.35, hh + wh / 2), Wp(s1 - 0.35, hh + wh / 2)
                    cv.erase([(l1[0], l1[1] - 0.35), (l2[0], l2[1] - 0.35), (l2[0], l2[1] + 0.35), (l1[0], l1[1] + 0.35)])
        if f["gable"] and f["R"] > 8.5 and rng.random() < 0.65:
            h0 = H + f["R"] * 0.24
            pen.blot([Wp(L / 2 - 1.5, h0), Wp(L / 2 + 1.5, h0), Wp(L / 2 + 1.5, h0 + 3.4), Wp(L / 2 - 1.5, h0 + 3.4)],
                     jitter=0.05)
    pen.closed(poly, W0 * 0.88, overshoot=0.4, wobble=0.2)


def _dormer(pen, P, S, u_mid, st):
    """Small projecting gabled dormer window nestled on a south roof slope."""
    W0 = pen.width
    du = 0.12
    p0 = S(u_mid - du, 0.22)
    p1 = S(u_mid + du, 0.22)
    h_d = 5.4
    peak_d = 3.2
    front = [p0, p1, (p1[0], p1[1] - h_d), ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5 - h_d - peak_d),
             (p0[0], p0[1] - h_d)]
    # small side roof plane extending back into main roof slope
    back_r = S(u_mid + du * 0.8, 0.72)
    back_peak = S(u_mid, 0.72)
    side_roof = [(p1[0], p1[1] - h_d), back_r, back_peak, front[3]]
    pen.cv.erase(side_roof)
    pen.hatch(side_roof, 76, 1.5, W0 * 0.48, coverage=(0.7, 1.0))
    pen.closed(side_roof, W0 * 0.68, wobble=0.1)
    pen.cv.erase(front)
    # tiny dark dormer window
    wm = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5 - h_d * 0.48)
    pen.blot([(wm[0] - 1.3, wm[1] + 1.6), (wm[0] + 1.3, wm[1] + 1.6),
              (wm[0] + 1.3, wm[1] - 1.7), (wm[0] - 1.3, wm[1] - 1.7)], jitter=0.04)
    pen.closed(front, W0 * 0.72, wobble=0.1)


def _roof(pen, P, f, st):
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    e0, e1, r1, r0 = f["quad"]
    poly = [P(*e0), P(*e1), P(*r1), P(*r0)]
    cv.erase(poly)
    shadow = f["n"][0] > 0.3
    # Scale stroke spacing by ACTUAL projected screen width so east roofs never collapse into solid black!
    L_scr = max(6.0, math.dist(poly[0], poly[1]))

    def S(u, v):
        return P(*_lerp3(_lerp3(e0, e1, u), _lerp3(r0, r1, u), v))

    if st["roof"] == "thatch":
        # Layered organic thatch bundles (tapered wisps, heavier eave)
        sp_px = 1.85 if shadow else 2.15
        sp = sp_px / L_scr
        for u in np.arange(sp * rng.uniform(0.2, 0.8), 1.0, sp):
            top = rng.uniform(0.82, 1.0) if shadow else rng.uniform(0.45, 0.96)
            pen.stroke([S(u, 0.01), S(u + rng.normal(0, sp * 0.12), top)],
                       W0 * (0.56 if shadow else 0.44), taper=(0.05, 0.65), wobble=0.14)
        # subtle overlapping thatch tier wave
        for v in (0.48,):
            pts = [S(u, v + rng.normal(0, 0.02)) for u in np.linspace(0.03, 0.97, 10)]
            pen.stroke(pts, W0 * 0.36, taper=(0.25, 0.25), wobble=0.22)
    else:
        # Crisp hand-inked shingle / tile roof with staggered courses & readable east hatching
        sp_px = 1.95 if shadow else 2.55
        sp = sp_px / L_scr
        for u in np.arange(sp * rng.uniform(0.2, 0.8), 1.0, sp):
            top = 0.98 if shadow else rng.uniform(0.55, 0.98)
            pen.stroke([S(u, 0.02), S(u + rng.normal(0, sp * 0.06), top)],
                       W0 * (0.56 if shadow else 0.42), taper=(0.06, 0.55), wobble=0.1)
        rows = (0.28, 0.54, 0.78) if shadow else (0.33, 0.66)
        for v in rows:
            pen.stroke([S(0.02, v), S(0.98, v)], W0 * (0.40 if shadow else 0.34), taper=(0.15, 0.15), wobble=0.2)

    pen.closed(poly, W0 * 0.88, overshoot=0.45, wobble=0.18)
    # bold shadow eave + crisp ridge cap line
    pen.stroke([poly[0], poly[1]], W0 * 1.25, taper=(0.05, 0.05), wobble=0.18)
    pen.stroke([poly[2], poly[3]], W0 * 0.98, taper=(0.05, 0.05), wobble=0.12)

    if st.get("dormer") and f.get("south") and L_scr > 24:
        _dormer(pen, P, S, rng.uniform(0.35, 0.65), st)


def draw_faces(pen, P, faces, st):
    tv = _to_viewer(P)
    vis = [f for f in faces if f["n"] @ tv > 0.03]
    for f in vis:
        if f["kind"] == "wall":
            _wall(pen, P, f, st)
    for f in vis:
        if f["kind"] == "roof":
            _roof(pen, P, f, st)


# --------------------------------------------------------------------------- houses

def plan_house(rng, kind=None):
    kind = kind or str(rng.choice(["cottage", "house", "long", "barn"], p=[0.4, 0.35, 0.15, 0.1]))
    if kind == "cottage":
        W, D, H, floors = rng.uniform(26, 36), rng.uniform(18, 24), rng.uniform(11, 14), 1
    elif kind == "house":
        W, D, H, floors = rng.uniform(30, 42), rng.uniform(20, 26), rng.uniform(19, 24), 2
    elif kind == "long":
        W, D, H, floors = rng.uniform(46, 60), rng.uniform(18, 22), rng.uniform(11, 14), 1
    else:
        W, D, H, floors = rng.uniform(36, 48), rng.uniform(24, 30), rng.uniform(12, 15), 0
    ridge_x = rng.random() < (0.8 if kind in ("long", "barn") else 0.6)
    if not ridge_x:
        W, D = D, W
    span = (D if ridge_x else W) / 2
    timber = kind in ("cottage", "house") and rng.random() < 0.38
    return dict(kind=kind, W=W, D=D, H=H, R=span * rng.uniform(0.85, 1.25), ridge_x=ridge_x, floors=floors,
                timber=timber, plinth=(not timber) and rng.random() < 0.6, quoins=(not timber) and rng.random() < 0.45,
                arch_door=kind != "barn" and rng.random() < 0.3,
                dormer=kind == "house" and ridge_x and rng.random() < 0.35,
                chimney=kind != "barn" and rng.random() < 0.68,
                smoke=rng.random() < 0.22,
                leanto=ridge_x and kind != "long" and rng.random() < 0.28,
                cross=rng.random() < 0.55, roof="thatch" if rng.random() < 0.38 else "tile",
                win=(rng.uniform(3.2, 4.2), rng.uniform(4.5, 5.6)))


def footprint(pl, X0, Y0):
    extra = pl["W"] * 0.35 if pl["leanto"] else 0
    return (X0 - 2, Y0 - 2, X0 + pl["W"] + 2 + extra, Y0 + pl["D"] + 2)


def draw_house(pen, P, X0, Y0, pl):
    rng, W0 = pen.rng, pen.width
    W, D, H, R = pl["W"], pl["D"], pl["H"], pl["R"]
    draw_faces(pen, P, house_faces(X0, Y0, W, D, H, R, pl["ridge_x"]), pl)
    if pl["chimney"]:
        if pl["ridge_x"]:
            cx, cy = X0 + W * rng.uniform(0.16, 0.84), Y0 + D * rng.uniform(0.30, 0.42)
            hb = H + R * (1 - abs(cy - (Y0 + D / 2)) / (D / 2))
        else:
            cx, cy = X0 + W * rng.uniform(0.58, 0.70), Y0 + D * rng.uniform(0.20, 0.80)
            hb = H + R * (1 - abs(cx - (X0 + W / 2)) / (W / 2))
        cs = pl.get("chimney_scale", 1.0)
        ctop = H + R + rng.uniform(2.8, 5.8) * cs
        box(pen, P, cx - 2.1 * cs, cy - 2.1 * cs, cx + 2.1 * cs, cy + 2.1 * cs, hb - 2.4 * cs, ctop)
        # small overhanging stone chimney cap rim
        box(pen, P, cx - 2.6 * cs, cy - 2.6 * cs, cx + 2.6 * cs, cy + 2.6 * cs, ctop, ctop + 1.3 * cs, shade=True)
        if pl.get("smoke"):
            # delicate short curling hearth-smoke wisp
            sx, sy = P(cx, cy, ctop + 2.2 * cs)
            pts = [(sx, sy), (sx + 2.5, sy - 4.0), (sx + 1.0, sy - 8.0), (sx + 4.5, sy - 11.5)]
            pen.stroke(pts, W0 * 0.42, taper=(0.35, 0.65), smooth=2, wobble=0.2)
    if pl["leanto"]:
        Wa = W * rng.uniform(0.22, 0.32)
        Hh = H * rng.uniform(0.75, 0.9)
        draw_faces(pen, P, leanto_faces(X0 + W, Y0 + D * 0.12, Y0 + D * 0.88, Wa, Hh, Hh - Wa * 0.45),
                   dict(pl, timber=False, dormer=False))


def _draw_cross(pen, P, x, y, z, h=9.0, w_scale=0.72):
    """Crisp upright church cross finial projected at world (x, y, z)."""
    W0 = pen.width
    ax, ay = P(x, y, z)
    pen.stroke([(ax, ay), (ax, ay - h)], W0 * w_scale, taper=(0.0, 0.12), wobble=0)
    bar_y = ay - h * 0.68
    bar_hw = h * 0.36
    pen.stroke([(ax - bar_hw, bar_y), (ax + bar_hw, bar_y)], W0 * (w_scale * 0.88), taper=(0.08, 0.08), wobble=0)


def _draw_bell_tower(pen, P, rng, tx0, tx1, ty0, ty1, z0, th, spire_h, crenellated=False):
    """Gothic stone bell tower with stepped buttresses, rose/clock oculus, twin belfry arches & cross steeple."""
    W0 = pen.width
    tx = (tx0 + tx1) * 0.5
    ty = (ty0 + ty1) * 0.5
    tw = tx1 - tx0
    td = ty1 - ty0
    # Stepped corner plinth & buttress base (only when grounded at z0 == 0)
    if z0 <= 0.5:
        box(pen, P, tx0 - 1.6, ty0 - 1.6, tx1 + 1.6, ty1 + 1.6, 0, th * 0.24)
        # Compact stepped buttress piers flanking south corners
        for bx0, bx1 in ((tx0 - 1.2, tx0 + 2.2), (tx1 - 2.2, tx1 + 1.2)):
            box(pen, P, bx0, ty0 - 1.8, bx1, ty0 + 1.0, 0, th * 0.44)
    # Main stone tower shaft
    box(pen, P, tx0, ty0, tx1, ty1, z0, th)

    # Horizontal stone stringcourse cornices along south & east faces
    for frac in (0.62, 0.96):
        hz = z0 + (th - z0) * frac
        pen.stroke([P(tx0, ty0, hz), P(tx1, ty0, hz)], W0 * 0.58, taper=(0.04, 0.04), wobble=0.08)
        pen.stroke([P(tx1, ty0, hz), P(tx1, ty1, hz)], W0 * 0.52, taper=(0.04, 0.04), wobble=0.08)

    # Circular clock / rose oculus window on the lit south shaft face
    if th - z0 > 32:
        oc_z = z0 + (th - z0) * 0.48
        ox, oy = P(tx, ty0, oc_z)
        oc_r = min(3.2, tw * 0.22)
        ring = ellipse_pts(ox, oy, oc_r, oc_r * 1.05, 16)
        pen.cv.erase(ring)
        pen.closed(ring, W0 * 0.62, wobble=0.05)
        pen.stroke([(ox, oy - oc_r * 0.85), (ox, oy + oc_r * 0.85)], W0 * 0.36, taper=(0, 0), wobble=0)
        pen.stroke([(ox - oc_r * 0.85, oy), (ox + oc_r * 0.85, oy)], W0 * 0.36, taper=(0, 0), wobble=0)

    # Twin pointed gothic belfry louver openings on south face (and east face)
    bz0 = z0 + (th - z0) * 0.70
    bh = (th - z0) * 0.20
    bw = min(2.2, tw * 0.16)
    for off in (-tw * 0.20, tw * 0.20):
        bx = tx + off
        pen.blot([P(bx - bw * 0.5, ty0, bz0), P(bx + bw * 0.5, ty0, bz0),
                  P(bx + bw * 0.5, ty0, bz0 + bh * 0.72), P(bx, ty0, bz0 + bh * 1.08),
                  P(bx - bw * 0.5, ty0, bz0 + bh * 0.72)], jitter=0.04)
    # Single dark arched belfry louver on shadowed east face
    ebw = min(2.4, td * 0.20)
    pen.blot([P(tx1, ty - ebw * 0.5, bz0), P(tx1, ty + ebw * 0.5, bz0),
              P(tx1, ty + ebw * 0.5, bz0 + bh * 0.72), P(tx1, ty, bz0 + bh * 1.08),
              P(tx1, ty - ebw * 0.5, bz0 + bh * 0.72)], jitter=0.04)

    if crenellated:
        # English country parish style: crenellated stone parapet + short spirelet & tall cross!
        box(pen, P, tx0 - 0.8, ty0 - 0.8, tx1 + 0.8, ty1 + 0.8, th, th + 3.2)
        for cx_p in (tx0 - 0.8, tx1 - 1.8):
            for cy_p in (ty0 - 0.8, ty1 - 1.8):
                box(pen, P, cx_p, cy_p, cx_p + 2.6, cy_p + 2.6, th + 3.2, th + 6.4)
        apex_h = th + 3.2 + spire_h * 0.55
        pyramid(pen, P, tx0 + 1.4, ty0 + 1.4, tx1 - 1.4, ty1 - 1.4, th + 3.2, apex_h)
        _draw_cross(pen, P, tx, ty, apex_h, h=9.5, w_scale=0.76)
    else:
        # Classic soaring shingled parish steeple + bold cross finial
        apex_h = th + spire_h
        pyramid(pen, P, tx0 - 1.3, ty0 - 1.3, tx1 + 1.3, ty1 + 1.3, th, apex_h)
        _draw_cross(pen, P, tx, ty, apex_h, h=10.0, w_scale=0.78)


def plan_chapel(rng):
    """Plan a medieval village parish church (west-steeple basilica or cruciform transept church)."""
    W = rng.uniform(50, 62)
    D = rng.uniform(20, 24)
    H = rng.uniform(17.5, 21.5)
    R = (D / 2) * rng.uniform(1.12, 1.34)
    cruciform = rng.random() < 0.48
    crossing_tower = cruciform and rng.random() < 0.45
    s = rng.uniform(7.2, 8.6)
    tr_proj = D * rng.uniform(0.36, 0.48) if cruciform else 0.0
    pl = dict(kind="church", W=W, D=D, H=H, R=R, ridge_x=True, floors=1,
              timber=False, plinth=True, quoins=True, arch_door=True, lancet=True,
              dormer=False, chimney=False, smoke=False, leanto=False, cross=False,
              roof="tile", win=(2.8, 7.6),
              cruciform=cruciform, crossing_tower=crossing_tower,
              tower_s=s, tr_proj=tr_proj,
              crenellated=(not crossing_tower) and rng.random() < 0.22)
    return pl


def chapel_footprint(pl, X0, Y0):
    s = pl["tower_s"]
    x_min = X0 - (0.0 if pl["crossing_tower"] else s * 1.75) - 3.0
    y_min = Y0 - pl["tr_proj"] - 4.0
    x_max = X0 + pl["W"] + 3.0
    y_max = Y0 + pl["D"] + pl["tr_proj"] + 3.0
    return (x_min, y_min, x_max, y_max)


def chapel(pen, P, X0, Y0, rng, pl=None):
    """Village parish church: buttressed gothic nave, optional cruciform transepts, tall bell steeple & crosses."""
    pl = pl or plan_chapel(rng)
    W, D, H, R = pl["W"], pl["D"], pl["H"], pl["R"]
    s = pl["tower_s"]
    ty = Y0 + D * 0.5
    cruciform = pl["cruciform"]
    crossing_tower = pl["crossing_tower"]

    tr_w = D * rng.uniform(0.76, 0.88)
    tr_x0 = X0 + W * 0.30
    tr_x1 = tr_x0 + tr_w
    tr_proj = pl["tr_proj"]
    tr_st = dict(pl, W=tr_w, D=tr_proj + D * 0.5, H=H * 0.96, R=(tr_w / 2) * 1.16,
                 ridge_x=False, floors=1, win=(2.6, 6.8))

    # 1. Far north transept wing (strictly behind nave)
    if cruciform:
        draw_faces(pen, P, house_faces(tr_x0, ty, tr_w, D * 0.5 + tr_proj, tr_st["H"], tr_st["R"], False),
                   dict(tr_st, floors=0))

    # 2. West bell tower (strictly west/behind the nave in az=28° view)
    if not crossing_tower:
        tx0, tx1 = X0 - s * 1.55, X0 + s * 0.45
        th = rng.uniform(50, 64)
        spire_h = rng.uniform(28, 40)
        _draw_bell_tower(pen, P, rng, tx0, tx1, ty - s, ty + s, 0.0, th, spire_h, crenellated=pl["crenellated"])

    # 3. Main gothic stone nave + exterior south wall buttresses + east gable cross
    draw_house(pen, P, X0, Y0, pl)
    for frac in (0.14, 0.86) if cruciform else (0.16, 0.42, 0.68, 0.90):
        bx = X0 + W * frac
        box(pen, P, bx - 1.3, Y0 - 2.1, bx + 1.3, Y0 + 0.4, 0, H * 0.72, inset=0.5)
    _draw_cross(pen, P, X0 + W, ty, H + R, h=6.2, w_scale=0.58)

    # 4. Near south transept wing (projects forward in front of nave, with arched portal & gable rose/lancet!)
    if cruciform:
        draw_faces(pen, P, house_faces(tr_x0, Y0 - tr_proj, tr_w, tr_proj + D * 0.5, tr_st["H"], tr_st["R"], False),
                   tr_st)
        # Small stone rose oculus tucked into the south transept gable
        gx, gy = P(tr_x0 + tr_w * 0.5, Y0 - tr_proj, tr_st["H"] + tr_st["R"] * 0.34)
        ring = ellipse_pts(gx, gy, 2.5, 2.6, 14)
        pen.cv.erase(ring)
        pen.closed(ring, pen.width * 0.55, wobble=0.05)

    # 5. Central crossing steeple rising straight out of the cruciform roof intersection!
    if crossing_tower:
        cs = s * 0.82
        cx = tr_x0 + tr_w * 0.5
        z_base = H + R * 0.32
        th = z_base + rng.uniform(26, 36)
        spire_h = rng.uniform(28, 38)
        _draw_bell_tower(pen, P, rng, cx - cs, cx + cs, ty - cs, ty + cs, z_base, th, spire_h, crenellated=False)
    return pl


def well(pen, P, X, Y):
    W0 = pen.width
    cylinder(pen, P, X, Y, 4.6, 0, 4.2)
    cx, cy = P(X, Y, 4.2)
    pen.blot(ellipse_pts(cx, cy, 3.1, 3.1 * P.se, 14), jitter=0.05)
    for dx in (-4.1, 4.1):
        pen.beam(P(X + dx, Y, 4.0), P(X + dx, Y, 12.2), 1.2)
    # horizontal windlass bar
    pen.stroke([P(X - 4.1, Y, 8.2), P(X + 4.1, Y, 8.2)], W0 * 0.5, taper=(0.05, 0.05), wobble=0)
    pyramid(pen, P, X - 6.2, Y - 4.2, X + 6.2, Y + 4.2, 12.0, 16.5, shade=True)


def fence(pen, P, A, B, h=4.0):
    W0 = pen.width
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = max(1, int(np.hypot(*(B - A)) / 6))
    tops, mids = [], []
    for i in range(n + 1):
        g = A + (B - A) * i / n
        base, top = P(g[0], g[1], 0), P(g[0], g[1], h)
        pen.stroke([base, top], W0 * 0.58, taper=(0, 0.1), wobble=0)
        tops.append(P(g[0], g[1], h * 0.85))
        mids.append(P(g[0], g[1], h * 0.4))
    pen.stroke(tops, W0 * 0.44, taper=(0.05, 0.05), wobble=0.18)
    pen.stroke(mids, W0 * 0.44, taper=(0.05, 0.05), wobble=0.18)


# --------------------------------------------------------------------------- layout

def _overlaps(box_, boxes, gap):
    x0, y0, x1, y1 = box_
    return any(not (x1 + gap < b[0] or x0 - gap > b[2] or y1 + gap < b[1] or y0 - gap > b[3]) for b in boxes)


def _render(pen, P, items):
    """items: (box, (kind, args)). Exact 3D box topological sort, farther first."""
    rng = pen.rng
    for kind, args in sort_boxes(P, items):
        if kind == "house":
            draw_house(pen, P, *args)
        elif kind == "chapel":
            chapel(pen, P, args[0], args[1], rng, args[2])
        elif kind == "tree":
            x, y = P(*args)
            tree(pen, x, y)
        elif kind == "well":
            well(pen, P, *args)
        elif kind == "fence":
            fence(pen, P, *args)
        elif kind == "tuft":
            pen.tuft(*P(*args), rng.uniform(2.8, 4.6))


def _house_item(pl, X0, Y0):
    return (footprint(pl, X0, Y0), ("house", (X0, Y0, pl)))


def _trees_around(rng, items, boxes, n, region):
    (xa, xb), (ya, yb) = region
    for _ in range(n):
        for _ in range(40):
            x, y = rng.uniform(xa, xb), rng.uniform(ya, yb)
            tbox = (x - 4.5, y - 4.5, x + 4.5, y + 4.5)
            if not _overlaps(tbox, boxes, 1.5):
                items.append((tbox, ("tree", (x, y))))
                boxes.append(tbox)
                break


def _tufts(rng, items, host_boxes, all_boxes=None):
    check = all_boxes if all_boxes is not None else host_boxes
    for b in host_boxes[:]:
        for _ in range(rng.integers(0, 2)):
            x = rng.uniform(b[0] + 2, max(b[0] + 2.5, b[2] - 2))
            y = b[1] - rng.uniform(1.8, 3.6)
            tbox = (x - 1.5, y - 1.5, x + 1.5, y + 1.5)
            if not _overlaps(tbox, check, 0.2):
                items.append((tbox, ("tuft", (x, y))))


def single_house(pen, P, rng):
    pl = plan_house(rng)
    X0, Y0 = -pl["W"] / 2, -pl["D"] / 2
    fp = footprint(pl, X0, Y0)
    items = [_house_item(pl, X0, Y0)]
    boxes = [fp]
    if rng.random() < 0.45:
        _trees_around(rng, items, boxes, int(rng.integers(1, 3)),
                      ((X0 - 12, X0 + pl["W"] + 12), (Y0 - 4, Y0 + pl["D"] + 9)))
    _tufts(rng, items, [fp], boxes)
    _render(pen, P, items)


def hamlet(pen, P, rng):
    """Compact 3-4 house hamlet in varied shapes: wide (E-W), tall (screen-vertical N-S), or balanced round."""
    shape = str(rng.choice(["wide", "tall", "round"], p=[0.34, 0.33, 0.33]))
    n = int(rng.integers(3, 5))
    items, boxes = [], []
    # Optional central well placed first on ~50% of round hamlets so houses ring naturally around it
    if shape == "round" and rng.random() < 0.55:
        wbox = (-6.5, -5.0, 6.5, 5.0)
        items.append((wbox, ("well", (0.0, 0.0))))
        boxes.append((-8.0, -7.0, 8.0, 7.0))
    for _ in range(360):
        if len(boxes) - (1 if any(it[1][0] == "well" for it in items) else 0) >= n:
            break
        pl = plan_house(rng, str(rng.choice(["cottage", "house", "long", "barn"], p=[0.45, 0.35, 0.1, 0.1])))
        if shape == "wide":
            cx = rng.normal(0, 26)
            cy = rng.normal(0, 11)
        elif shape == "tall":
            # Along x = -0.44 * y so the hamlet runs cleanly top-to-bottom on screen (narrow left-to-right!)
            cy = rng.normal(0, 32)
            cx = -0.44 * cy + rng.normal(0, 12)
        else:
            # Balanced circular/equilibrated yard cluster
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(20, 42)
            cx = rad * math.cos(ang) * 0.92 - 0.15 * rad * math.sin(ang)
            cy = rad * math.sin(ang) * 1.05
        X0 = cx - pl["W"] / 2
        Y0 = cy - pl["D"] / 2
        fp = footprint(pl, X0, Y0)
        if not _overlaps(fp, boxes, rng.uniform(2, 5)):
            boxes.append(fp)
            items.append(_house_item(pl, X0, Y0))
    bldg_boxes = list(boxes)
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    region = ((min(xs) - 10, max(xs) + 10), (min(ys) - 4, max(ys) + 8))
    if rng.random() < 0.65:
        _trees_around(rng, items, boxes, int(rng.integers(1, 3)), region)
    if not any(it[1][0] == "well" for it in items) and rng.random() < 0.35:
        for _ in range(30):
            x, y = rng.uniform(*region[0]), rng.uniform(*region[1])
            wbox = (x - 7, y - 5, x + 7, y + 5)
            if not _overlaps(wbox, boxes, 1):
                items.append((wbox, ("well", (x, y))))
                boxes.append(wbox)
                break
    if rng.random() < 0.35:
        b = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
        A, B = (b[0], b[1] - 5), (b[2], b[1] - 5)
        fbox = (A[0], A[1] - 1, B[0], A[1] + 1)
        if not _overlaps(fbox, boxes, 0):
            items.append((fbox, ("fence", (A, B))))
            boxes.append(fbox)
    _tufts(rng, items, bldg_boxes, boxes)
    _render(pen, P, items)


def _draw_road_strip(pen, P, rng, pts_world, half_w=3.2):
    """Draw a hand-inked double-edged dirt village road along world polyline `pts_world`."""
    W0 = pen.width
    pts_w = np.asarray(pts_world, float)
    if len(pts_w) < 3:
        return
    tang = np.gradient(pts_w, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    nrm /= np.maximum(1e-6, np.hypot(nrm[:, 0], nrm[:, 1]))[:, None]
    for sgn in (-1.0, 1.0):
        edge = [P(*(p + n * (sgn * half_w))) for p, n in zip(pts_w, nrm)]
        i = 0
        while i < len(edge) - 3:
            j = min(len(edge), i + int(rng.integers(10, 22)))
            pen.stroke(edge[i:j], W0 * 0.48, taper=(0.32, 0.32), wobble=0.35)
            i = j + int(rng.integers(1, 3))


def village(pen, P, rng):
    """6-12 house village in three distinct shapes: wide E-W road, tall N-S road, or round green/crossroads."""
    shape = str(rng.choice(["wide", "tall", "round"], p=[0.34, 0.33, 0.33]))
    road_boxes = []
    items, boxes = [], []
    road_pts_list = []

    if shape == "wide":
        L = rng.uniform(195, 245)
        xs = np.linspace(-L / 2, L / 2, 60)
        ph = rng.uniform(0, 2 * math.pi)
        ys = rng.uniform(8, 18) * np.sin(xs / L * math.pi * rng.uniform(1, 2) + ph)
        road_pts_list.append(np.stack([xs, ys], axis=1))
    elif shape == "tall":
        # Screen-vertical winding village lane (x ≈ -0.44 * y runs cleanly top-to-bottom!)
        L = rng.uniform(175, 215)
        ys = np.linspace(-L / 2, L / 2, 56)
        ph = rng.uniform(0, 2 * math.pi)
        xs = -0.44 * ys + rng.uniform(7, 15) * np.sin(ys / L * math.pi * rng.uniform(1, 1.8) + ph)
        road_pts_list.append(np.stack([xs, ys], axis=1))
    else:
        # Round / equilibrated village: either a crossroads (+) or a circular village green!
        round_mode = "green" if rng.random() < 0.5 else "cross"
        if round_mode == "cross":
            L1 = rng.uniform(140, 175)
            xs1 = np.linspace(-L1 / 2, L1 / 2, 42)
            ys1 = rng.uniform(4, 10) * np.sin(xs1 / L1 * math.pi + rng.uniform(0, 2 * math.pi))
            L2 = rng.uniform(135, 165)
            ys2 = np.linspace(-L2 / 2, L2 / 2, 42)
            xs2 = -0.44 * ys2 + rng.uniform(4, 10) * np.sin(ys2 / L2 * math.pi + rng.uniform(0, 2 * math.pi))
            road_pts_list.append(np.stack([xs1, ys1], axis=1))
            road_pts_list.append(np.stack([xs2, ys2], axis=1))
        else:
            rx_g, ry_g = rng.uniform(32, 42), rng.uniform(36, 46)
            ts_g = np.linspace(0, 2 * math.pi, 48)
            gx_r = rx_g * np.cos(ts_g) - 0.18 * ry_g * np.sin(ts_g)
            gy_r = ry_g * np.sin(ts_g)
            road_pts_list.append(np.stack([gx_r, gy_r], axis=1))
            # Two short entry lanes leaving the village green
            road_pts_list.append(np.stack([np.linspace(rx_g, rx_g + 42, 14), np.zeros(14)], axis=1))
            road_pts_list.append(np.stack([np.linspace(-rx_g - 38, -rx_g, 14), np.zeros(14)], axis=1))

    for rpts in road_pts_list:
        _draw_road_strip(pen, P, rng, rpts)
        for x, y in rpts:
            road_boxes.append((x - 2.5, y - 4.5, x + 2.5, y + 4.5))

    # Optional village parish church landmark (~64% of villages)
    if rng.random() < 0.64:
        for _ in range(80):
            pl = plan_chapel(rng)
            if shape == "wide":
                rpts = road_pts_list[0]
                x = rng.uniform(-90, 80)
                side = 1 if rng.random() < 0.88 else -1
                X0 = x - pl["W"] / 2
                fp_try = chapel_footprint(pl, X0, 0.0)
                seg = (rpts[:, 0] >= fp_try[0]) & (rpts[:, 0] <= fp_try[2])
                yr_edge = float(rpts[seg, 1].max() if side > 0 else rpts[seg, 1].min()) if seg.any() else 0.0
                Y0 = yr_edge + 14.0 + pl["tr_proj"] if side > 0 else yr_edge - 14.0 - pl["D"] - pl["tr_proj"]
            elif shape == "tall":
                rpts = road_pts_list[0]
                yc = rng.uniform(-25, 55)
                xc_road = float(np.interp(yc, rpts[:, 1], rpts[:, 0]))
                side = int(rng.choice([-1, 1]))
                Y0 = yc - pl["D"] / 2
                X0 = xc_road + 14.0 + (0.0 if pl["crossing_tower"] else pl["tower_s"] * 1.75) if side > 0 else xc_road - 14.0 - pl["W"]
            else:
                # Overlooking the north/upper edge of the village green or north-west crossroads quadrant
                X0 = rng.uniform(-38, 12) - pl["W"] * 0.4
                Y0 = rng.uniform(22, 56)
            fp = chapel_footprint(pl, X0, Y0)
            if not _overlaps(fp, boxes + road_boxes, 2.5):
                boxes.append(fp)
                items.append((fp, ("chapel", (X0, Y0, pl))))
                break

    # Center well on round village greens (so the green feels alive!)
    if shape == "round" and len(road_pts_list) == 3:
        wbox = (-6.5, -5.0, 6.5, 5.0)
        if not _overlaps((-8, -6, 8, 6), boxes + road_boxes, 1):
            items.append((wbox, ("well", (0.0, 0.0))))
            boxes.append((-8, -6, 8, 6))

    target = int(rng.integers(7, 13))
    for _ in range(750):
        if len(items) >= target:
            break
        pl = plan_house(rng)
        if shape == "wide":
            rpts = road_pts_list[0]
            x = rng.uniform(rpts[0, 0] + 10, rpts[-1, 0] - 10)
            yr = float(np.interp(x, rpts[:, 0], rpts[:, 1]))
            side = rng.choice([-1, 1])
            setback = rng.uniform(5, 12) + (rng.uniform(0, 26) if rng.random() < 0.3 else 0)
            X0 = x - pl["W"] / 2
            Y0 = yr + setback if side > 0 else yr - setback - pl["D"]
        elif shape == "tall":
            rpts = road_pts_list[0]
            y = rng.uniform(rpts[0, 1] + 10, rpts[-1, 1] - 10)
            xr = float(np.interp(y, rpts[:, 1], rpts[:, 0]))
            side = rng.choice([-1, 1])
            setback = rng.uniform(6, 13) + (rng.uniform(0, 22) if rng.random() < 0.28 else 0)
            X0 = xr + setback if side > 0 else xr - setback - pl["W"]
            Y0 = y - pl["D"] / 2
        else:
            # Ring around the central plaza / crossroads evenly in all directions
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(34, 86)
            cx = rad * math.cos(ang) * 0.94 - 0.22 * rad * math.sin(ang)
            cy = rad * math.sin(ang) * 1.02
            X0 = cx - pl["W"] / 2
            Y0 = cy - pl["D"] / 2
        fp = footprint(pl, X0, Y0)
        if not _overlaps(fp, boxes + road_boxes, rng.uniform(2.5, 6)):
            boxes.append(fp)
            items.append(_house_item(pl, X0, Y0))

    bldg_boxes = list(boxes)
    if not any(it[1][0] == "well" for it in items) and rng.random() < 0.65:
        for _ in range(40):
            rpts = road_pts_list[0]
            idx = int(rng.integers(len(rpts) // 4, max(len(rpts) // 4 + 1, 3 * len(rpts) // 4)))
            x = float(rpts[idx, 0]) + rng.choice([-1, 1]) * 10
            y = float(rpts[idx, 1]) + rng.choice([-1, 1]) * 10
            wbox = (x - 7, y - 5, x + 7, y + 5)
            if not _overlaps(wbox, boxes + road_boxes, 1):
                items.append((wbox, ("well", (x, y))))
                boxes.append(wbox)
                break
    for _ in range(int(rng.integers(1, 4))):  # garden fences
        b = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
        A, B = (b[0], b[1] - 6), (b[2], b[1] - 6)
        fbox = (A[0], A[1] - 1, B[0], A[1] + 1)
        if not _overlaps(fbox, boxes + road_boxes, 0):
            items.append((fbox, ("fence", (A, B))))
            boxes.append(fbox)
    bx_all = [b[0] for b in boxes] + [b[2] for b in boxes]
    by_all = [b[1] for b in boxes] + [b[3] for b in boxes]
    occ_boxes = list(boxes) + list(road_boxes)
    if shape == "tall":
        # Keep trees hugging the screen-vertical lane instead of drifting into empty diagonal corners
        rpts = road_pts_list[0]
        for _ in range(int(rng.integers(3, 7))):
            for _ in range(40):
                y = rng.uniform(min(by_all) - 6, max(by_all) + 10)
                xr = float(np.interp(y, rpts[:, 1], rpts[:, 0]))
                x = xr + rng.uniform(-52, 52)
                tbox = (x - 4.5, y - 4.5, x + 4.5, y + 4.5)
                if not _overlaps(tbox, occ_boxes, 1.5):
                    items.append((tbox, ("tree", (x, y))))
                    occ_boxes.append(tbox)
                    boxes.append(tbox)
                    break
    else:
        _trees_around(rng, items, occ_boxes, int(rng.integers(3, 7)),
                      ((min(bx_all) - 10, max(bx_all) + 10), (min(by_all) - 6, max(by_all) + 10)))
    _tufts(rng, items, bldg_boxes, occ_boxes)
    _render(pen, P, items)
