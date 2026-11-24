"""Intact settlements: single houses, hamlets and villages.

Houses are convex solids (walls + gable roof) drawn with back-face culling, so any
orientation shades correctly: east-facing walls/roofs are in shadow. Houses and hamlets
share one scale and crop tightly without stray clutter so they compose into towns.
"""
import math

import numpy as np

from .flora import tree
from .ink import chaikin, ellipse_pts
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

        rows = {0: [], 1: [H * 0.36], 2: [H * 0.16, H * 0.60]}[min(2, st["floors"])]
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
        ctop = H + R + rng.uniform(2.8, 5.8)
        box(pen, P, cx - 2.1, cy - 2.1, cx + 2.1, cy + 2.1, hb - 2.4, ctop)
        # small overhanging stone chimney cap rim
        box(pen, P, cx - 2.6, cy - 2.6, cx + 2.6, cy + 2.6, ctop, ctop + 1.3, shade=True)
        if pl.get("smoke"):
            # delicate short curling hearth-smoke wisp
            sx, sy = P(cx, cy, ctop + 2.2)
            pts = [(sx, sy), (sx + 2.5, sy - 4.0), (sx + 1.0, sy - 8.0), (sx + 4.5, sy - 11.5)]
            pen.stroke(pts, W0 * 0.42, taper=(0.35, 0.65), smooth=2, wobble=0.2)
    if pl["leanto"]:
        Wa = W * rng.uniform(0.22, 0.32)
        Hh = H * rng.uniform(0.75, 0.9)
        draw_faces(pen, P, leanto_faces(X0 + W, Y0 + D * 0.12, Y0 + D * 0.88, Wa, Hh, Hh - Wa * 0.45),
                   dict(pl, timber=False, dormer=False))


def chapel(pen, P, X0, Y0, rng):
    """Village church / chapel: stone nave with lancet windows, buttressed west bell-tower & cross spire."""
    W0 = pen.width
    pl = dict(plan_house(rng, "long"), ridge_x=True, chimney=False, leanto=False, timber=False,
              cross=False, roof="tile", plinth=True, quoins=True, arch_door=True, lancet=True, dormer=False)
    pl.update(H=rng.uniform(15, 19), floors=1, win=(2.6, 6.8))
    s = rng.uniform(5.4, 6.8)
    ty = Y0 + pl["D"] / 2
    th = rng.uniform(40, 52)
    tx0, tx1 = X0 - s * 1.65, X0 + s * 0.35
    tx = (tx0 + tx1) * 0.5
    # Stepped corner buttresses + bell tower shaft
    box(pen, P, tx0 - 1.2, ty - s - 1.2, tx1 + 1.2, ty + s + 1.2, 0, th * 0.28)
    box(pen, P, tx0, ty - s, tx1, ty + s, 0, th)
    # Stringcourse molding under belfry + pointed lancet & belfry openings
    pen.stroke([P(tx0, ty - s, th * 0.66), P(tx1, ty - s, th * 0.66)], W0 * 0.55, taper=(0.05, 0.05))
    for h in (th * 0.40, th * 0.74):
        pen.blot([P(tx - 1.2, ty - s, h), P(tx + 1.2, ty - s, h), P(tx + 1.2, ty - s, h + 5.2),
                  P(tx, ty - s, h + 6.8), P(tx - 1.2, ty - s, h + 5.2)], jitter=0.05)
    # Tall shingled steeple + small cross finial
    apex_h = th + rng.uniform(22, 30)
    pyramid(pen, P, tx0 - 1.0, ty - s - 1.0, tx1 + 1.0, ty + s + 1.0, th, apex_h)
    ax, ay = P(tx, ty, apex_h)
    pen.stroke([(ax, ay), (ax, ay - 6.5)], W0 * 0.62, taper=(0.0, 0.15), wobble=0)
    pen.stroke([(ax - 2.3, ay - 4.4), (ax + 2.3, ay - 4.4)], W0 * 0.55, taper=(0.1, 0.1), wobble=0)
    draw_house(pen, P, X0, Y0, pl)
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
    """items: (depth, kind, args). Farther first."""
    rng = pen.rng
    for _, kind, args in sorted(items, key=lambda it: -it[0]):
        if kind == "house":
            draw_house(pen, P, *args)
        elif kind == "chapel":
            chapel(pen, P, *args, rng)
        elif kind == "tree":
            x, y = P(*args)
            tree(pen, x, y)
        elif kind == "well":
            well(pen, P, *args)
        elif kind == "fence":
            fence(pen, P, *args)
        elif kind == "tuft":
            pen.tuft(*P(*args), rng.uniform(2.8, 4.6))


def _house_item(P, pl, X0, Y0):
    return (P.depth(X0 + pl["W"], Y0), "house", (X0, Y0, pl))


def _trees_around(rng, P, items, boxes, n, region):
    (xa, xb), (ya, yb) = region
    for _ in range(n):
        for _ in range(40):
            x, y = rng.uniform(xa, xb), rng.uniform(ya, yb)
            if not _overlaps((x - 4, y - 4, x + 4, y + 4), boxes, 1):
                items.append((P.depth(x, y), "tree", (x, y)))
                boxes.append((x - 4, y - 4, x + 4, y + 4))
                break


def _tufts(rng, P, items, boxes):
    for b in boxes[:]:
        for _ in range(rng.integers(0, 2)):
            x = rng.uniform(b[0] + 2, b[2] - 2)
            y = b[1] - rng.uniform(0.4, 2.2)
            items.append((P.depth(x, y), "tuft", (x, y)))


def single_house(pen, P, rng):
    pl = plan_house(rng)
    X0, Y0 = -pl["W"] / 2, -pl["D"] / 2
    items = [_house_item(P, pl, X0, Y0)]
    boxes = [footprint(pl, X0, Y0)]
    if rng.random() < 0.45:
        _trees_around(rng, P, items, boxes, int(rng.integers(1, 3)),
                      ((X0 - 12, X0 + pl["W"] + 12), (Y0 - 3, Y0 + pl["D"] + 9)))
    _tufts(rng, P, items, boxes[:1])
    _render(pen, P, items)


def hamlet(pen, P, rng):
    n = int(rng.integers(3, 5))
    items, boxes = [], []
    for _ in range(300):
        if len(boxes) >= n:
            break
        pl = plan_house(rng, str(rng.choice(["cottage", "house", "long", "barn"], p=[0.45, 0.35, 0.1, 0.1])))
        X0 = rng.normal(0, 22) - pl["W"] / 2
        Y0 = rng.normal(0, 14) - pl["D"] / 2
        fp = footprint(pl, X0, Y0)
        if not _overlaps(fp, boxes, rng.uniform(2, 5)):
            boxes.append(fp)
            items.append(_house_item(P, pl, X0, Y0))
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    region = ((min(xs) - 10, max(xs) + 10), (min(ys) - 4, max(ys) + 8))
    if rng.random() < 0.65:
        _trees_around(rng, P, items, boxes, int(rng.integers(1, 3)), region)
    if rng.random() < 0.35:
        for _ in range(30):
            x, y = rng.uniform(*region[0]), rng.uniform(*region[1])
            if not _overlaps((x - 7, y - 5, x + 7, y + 5), boxes, 1):
                items.append((P.depth(x, y), "well", (x, y)))
                boxes.append((x - 7, y - 5, x + 7, y + 5))
                break
    if rng.random() < 0.35:
        b = boxes[int(rng.integers(0, len(boxes)))]
        A, B = (b[0], b[1] - 5), (b[2], b[1] - 5)
        if not _overlaps((A[0], A[1] - 1, B[0], A[1] + 1), boxes, 0):
            items.append((P.depth((A[0] + B[0]) / 2, A[1]), "fence", (A, B)))
    _tufts(rng, P, items, boxes)
    _render(pen, P, items)


def village(pen, P, rng):
    W0 = pen.width
    L = rng.uniform(200, 250)
    xs = np.linspace(-L / 2, L / 2, 60)
    ph = rng.uniform(0, 2 * math.pi)
    road = rng.uniform(8, 18) * np.sin(xs / L * math.pi * rng.uniform(1, 2) + ph)
    # Gently winding dirt road drawn at ground level
    for off in (-3.2, 3.2):
        pts = [P(x, y + off) for x, y in zip(xs, road)]
        i = 0
        while i < len(pts) - 3:
            j = min(len(pts), i + int(rng.integers(10, 22)))
            pen.stroke(pts[i:j], W0 * 0.48, taper=(0.32, 0.32), wobble=0.35)
            i = j + int(rng.integers(1, 3))
    road_boxes = [(x - 2, y - 4.5, x + 2, y + 4.5) for x, y in zip(xs, road)]
    items, boxes = [], []
    if rng.random() < 0.7:  # village chapel landmark
        for _ in range(60):
            pl = plan_house(rng, "long")
            x = rng.uniform(-L * 0.28, L * 0.28)
            side = 1 if rng.random() < 0.75 else -1
            X0 = x - pl["W"] / 2
            seg = (xs >= X0 - 16) & (xs <= X0 + pl["W"] + 2)
            yr_edge = float(road[seg].max() if side > 0 else road[seg].min()) if seg.any() else float(np.interp(x, xs, road))
            Y0 = yr_edge + 11.0 if side > 0 else yr_edge - 11.0 - pl["D"]
            fp = (X0 - 16, Y0 - 3, X0 + pl["W"] + 2, Y0 + pl["D"] + 2)
            if not _overlaps(fp, boxes + road_boxes, 2):
                boxes.append(fp)
                items.append((P.depth(X0 + pl["W"], Y0), "chapel", (X0, Y0)))
                break
    target = int(rng.integers(7, 13))
    for _ in range(600):
        if len(items) >= target:
            break
        pl = plan_house(rng)
        x = rng.uniform(-L / 2 + 10, L / 2 - 10)
        yr = float(np.interp(x, xs, road))
        side = rng.choice([-1, 1])
        setback = rng.uniform(5, 12) + (rng.uniform(0, 28) if rng.random() < 0.3 else 0)
        X0 = x - pl["W"] / 2
        Y0 = yr + setback if side > 0 else yr - setback - pl["D"]
        fp = footprint(pl, X0, Y0)
        if not _overlaps(fp, boxes + road_boxes, rng.uniform(2.5, 6)):
            boxes.append(fp)
            items.append(_house_item(P, pl, X0, Y0))
    if rng.random() < 0.65:
        for _ in range(40):
            x = rng.uniform(-L * 0.35, L * 0.35)
            y = float(np.interp(x, xs, road)) + rng.choice([-1, 1]) * 9
            if not _overlaps((x - 7, y - 5, x + 7, y + 5), boxes + road_boxes, 1):
                items.append((P.depth(x, y), "well", (x, y)))
                boxes.append((x - 7, y - 5, x + 7, y + 5))
                break
    for _ in range(int(rng.integers(1, 4))):  # garden fences
        b = boxes[int(rng.integers(0, len(boxes)))]
        A, B = (b[0], b[1] - 6), (b[2], b[1] - 6)
        if not _overlaps((A[0], A[1] - 1, B[0], A[1] + 1), boxes + road_boxes, 0):
            items.append((P.depth((A[0] + B[0]) / 2, A[1]), "fence", (A, B)))
    by = [b[1] for b in boxes] + [b[3] for b in boxes]
    _trees_around(rng, P, items, boxes + road_boxes, int(rng.integers(3, 7)),
                  ((-L / 2, L / 2), (min(by) - 6, max(by) + 10)))
    _tufts(rng, P, items, boxes)
    _render(pen, P, items)
