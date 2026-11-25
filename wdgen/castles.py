"""Dilapidated grand castles and tall wizard/gothic spires."""
import math

import numpy as np

from .ink import TAU
from .lairs import crack
from .proj import sort_boxes
from .ruins import _ivy, damage_profile, draw_house, draw_wall, plan_house, ruined_tower
from .solids import box, broken_cylinder, cone, cylinder, pyramid


def crenellate(prof, H, period=5.2, mw=2.6, mh=2.8):
    """Add merlons wherever the wall top is still intact at height H."""
    out = [prof[0]]
    for (s0, h0), (s1, h1) in zip(prof, prof[1:]):
        if abs(h0 - H) < 0.35 and abs(h1 - H) < 0.35 and s1 > s0:
            k = math.floor(s0 / period)
            while k * period < s1:
                ms, me = k * period + 0.6, k * period + 0.6 + mw
                if ms > s0 and me < s1:
                    out += [(ms, H), (ms, H + mh), (me, H + mh), (me, H)]
                k += 1
        out.append((s1, h1))
    return out


def curtain(pen, P, a, b, H, t, sev, crenel=True):
    """Free-standing ruined curtain wall between world points a and b (axis aligned)."""
    rng = pen.rng
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    if L < 2:
        return
    if rng.random() < 0.12 * (1 + sev):
        for f in np.linspace(0.1, 0.9, max(2, int(L / 18))):
            x, y = P(ax + (bx - ax) * f, ay + (by - ay) * f)
            pen.rubble(x, y, rng.uniform(7, 12), rng.uniform(3, 5))
        return
    prof = damage_profile(rng, L, H, 0, False, sev)
    if crenel:
        prof = crenellate(prof, H)
    if abs(by - ay) < 1e-6:  # runs east-west: south face visible
        draw_wall(pen, P, (ax, ay - t / 2), (bx, by - t / 2), prof, (0, t))
    else:  # runs north-south: east face visible
        draw_wall(pen, P, (ax + t / 2, ay), (bx + t / 2, by), prof, (-t, 0), shade="dark")


def _square_tower(pen, P, X, Y, k, sev):
    pl = plan_house(pen.rng, k=k, keep=True, severity=sev)
    for w in pl["walls"]:
        pl["walls"][w] = crenellate(pl["walls"][w], pl["H"])
    draw_house(pen, P, X - pl["W"] / 2, Y - pl["D"] / 2, pl, exterior=False)


def _capped_turret(pen, P, rng, X, Y, k):
    """Intact or caved-in conical-roofed castle turret for iconic fortress variety."""
    r = rng.uniform(13, 17) * k
    H = rng.uniform(42, 58) * k
    C = cylinder(pen, P, X, Y, r, 0, H, top=False)
    _windows_on(pen, P, C, r, rng, 10, H - 10, int(rng.integers(1, 4)))
    broken = rng.uniform(0.35, 0.65) if rng.random() < 0.45 else 0.0
    cone(pen, P, X, Y, r * 1.18, H, H + r * rng.uniform(1.8, 2.5), broken=broken)


def _curtain_seg(pen, P, a, b, H, t, sev, show_inner=False, crenel=True):
    """Arbitrary 2D ruined curtain wall segment `a -> b` around origin (0, 0), shaded cleanly for az=28°."""
    rng = pen.rng
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    if L < 3.0:
        return
    if rng.random() < 0.10 * (1 + sev):
        for f in np.linspace(0.15, 0.85, max(2, int(L / 18))):
            pen.rubble(*P(*(a + (b - a) * f)), rng.uniform(7, 12), rng.uniform(3, 5))
        return
    u = (b - a) / L
    n_in = np.array([-u[1], u[0]])
    if float(n_in @ (-(a + b) * 0.5)) < 0:
        n_in = -n_in
    n_out = -n_in
    prof = damage_profile(rng, L, H, 0, False, sev)
    if crenel:
        prof = crenellate(prof, H)
    if show_inner:
        shade = "dark" if n_in[0] > 0.35 else "light"
        draw_wall(pen, P, a + n_in * t, b + n_in * t, prof, -n_in * t, shade=shade)
    else:
        shade = "dark" if n_out[0] > 0.35 else None
        draw_wall(pen, P, a, b, prof, n_in * t, shade=shade)


def castle(pen, P, rng):
    """Grand ruined fortress in three varied shapes: wide (E-W), tall (deep N-S), or round/square citadel."""
    shape = str(rng.choice(["wide", "tall", "round"], p=[0.34, 0.33, 0.33]))
    H, t, sev = rng.uniform(26, 34), rng.uniform(4.2, 5.5), rng.uniform(0.28, 0.78)
    round_towers = rng.random() < 0.68
    tk = rng.uniform(0.90, 1.10)

    def tower(X, Y, allow_cap=True):
        if round_towers:
            if allow_cap and rng.random() < 0.32:
                _capped_turret(pen, P, rng, X, Y, tk)
            else:
                ruined_tower(pen, P, rng, X, Y, k=tk, exterior=False)
        else:
            _square_tower(pen, P, X, Y, tk * 0.92, rng.uniform(0.22, 0.80))

    def gatehouse(gx, y_front, gw, gk_range=(0.54, 0.66)):
        gk = rng.uniform(*gk_range)
        g_off = gw / 2 + 8.2 * gk / 0.6
        _square_tower(pen, P, gx - g_off, y_front, gk, rng.uniform(0.18, 0.58))
        hA = H * rng.uniform(0.75, 0.92)
        gate_arch = [P(gx - gw * 0.42, y_front - t * 0.4, 0), P(gx - gw * 0.42, y_front - t * 0.4, hA * 0.62),
                     P(gx, y_front - t * 0.4, hA * 0.95), P(gx + gw * 0.42, y_front - t * 0.4, hA * 0.62),
                     P(gx + gw * 0.42, y_front - t * 0.4, 0)]
        pen.blot(gate_arch, jitter=0.1)
        for px in np.linspace(gx - gw * 0.3, gx + gw * 0.3, 4):
            p0 = P(px, y_front - t * 0.4, hA * 0.68)
            pen.cv.erase([(p0[0] - 0.5, p0[1]), (p0[0] + 0.5, p0[1]), (p0[0], p0[1] + 2.6)])
        if rng.random() < 0.75:
            box(pen, P, gx - gw / 2 - 2, y_front - t / 2, gx + gw / 2 + 2, y_front + t / 2, hA * 0.88, hA + 5.5, shade=True)
        _square_tower(pen, P, gx + g_off, y_front, gk, rng.uniform(0.18, 0.58))

    # Hexagonal circular ring-wall citadel (~half of "round" castles!)
    if shape == "round" and rng.random() < 0.55:
        R = rng.uniform(76, 86)
        tk = rng.uniform(0.78, 0.92)
        # 6 vertices oriented with a flat south gate wall (v0 -> v1)
        vs = [(R * math.cos(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3),
               R * math.sin(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3)) for k in range(6)]
        v0, v1, v2, v3, v4, v5 = vs  # SW, SE, E, NE, NW, W
        # Back-to-front ring wall & corner towers around central circular bailey
        tower(*v4)
        _curtain_seg(pen, P, v4, v3, H, t, sev, show_inner=True)
        _curtain_seg(pen, P, v5, v4, H, t, sev, show_inner=True)
        tower(*v3)
        tower(*v5)
        # Towering central keep inside the round ring citadel + sorted bailey debris
        kp = plan_house(rng, k=rng.uniform(1.24, 1.46), keep=True, severity=rng.uniform(0.20, 0.60))
        for w in kp["walls"]:
            kp["walls"][w] = crenellate(kp["walls"][w], kp["H"])
        kx = -kp["W"] / 2 + rng.uniform(-6, 6)
        ky = -kp["D"] / 2 + rng.uniform(-4, 8)
        k_fp = (kx - 3.0, ky - 3.0, kx + kp["W"] + 3.0, ky + kp["D"] + 3.0)
        hex_items = [(k_fp, ("keep", (kp, kx, ky)))]
        for _ in range(rng.integers(3, 6)):
            kind = "rubble" if rng.random() < 0.5 else "tuft"
            rad = rng.uniform(6, 10) if kind == "rubble" else 3.5
            for _ in range(20):
                ang = rng.uniform(0, TAU)
                rr = rng.uniform(R * 0.38, R * 0.68)
                px, py = rr * math.cos(ang), rr * math.sin(ang)
                if not (k_fp[0] - rad - 3 < px < k_fp[2] + rad + 3 and k_fp[1] - rad - 3 < py < k_fp[3] + rad + 3):
                    hex_items.append(((px - rad, py - rad * 0.5, px + rad, py + rad * 0.5), (kind, (px, py, rad))))
                    break
        for kind, args in sort_boxes(P, hex_items):
            if kind == "keep":
                draw_house(pen, P, args[1], args[2], args[0], exterior=False)
            elif kind == "rubble":
                pen.rubble(*P(args[0], args[1]), args[2], args[2] * 0.45)
            else:
                pen.tuft(*P(args[0], args[1]), rng.uniform(3.5, 5.5))
        _curtain_seg(pen, P, v3, v2, H, t, sev, show_inner=False)
        _curtain_seg(pen, P, v5, v0, H, t, sev, show_inner=False)
        tower(*v2)
        tower(*v0)
        _curtain_seg(pen, P, v2, v1, H, t, sev, show_inner=False)
        # South front curtain & arched gatehouse spanning v0 -> v1 with crisp tower clearance
        gw = rng.uniform(13, 16)
        y_f = v0[1]
        curtain(pen, P, v0, (-gw / 2 - 4, y_f), H, t, sev)
        gatehouse(0.0, y_f, gw, gk_range=(0.40, 0.48))
        curtain(pen, P, (gw / 2 + 4, y_f), v1, H, t, sev)
        tower(*v1)
        t_clr = 20.0 * tk + 4.0
        fg = []
        for _ in range(rng.integers(3, 7)):
            x, y = rng.uniform(-R * 0.55, R * 0.55), y_f - rng.uniform(t_clr, t_clr + 14.0)
            fg.append((P.depth(x, y), "rubble" if rng.random() < 0.42 else "tuft", (x, y)))
        if rng.random() < 0.50:
            bx, by = rng.uniform(-R * 0.45, R * 0.45), y_f - rng.uniform(t_clr + 3.0, t_clr + 12.0)
            fg.append((P.depth(bx, by), "bush", (bx, by)))
        for _, kind, (x, y) in sorted(fg, key=lambda it: -it[0]):
            if kind == "rubble":
                pen.rubble(*P(x, y), rng.uniform(6, 11), rng.uniform(3, 4.8))
            elif kind == "bush":
                pen.bush(*P(x, y), rng.uniform(5.5, 8.5), outline=True)
            else:
                pen.tuft(*P(x, y), rng.uniform(3.5, 6))
        return

    if shape == "wide":
        CW, CD = rng.uniform(132, 166), rng.uniform(82, 106)
    elif shape == "tall":
        CW, CD = rng.uniform(88, 110), rng.uniform(138, 172)
    else:  # balanced square castle
        CW = rng.uniform(112, 132)
        CD = CW * rng.uniform(0.95, 1.08)
    x0, y0, x1, y1 = -CW / 2, -CD / 2, CW / 2, CD / 2

    def wall(a, b):
        curtain(pen, P, a, b, H, t, sev)

    def interior():
        items = []
        b_fps = []
        kp = plan_house(rng, k=rng.uniform(1.18, 1.44), keep=True, severity=rng.uniform(0.2, 0.65))
        for w in kp["walls"]:
            kp["walls"][w] = crenellate(kp["walls"][w], kp["H"])
        kx = rng.uniform(x0 + 18, max(x0 + 19, x1 - 18 - kp["W"]))
        ky = rng.uniform(y0 + 16, max(y0 + 17, y1 - 16 - kp["D"]))
        k_fp = (kx - 3.0, ky - 3.0, kx + kp["W"] + 3.0, ky + kp["D"] + 3.0)
        b_fps.append(k_fp)
        items.append((k_fp, ("house", (kp, kx, ky))))
        if rng.random() < 0.75:  # ruined great hall along a bailey wall
            hp = plan_house(rng, k=rng.uniform(0.74, 0.94))
            if shape == "tall":
                hx = rng.uniform(x0 + 12, max(x0 + 13, x1 - 12 - hp["W"]))
                hy = y0 + 14 if ky > 0 else y1 - 14 - hp["D"]
            else:
                hx = x0 + 11 if kx > 0 else x1 - 11 - hp["W"]
                hy = rng.uniform(y0 + 14, max(y0 + 15, y1 - 12 - hp["D"]))
            if (x0 + 6 <= hx and hx + hp["W"] <= x1 - 6 and y0 + 6 <= hy and hy + hp["D"] <= y1 - 6
                    and not (hx < kx + kp["W"] + 6 and hx + hp["W"] > kx - 6 and hy < ky + kp["D"] + 6 and hy + hp["D"] > ky - 6)):
                h_fp = (hx, hy, hx + hp["W"], hy + hp["D"])
                b_fps.append(h_fp)
                items.append((h_fp, ("house", (hp, hx, hy))))
        for _ in range(rng.integers(2, 6)):
            kind = str(rng.choice(["rubble", "tuft", "tuft"]))
            rad = rng.uniform(6, 10) if kind == "rubble" else 3.5
            for _ in range(20):
                x, y = rng.uniform(x0 + 12, x1 - 12), rng.uniform(y0 + 10, y1 - 10)
                if all(not (fp[0] - rad - 3 < x < fp[2] + rad + 3 and fp[1] - rad - 3 < y < fp[3] + rad + 3) for fp in b_fps):
                    items.append(((x - rad, y - rad * 0.5, x + rad, y + rad * 0.5), (kind, (x, y, rad))))
                    break
        for kind, args in sort_boxes(P, items):
            if kind == "house":
                draw_house(pen, P, args[1], args[2], args[0], exterior=False)
            elif kind == "rubble":
                pen.rubble(*P(args[0], args[1]), args[2], args[2] * 0.45)
            else:
                pen.tuft(*P(args[0], args[1]), rng.uniform(3.5, 6))

    gw = rng.uniform(15, 19)
    gx = rng.uniform(-CW * 0.14, CW * 0.14) if CW > 105 else 0.0
    mid_towers = (shape == "tall" and rng.random() < 0.70)
    tower(x0, y1)
    wall((x0, y1), (x1, y1))
    tower(x1, y1)
    if mid_towers:
        wall((x0, 0), (x0, y1))
        tower(x0, 0, allow_cap=False)
        wall((x0, y0), (x0, 0))
    else:
        wall((x0, y0), (x0, y1))
    interior()
    if mid_towers:
        wall((x1, 0), (x1, y1))
        tower(x1, 0, allow_cap=False)
        wall((x1, y0), (x1, 0))
    else:
        wall((x1, y0), (x1, y1))
    tower(x0, y0)
    wall((x0, y0), (gx - gw / 2 - 6, y0))
    gatehouse(gx, y0, gw)
    wall((gx + gw / 2 + 6, y0), (x1, y0))
    tower(x1, y0)

    # Outside rubble, shrubs & scrub strictly clear of the front corner towers & gatehouse
    t_clr = 21.0 * tk + 4.0
    fg = []
    for _ in range(rng.integers(3, 8)):
        x, y = rng.uniform(x0 + 4, x1 - 4), y0 - rng.uniform(t_clr, t_clr + 14.0)
        fg.append((P.depth(x, y), "rubble" if rng.random() < 0.42 else "tuft", (x, y)))
    if rng.random() < 0.50:
        bx, by = rng.uniform(x0 + 8, x1 - 8), y0 - rng.uniform(t_clr + 3.0, t_clr + 12.0)
        fg.append((P.depth(bx, by), "bush", (bx, by)))
    for _, kind, (x, y) in sorted(fg, key=lambda it: -it[0]):
        if kind == "rubble":
            pen.rubble(*P(x, y), rng.uniform(6, 11), rng.uniform(3, 4.8))
        elif kind == "bush":
            pen.bush(*P(x, y), rng.uniform(5.5, 8.5), outline=True)
        else:
            pen.tuft(*P(x, y), rng.uniform(3.5, 6))


# --------------------------------------------------------------------------- spires

def _windows_on(pen, P, C, r, rng, h_lo, h_hi, n):
    W0 = pen.width
    for _ in range(n):
        t = P.az + 1.5 * math.pi + rng.uniform(-0.55, 0.55)
        h0 = rng.uniform(h_lo, h_hi)
        dt = 1.45 / r
        q = [C(t - dt, h0), C(t + dt, h0), C(t + dt, h0 + 5.8), C(t, h0 + 7.6), C(t - dt, h0 + 5.8)]
        pen.blot(q, jitter=0.08)
        # crisp stone window sill
        pen.stroke([C(t - dt * 1.35, h0 - 0.3), C(t + dt * 1.35, h0 - 0.3)], W0 * 0.45, taper=(0.1, 0.1), wobble=0)


def _corbel_balcony(pen, P, r, hb):
    """Architectural stone balcony ring with underside corbel brackets and parapet crenels."""
    W0 = pen.width
    a = P.az
    rb = r * 1.38
    # Underside corbel brackets fanning out from tower shaft to balcony rim
    for t in np.linspace(a + math.pi + 0.18, a + TAU - 0.18, 9):
        p0 = P(r * math.cos(t), r * math.sin(t), hb - 3.2)
        p1 = P(rb * math.cos(t), rb * math.sin(t), hb)
        pen.stroke([p0, p1], W0 * 0.65, taper=(0.2, 0.05), wobble=0.05)
    cylinder(pen, P, 0, 0, rb, hb, hb + 3.4)
    # Parapet crenellation notches along the front balcony rim
    for t in np.linspace(a + math.pi + 0.25, a + TAU - 0.25, 7):
        p0 = P(rb * math.cos(t), rb * math.sin(t), hb + 3.4)
        p1 = P(rb * math.cos(t), rb * math.sin(t), hb + 5.2)
        pen.stroke([p0, p1], W0 * 1.1, taper=(0, 0), wobble=0.05)


def spire(pen, P, rng, kind=None):
    """Tall fantasy wizard/cathedral spire (round corbelled tower OR buttressed gothic spire)."""
    W0 = pen.width
    a = P.az
    kind = kind or ("round" if rng.random() < 0.5 else "gothic")

    if kind == "round":
        r = rng.uniform(10.5, 14.0)
        H = rng.uniform(96, 138)
        base_h = 0.0
        if rng.random() < 0.55:  # wide lower drum with its own conical skirt
            R2, base_h = r * rng.uniform(1.5, 1.85), rng.uniform(18, 30)
            C2 = cylinder(pen, P, 0, 0, R2, 0, base_h, top=False)
            _windows_on(pen, P, C2, R2, rng, 5, base_h - 9, 2)
            # subtle masonry bands on drum
            for _ in range(3):
                t = a + 1.5 * math.pi + rng.uniform(-0.8, 0.5)
                hm = rng.uniform(4, base_h - 4)
                pen.stroke([C2(t, hm), C2(t + 0.3, hm)], W0 * 0.42, taper=(0.25, 0.25))
            cone(pen, P, 0, 0, R2 * 1.12, base_h, base_h + R2 * 0.88,
                 broken=rng.uniform(0.35, 0.68) if rng.random() < 0.35 else 0)
        else:
            # stepped circular stone plinth base
            cylinder(pen, P, 0, 0, r * 1.28, 0, 5.5)

        hb = base_h + (H - base_h) * rng.uniform(0.58, 0.76)
        C = cylinder(pen, P, 0, 0, r, base_h, hb, top=False)
        # horizontal stone stringcourse molding midway up the shaft
        hm_band = (base_h + hb) * 0.52
        pen.stroke([C(t, hm_band) for t in np.linspace(a + math.pi, a + TAU, 24)], W0 * 0.62, taper=(0.05, 0.05))
        _windows_on(pen, P, C, r, rng, base_h + 9, hb - 10, int(rng.integers(2, 4)))
        # subtle masonry joints & weathered crack
        for _ in range(rng.integers(3, 6)):
            t = a + 1.5 * math.pi + rng.uniform(-0.7, 0.45)
            h_m = rng.uniform(base_h + 6, hb - 6)
            pen.stroke([C(t, h_m), C(t + 3.8 / r, h_m)], W0 * 0.42, taper=(0.25, 0.25))
        if rng.random() < 0.45:
            x, y = C(a + 1.5 * math.pi + rng.uniform(-0.4, 0.3), rng.uniform(base_h + 12, hb - 8))
            crack(pen, x, y, math.pi * 0.5 + rng.normal(0, 0.4), rng.uniform(7, 14), W0 * 0.48, branch=False)

        _corbel_balcony(pen, P, r, hb)
        C_top = cylinder(pen, P, 0, 0, r, hb + 3.4, H, top=False)
        _windows_on(pen, P, C_top, r, rng, hb + 7, H - 9, int(rng.integers(1, 3)))

        state = str(rng.choice(["intact", "broken", "snapped"], p=[0.48, 0.37, 0.15]))
        if state == "snapped":
            broken_cylinder(pen, P, 0, 0, r, H - 2, H + 8, jag=10)
        else:
            apex = H + r * rng.uniform(2.5, 3.6)
            cone(pen, P, 0, 0, r * 1.22, H, apex, broken=rng.uniform(0.32, 0.65) if state == "broken" else 0)
            if state == "intact":
                tip = P(0, 0, apex)
                pen.stroke([tip, (tip[0], tip[1] - 11)], W0 * 0.7, taper=(0, 0.25), wobble=0)
                pen.closed([(tip[0] + 1.8 * math.cos(t), tip[1] - 6.5 + 1.8 * math.sin(t))
                            for t in np.linspace(0, TAU, 10, endpoint=False)], W0 * 0.5, wobble=0)
                if rng.random() < 0.45:  # tiny weathervane pennant
                    pen.stroke([(tip[0], tip[1] - 10), (tip[0] + 5.5, tip[1] - 8.8), (tip[0], tip[1] - 7.5)],
                               W0 * 0.5, taper=(0.1, 0.1), wobble=0.05)

        if rng.random() < 0.4:  # ivy creeping up base
            tt0 = a + 1.5 * math.pi - rng.uniform(0.2, 0.7)
            r_iv = r * (1.6 if base_h > 0 else 1.02)
            pts = [P(r_iv * math.cos(tt0 + math.sin(i * 0.8) * 0.08),
                     r_iv * math.sin(tt0 + math.sin(i * 0.8) * 0.08), i * 3.2) for i in range(6)]
            _ivy(pen, pts)
        base_rad = (R2 if base_h > 0 else r * 1.28)
        base_box = None

    else:  # True buttressed Gothic wizard/cathedral spire!
        s = rng.uniform(10.5, 13.8)
        H = rng.uniform(86, 120)
        # 1. Stepped stone gothic plinth base
        box(pen, P, -s * 1.48, -s * 1.48, s * 1.48, s * 1.48, 0, 5.5, inset=0.8)
        box(pen, P, -s * 1.24, -s * 1.24, s * 1.24, s * 1.24, 5.5, 12.0, inset=0.8)

        # 2. Main slender square stone tower shaft
        box(pen, P, -s, -s, s, s, 12.0, H)

        # 3. Tapered corner buttresses rising ~68% up the visible corners
        bh = H * rng.uniform(0.55, 0.72)
        bw = s * 0.34
        for cx, cy in ((-s, -s), (s, -s), (s, s)):
            # two-tier stepped gothic buttress
            box(pen, P, cx - bw, cy - bw, cx + bw, cy + bw, 0, bh * 0.52, inset=bw * 0.25)
            box(pen, P, cx - bw * 0.72, cy - bw * 0.72, cx + bw * 0.72, cy + bw * 0.72, bh * 0.52, bh, inset=bw * 0.3)
            pyramid(pen, P, cx - bw * 0.55, cy - bw * 0.55, cx + bw * 0.55, cy + bw * 0.55, bh, bh + 4.5)

        # 4. Horizontal stone stringcourses dividing the tower cleanly into 3 architectural stories
        stories = [12.0 + (H - 12.0) * f for f in (0.34, 0.68)]
        for hs in stories:
            pen.stroke([P(-s - 0.5, -s, hs), P(s + 0.5, -s, hs)], W0 * 0.72, taper=(0.05, 0.05), wobble=0.1)
            pen.stroke([P(s, -s - 0.4, hs), P(s, s + 0.4, hs)], W0 * 0.65, taper=(0.05, 0.05), wobble=0.1)

        # 5. Tall aligned pointed Gothic lancet windows + grand belfry paired lancets
        def lancet(x, h0, lw, lh):
            q = [P(x - lw, -s, h0), P(x + lw, -s, h0), P(x + lw, -s, h0 + lh * 0.76),
                 P(x, -s, h0 + lh), P(x - lw, -s, h0 + lh * 0.76)]
            pen.blot(q, jitter=0.05)
            pen.stroke([P(x - lw * 1.3, -s, h0 - 0.3), P(x + lw * 1.3, -s, h0 - 0.3)], W0 * 0.45, taper=(0.1, 0.1), wobble=0)

        # Ground-story arched portal
        lancet(0.0, 12.0, s * 0.24, 10.5)
        # Middle story single tall lancet
        lancet(0.0, stories[0] + 5.0, s * 0.17, (stories[1] - stories[0]) * 0.60)
        # Upper belfry story: twin paired gothic lancets!
        b_lo = stories[1] + 4.5
        b_h = (H - stories[1]) * 0.58
        for dx in (-s * 0.32, s * 0.32):
            lancet(dx, b_lo, s * 0.14, b_h)
        # East face narrow lancet silhouette on belfry
        pen.blot([P(s, -1.3, b_lo), P(s, 1.3, b_lo), P(s, 1.3, b_lo + b_h * 0.78),
                  P(s, 0, b_lo + b_h), P(s, -1.3, b_lo + b_h * 0.78)], jitter=0.05)

        # Subtle masonry ticks & weathered crack
        for _ in range(rng.integers(3, 6)):
            mx, mh = rng.uniform(-s * 0.75, s * 0.55), rng.uniform(16, H - 10)
            pen.stroke([P(mx, -s, mh), P(mx + 3.5, -s, mh)], W0 * 0.4, taper=(0.25, 0.25))
        if rng.random() < 0.45:
            x, y = P(rng.uniform(-s * 0.4, s * 0.4), -s, rng.uniform(22, H - 14))
            crack(pen, x, y, math.pi * 0.5 + rng.normal(0, 0.4), rng.uniform(7, 15), W0 * 0.46, branch=False)

        # 6. Corbelled gothic parapet overhang + 4 slender corner pinnacles
        sp = s * 1.16
        for kx in np.linspace(-s * 0.85, s * 0.85, 5):
            pen.stroke([P(kx, -s, H - 3.2), P(kx * 1.12, -sp, H)], W0 * 0.55, taper=(0.2, 0.05), wobble=0)
        box(pen, P, -sp, -sp, sp, sp, H, H + 4.5)
        hp = H + 4.5
        # Slender gothic corner pinnacles with delicate finials
        for cx, cy in ((-sp, sp), (-sp, -sp), (sp, -sp), (sp, sp)):
            ps = 2.0
            box(pen, P, cx - ps, cy - ps, cx + ps, cy + ps, hp - 1.5, hp + 4.5)
            pyramid(pen, P, cx - ps * 1.05, cy - ps * 1.05, cx + ps * 1.05, cy + ps * 1.05, hp + 4.5, hp + 14.5)

        # 7. Dramatic tall gothic spire OR ruined jagged crown
        if rng.random() < 0.70:
            apex = hp + s * rng.uniform(4.2, 5.8)
            pyramid(pen, P, -s * 0.92, -s * 0.92, s * 0.92, s * 0.92, hp, apex)
            # Slender spire cross / star finial
            tx, ty = P(0, 0, apex)
            pen.stroke([(tx, ty), (tx, ty - 9.5)], W0 * 0.65, taper=(0, 0.25), wobble=0)
            pen.stroke([(tx - 2.8, ty - 6.2), (tx + 2.8, ty - 6.2)], W0 * 0.52, taper=(0.15, 0.15), wobble=0)
        else:  # Ruined broken gothic crown (jagged crenellated stump instead of flat black box!)
            pen.blot([P(-s + 1.5, -s + 1.5, hp), P(s - 1.5, -s + 1.5, hp),
                      P(s - 1.5, s - 1.5, hp), P(-s + 1.5, s - 1.5, hp)], jitter=0.2)
            # Broken jagged masonry tooth rising on one back edge
            box(pen, P, -s, s - 3, 0, s, hp, hp + rng.uniform(6, 14), inset=0.8)

        if rng.random() < 0.42:  # ivy creeping up the plinth & buttress
            vx = -s * rng.uniform(0.3, 0.85)
            pts = [P(vx + math.sin(i * 0.9) * 1.2, -s * 1.25, i * 3.0) for i in range(7)]
            _ivy(pen, pts)
        base_rad = s * 2.15
        base_box = (-s * 1.62, -s * 1.62, s * 1.62, s * 1.62)

    fg = []
    for _ in range(rng.integers(3, 7)):
        is_rubble = rng.random() < 0.35
        rad = rng.uniform(5.0, 8.5) if is_rubble else 3.0
        for _ in range(18):
            t = rng.uniform(a + math.pi + 0.30, a + TAU - 0.30)
            d = base_rad + rad * 0.6 + rng.uniform(3.5, 13.0)
            wx, wy = d * math.cos(t), d * math.sin(t)
            if base_box is not None:
                if base_box[0] - rad - 2.5 < wx < base_box[2] + rad + 2.5 and base_box[1] - rad - 2.5 < wy < base_box[3] + rad + 2.5:
                    continue
            fg.append((P.depth(wx, wy), "rubble" if is_rubble else "tuft", (wx, wy, rad)))
            break
    for _, kind_p, (wx, wy, rad) in sorted(fg, key=lambda it: -it[0]):
        sx, sy = P(wx, wy)
        if kind_p == "rubble":
            pen.rubble(sx, sy, rad, rad * 0.45)
        else:
            pen.tuft(sx, sy, rng.uniform(3.5, 6))
