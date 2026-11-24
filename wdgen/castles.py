"""Dilapidated grand castles and tall spires."""
import math

import numpy as np

from .ruins import damage_profile, draw_house, draw_wall, plan_house, ruined_tower
from .solids import box, broken_cylinder, cone, cylinder, pyramid


def crenellate(prof, H, period=5.0, mw=2.6, mh=2.8):
    """Add merlons wherever the wall top is still intact at height H."""
    out = [prof[0]]
    for (s0, h0), (s1, h1) in zip(prof, prof[1:]):
        if abs(h0 - H) < 0.3 and abs(h1 - H) < 0.3 and s1 > s0:
            k = math.floor(s0 / period)
            while k * period < s1:
                ms, me = k * period + 0.6, k * period + 0.6 + mw
                if ms > s0 and me < s1:
                    out += [(ms, H), (ms, H + mh), (me, H + mh), (me, H)]
                k += 1
        out.append((s1, h1))
    return out


def curtain(pen, P, a, b, H, t, sev, crenel=True):
    """Free-standing wall between world points a and b (axis aligned)."""
    rng = pen.rng
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    if L < 2:
        return
    if rng.random() < 0.12 * (1 + sev):  # collapsed to a line of rubble
        for f in np.linspace(0.1, 0.9, max(2, int(L / 18))):
            x, y = P(ax + (bx - ax) * f, ay + (by - ay) * f)
            pen.rubble(x, y, rng.uniform(7, 12), rng.uniform(3, 5))
        return
    prof = damage_profile(rng, L, H, 0, False, sev)
    if crenel:
        prof = crenellate(prof, H)
    if abs(by - ay) < 1e-6:  # runs east-west: we see its south face
        draw_wall(pen, P, (ax, ay - t / 2), (bx, by - t / 2), prof, (0, t))
    else:  # runs north-south: we see its east face
        draw_wall(pen, P, (ax + t / 2, ay), (bx + t / 2, by), prof, (-t, 0), shade="dark")


def _square_tower(pen, P, X, Y, k, sev):
    pl = plan_house(pen.rng, k=k, keep=True, severity=sev)
    for w in pl["walls"]:
        pl["walls"][w] = crenellate(pl["walls"][w], pl["H"])
    draw_house(pen, P, X - pl["W"] / 2, Y - pl["D"] / 2, pl)


def castle(pen, P, rng):
    CW, CD = rng.uniform(120, 165), rng.uniform(85, 115)
    x0, y0, x1, y1 = -CW / 2, -CD / 2, CW / 2, CD / 2
    H, t, sev = rng.uniform(26, 34), rng.uniform(4, 5.5), rng.uniform(0.3, 0.85)
    round_towers = rng.random() < 0.6
    tk = rng.uniform(0.95, 1.15)

    def tower(X, Y):
        if round_towers:
            ruined_tower(pen, P, rng, X, Y, k=tk)
        else:
            _square_tower(pen, P, X, Y, tk * 0.95, rng.uniform(0.25, 0.9))

    def wall(a, b):
        curtain(pen, P, a, b, H, t, sev)

    def interior():
        items = []
        kp = plan_house(rng, k=rng.uniform(1.25, 1.5), keep=True, severity=rng.uniform(0.2, 0.7))
        for w in kp["walls"]:
            kp["walls"][w] = crenellate(kp["walls"][w], kp["H"])
        kx = rng.uniform(x0 + 18, x1 - 18 - kp["W"])
        ky = rng.uniform(y0 + 14, max(y0 + 15, y1 - 12 - kp["D"]))
        items.append((P.depth(kx + kp["W"] / 2, ky + kp["D"] / 2), "house", (kp, kx, ky)))
        if rng.random() < 0.75:  # great hall along a side wall
            hp = plan_house(rng, k=rng.uniform(0.8, 1.0))
            hx = x0 + 10 if kx > 0 else x1 - 10 - hp["W"]
            hy = rng.uniform(y0 + 14, max(y0 + 15, y1 - 12 - hp["D"]))
            if not (hx < kx + kp["W"] + 6 and hx + hp["W"] > kx - 6 and hy < ky + kp["D"] + 6 and hy + hp["D"] > ky - 6):
                items.append((P.depth(hx + hp["W"] / 2, hy + hp["D"] / 2), "house", (hp, hx, hy)))
        for _ in range(rng.integers(2, 6)):
            x, y = rng.uniform(x0 + 10, x1 - 10), rng.uniform(y0 + 8, y1 - 8)
            items.append((P.depth(x, y), str(rng.choice(["rubble", "tuft", "tuft"])), (x, y)))
        for _, kind, args in sorted(items, key=lambda it: -it[0]):
            if kind == "house":
                draw_house(pen, P, args[1], args[2], args[0])
            elif kind == "rubble":
                pen.rubble(*P(*args), rng.uniform(6, 11), rng.uniform(3, 4.5))
            else:
                pen.tuft(*P(*args), rng.uniform(3.5, 6))

    gw = rng.uniform(14, 20)
    gx = rng.uniform(-CW * 0.2, CW * 0.2)
    tower(x0, y1)
    wall((x0, y1), (x1, y1))
    tower(x1, y1)
    wall((x0, y0), (x0, y1))
    interior()
    wall((x1, y0), (x1, y1))
    tower(x0, y0)
    wall((x0, y0), (gx - gw / 2, y0))
    gk = rng.uniform(0.55, 0.7)
    for side in (-1, 1):  # gatehouse towers
        _square_tower(pen, P, gx + side * (gw / 2 + 8 * gk / 0.6), y0, gk, rng.uniform(0.2, 0.7))
    if rng.random() < 0.5:  # surviving arch over the gate
        hA = H * rng.uniform(0.75, 0.95)
        box(pen, P, gx - gw / 2, y0 - t / 2, gx + gw / 2, y0 + t / 2, hA, hA + 5, shade=True)
    wall((gx + gw / 2, y0), (x1, y0))
    tower(x1, y0)
    # outside: rubble and scrub
    for _ in range(rng.integers(3, 8)):
        x, y = rng.uniform(x0 - 10, x1 + 10), y0 - rng.uniform(10, 25)
        if rng.random() < 0.4:
            pen.rubble(*P(x, y), rng.uniform(6, 12), rng.uniform(3, 5))
        else:
            pen.tuft(*P(x, y), rng.uniform(3.5, 6))
    if rng.random() < 0.5:
        pen.bush(*P(rng.uniform(x0, x1), y0 - rng.uniform(12, 20)), rng.uniform(6, 10))


# --------------------------------------------------------------------------- spires

def _windows_on(pen, P, C, r, rng, h_lo, h_hi, n):
    for _ in range(n):
        t = P.az + 1.5 * math.pi + rng.uniform(-0.6, 0.6)
        h0 = rng.uniform(h_lo, h_hi)
        dt = 1.4 / r
        q = [C(t - dt, h0), C(t + dt, h0), C(t + dt, h0 + 6), C(t, h0 + 7.5), C(t - dt, h0 + 6)]
        pen.blot(q, jitter=0.1)


def spire(pen, P, rng, kind=None):
    W0 = pen.width
    a = P.az
    kind = kind or ("round" if rng.random() < 0.65 else "gothic")
    if kind == "round":
        r = rng.uniform(10, 14)
        H = rng.uniform(95, 140)
        base_h = 0.0
        if rng.random() < 0.5:  # wide lower drum with its own roof skirt
            R2, base_h = r * rng.uniform(1.5, 1.9), rng.uniform(18, 30)
            C2 = cylinder(pen, P, 0, 0, R2, 0, base_h, top=False)
            _windows_on(pen, P, C2, R2, rng, 5, base_h - 10, 2)
            cone(pen, P, 0, 0, R2 * 1.1, base_h, base_h + R2 * 0.9, broken=rng.uniform(0.35, 0.7) if rng.random() < 0.4 else 0)
        hb = base_h + (H - base_h) * rng.uniform(0.6, 0.8)
        C = cylinder(pen, P, 0, 0, r, base_h, hb, top=False)
        _windows_on(pen, P, C, r, rng, base_h + 8, hb - 10, int(rng.integers(1, 4)))
        cylinder(pen, P, 0, 0, r * 1.4, hb, hb + 2.5)  # balcony
        C = cylinder(pen, P, 0, 0, r, hb + 2.5, H, top=False)
        _windows_on(pen, P, C, r, rng, hb + 6, H - 9, int(rng.integers(1, 3)))
        state = rng.choice(["intact", "broken", "snapped"], p=[0.45, 0.4, 0.15])
        if state == "snapped":
            broken_cylinder(pen, P, 0, 0, r, H - 2, H + 6, jag=10)
        else:
            apex = H + r * rng.uniform(2.4, 3.6)
            cone(pen, P, 0, 0, r * 1.2, H, apex, broken=rng.uniform(0.3, 0.65) if state == "broken" else 0)
            if state == "intact":
                tip = P(0, 0, apex)
                pen.stroke([tip, (tip[0], tip[1] - 10)], W0 * 0.7, taper=(0, 0.3), wobble=0)
                pen.closed([(tip[0] + 1.6 * math.cos(t), tip[1] - 6 + 1.6 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 10, endpoint=False)],
                           W0 * 0.5, wobble=0)
        foot = r * 1.9 if base_h == 0 else r * 2.2
    else:  # square gothic spire
        s = rng.uniform(9, 13)
        H = rng.uniform(80, 115)
        box(pen, P, -s, -s, s, s, 0, H)
        for _ in range(int(rng.integers(2, 5))):  # lancet windows on the south face
            x = rng.uniform(-s * 0.6, s * 0.6)
            h0 = rng.uniform(10, H - 20)
            pen.blot([P(x - 1.4, -s, h0), P(x + 1.4, -s, h0), P(x + 1.4, -s, h0 + 8), P(x, -s, h0 + 10.5), P(x - 1.4, -s, h0 + 8)])
        for cx, cy in ((-s, -s), (s, -s), (s, s)):  # corner pinnacles (front ones)
            pyramid(pen, P, cx - 2, cy - 2, cx + 2, cy + 2, H, H + 9)
        if rng.random() < 0.65:
            pyramid(pen, P, -s * 0.85, -s * 0.85, s * 0.85, s * 0.85, H, H + s * rng.uniform(4, 6))
        else:  # spire fallen: stump and a dark open top
            pen.blot([P(-s + 2, -s + 2, H), P(s - 2, -s + 2, H), P(s - 2, s - 2, H), P(-s + 2, s - 2, H)])
        foot = s * 1.8
    for _ in range(rng.integers(3, 8)):
        t = rng.uniform(a + math.pi, a + 2 * math.pi)
        d = foot * rng.uniform(0.9, 1.5)
        x, y = P(d * math.cos(t), d * math.sin(t))
        if rng.random() < 0.35:
            pen.rubble(x, y, rng.uniform(5, 9), rng.uniform(2.5, 4))
        else:
            pen.tuft(x, y, rng.uniform(3.5, 6))
