"""Dilapidated grand castles and tall wizard/gothic spires."""
import math

import numpy as np

from .ink import TAU
from .lairs import crack
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
    draw_house(pen, P, X - pl["W"] / 2, Y - pl["D"] / 2, pl)


def _capped_turret(pen, P, rng, X, Y, k):
    """Intact or caved-in conical-roofed castle turret for iconic fortress variety."""
    r = rng.uniform(13, 17) * k
    H = rng.uniform(42, 58) * k
    C = cylinder(pen, P, X, Y, r, 0, H, top=False)
    _windows_on(pen, P, C, r, rng, 10, H - 10, int(rng.integers(1, 4)))
    broken = rng.uniform(0.35, 0.65) if rng.random() < 0.45 else 0.0
    cone(pen, P, X, Y, r * 1.18, H, H + r * rng.uniform(1.8, 2.5), broken=broken)


def castle(pen, P, rng):
    """Grand ruined fortress: corner towers, twin-towered arched gatehouse, curtain walls, central keep."""
    W0 = pen.width
    CW, CD = rng.uniform(125, 165), rng.uniform(88, 118)
    x0, y0, x1, y1 = -CW / 2, -CD / 2, CW / 2, CD / 2
    H, t, sev = rng.uniform(26, 34), rng.uniform(4.2, 5.5), rng.uniform(0.28, 0.80)
    round_towers = rng.random() < 0.65
    tk = rng.uniform(0.92, 1.12)

    def tower(X, Y, allow_cap=True):
        if round_towers:
            if allow_cap and rng.random() < 0.30:
                _capped_turret(pen, P, rng, X, Y, tk)
            else:
                ruined_tower(pen, P, rng, X, Y, k=tk)
        else:
            _square_tower(pen, P, X, Y, tk * 0.92, rng.uniform(0.22, 0.82))

    def wall(a, b):
        curtain(pen, P, a, b, H, t, sev)

    def interior():
        items = []
        kp = plan_house(rng, k=rng.uniform(1.22, 1.48), keep=True, severity=rng.uniform(0.2, 0.65))
        for w in kp["walls"]:
            kp["walls"][w] = crenellate(kp["walls"][w], kp["H"])
        kx = rng.uniform(x0 + 20, x1 - 20 - kp["W"])
        ky = rng.uniform(y0 + 16, max(y0 + 17, y1 - 14 - kp["D"]))
        items.append((P.depth(kx + kp["W"] / 2, ky + kp["D"] / 2), "house", (kp, kx, ky)))
        if rng.random() < 0.75:  # ruined great hall along a bailey wall
            hp = plan_house(rng, k=rng.uniform(0.78, 0.98))
            hx = x0 + 11 if kx > 0 else x1 - 11 - hp["W"]
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

    gw = rng.uniform(15, 21)
    gx = rng.uniform(-CW * 0.18, CW * 0.18)
    tower(x0, y1)
    wall((x0, y1), (x1, y1))
    tower(x1, y1)
    wall((x0, y0), (x0, y1))
    interior()
    wall((x1, y0), (x1, y1))
    tower(x0, y0)
    wall((x0, y0), (gx - gw / 2 - 6, y0))

    # Twin-towered arched gatehouse with portcullis shadow
    gk = rng.uniform(0.55, 0.68)
    g_off = gw / 2 + 8.5 * gk / 0.6
    _square_tower(pen, P, gx - g_off, y0, gk, rng.uniform(0.18, 0.6))
    hA = H * rng.uniform(0.75, 0.92)
    # Recessed dark arched gate portal & surviving battlemented gatehouse arch bridge
    gate_arch = [P(gx - gw * 0.42, y0 - t * 0.4, 0), P(gx - gw * 0.42, y0 - t * 0.4, hA * 0.62),
                 P(gx, y0 - t * 0.4, hA * 0.95), P(gx + gw * 0.42, y0 - t * 0.4, hA * 0.62),
                 P(gx + gw * 0.42, y0 - t * 0.4, 0)]
    pen.blot(gate_arch, jitter=0.1)
    # Portcullis iron spikes along top of arch
    for px in np.linspace(gx - gw * 0.3, gx + gw * 0.3, 4):
        p0 = P(px, y0 - t * 0.4, hA * 0.68)
        pen.cv.erase([(p0[0] - 0.5, p0[1]), (p0[0] + 0.5, p0[1]), (p0[0], p0[1] + 2.6)])
    if rng.random() < 0.75:
        box(pen, P, gx - gw / 2 - 2, y0 - t / 2, gx + gw / 2 + 2, y0 + t / 2, hA * 0.88, hA + 5.5, shade=True)
    _square_tower(pen, P, gx + g_off, y0, gk, rng.uniform(0.18, 0.6))

    wall((gx + gw / 2 + 6, y0), (x1, y0))
    tower(x1, y0)

    # Outside rubble, shrubs & scrub
    for _ in range(rng.integers(3, 8)):
        x, y = rng.uniform(x0 - 10, x1 + 10), y0 - rng.uniform(10, 24)
        if rng.random() < 0.42:
            pen.rubble(*P(x, y), rng.uniform(6, 12), rng.uniform(3, 5))
        else:
            pen.tuft(*P(x, y), rng.uniform(3.5, 6))
    if rng.random() < 0.55:
        pen.bush(*P(rng.uniform(x0, x1), y0 - rng.uniform(12, 20)), rng.uniform(6, 9.5), outline=True)


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
        foot = r * 1.9 if base_h == 0 else r * 2.2

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
        foot = s * 1.95

    for _ in range(rng.integers(3, 8)):
        t = rng.uniform(a + math.pi, a + TAU)
        d = foot * rng.uniform(0.95, 1.5)
        x, y = P(d * math.cos(t), d * math.sin(t))
        if rng.random() < 0.35:
            pen.rubble(x, y, rng.uniform(5, 9), rng.uniform(2.5, 4))
        else:
            pen.tuft(x, y, rng.uniform(3.5, 6))
