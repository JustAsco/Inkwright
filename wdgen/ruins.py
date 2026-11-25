"""Ruined buildings: cottages, square keeps, round towers, ruined villages."""
import math

import numpy as np

from .ink import TAU, clip_halfplane
from .proj import sort_boxes
from .solids import box, pyramid


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def prof_at(prof, s):
    return float(np.interp(s, [p[0] for p in prof], [p[1] for p in prof]))


def prof_min(prof, s0, s1):
    return min([prof_at(prof, s0), prof_at(prof, s1)] + [h for s, h in prof if s0 <= s <= s1])


def _ivy(pen, pts):
    """Delicate creeping ivy vine climbing a ruined masonry wall."""
    rng, W0 = pen.rng, pen.width
    if len(pts) < 3:
        return
    pen.stroke(pts, W0 * 0.52, taper=(0.05, 0.72), smooth=1, wobble=0.15)
    for k, q in enumerate(pts[1:-1]):
        sgn = 1 if k % 2 else -1
        pen.dab(q[0] + sgn * 1.5, q[1], rng.uniform(1.3, 1.9), angle=sgn * 0.4, w=W0 * 0.65)


# --------------------------------------------------------------------------- walls

def damage_profile(rng, L, H, R, gable, severity, protect=None):
    """Top edge of a ruined masonry wall as [(s, h), ...]: stepped stone courses without hairline spikes."""
    gable = gable and R > 0
    top = H + (R if gable else 0.0)

    def intact(s):
        return H + (R * (1 - np.abs(2 * s / L - 1)) if gable else 0.0)

    # Wider masonry step blocks (min ~4.2 units) so walls feel solid & stone-built, never paper spikes
    step_s = rng.uniform(4.2, 6.2)
    ss = np.linspace(0, L, max(4, int(L / step_s)) + 1)
    if gable:
        ss = np.union1d(ss, [L / 2])
    n = len(ss)
    cap = np.full(n, 1e9)
    n_breach = int(rng.integers(0, 2)) + (severity > 0.35) + (severity > 0.72)
    for _ in range(n_breach):
        c, hw = rng.uniform(-0.1 * L, 1.1 * L), rng.uniform(0.24, 0.56) * L
        floor = H * rng.uniform(0.10, max(0.18, 0.82 - 0.58 * severity))
        f = np.clip(1 - np.abs(ss - c) / hw, 0, 1) ** (1 / rng.uniform(1.2, 2.8))
        cap = np.minimum(cap, top + 2 - f * (top + 2 - floor))
    if severity > 0.78 and rng.random() < 0.45:
        cap = np.minimum(cap, H * rng.uniform(0.25, 0.58))
    if protect is not None:
        cap[(ss >= protect[0]) & (ss <= protect[1])] = 1e9
    vals = intact(ss)
    broken = cap < vals - 0.6
    course = rng.uniform(3.0, 4.2)
    hb = np.round((cap + rng.normal(0, 1.2, n)) / course) * course
    h = np.where(broken, np.clip(np.minimum(hb, vals), 2.5, None), vals)
    # Remove isolated single-block thin needles taller than both neighbors
    for i in range(1, n - 1):
        if broken[i] and h[i] > h[i - 1] + course * 1.4 and h[i] > h[i + 1] + course * 1.4:
            h[i] = max(h[i - 1], h[i + 1]) + course * 0.5
    pts = [(ss[0], h[0])]
    for i in range(1, n):
        if (broken[i] or broken[i - 1]) and abs(h[i] - h[i - 1]) > 0.5 and rng.random() < 0.6:
            pts.append((ss[i], h[i - 1]))
        pts.append((ss[i], h[i]))
    return pts


def draw_wall(pen, Q, a, b, prof, far, shade=None, openings=(), marks=True, arched=False):
    """One ruined masonry wall face from ground point a to b; `far` = thickness vector."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    a, b, far = np.asarray(a, float), np.asarray(b, float), np.asarray(far, float)
    L = float(np.hypot(*(b - a)))
    u = (b - a) / max(1e-6, L)

    def pt(s, h, f=False):
        g = a + u * s + (far if f else 0.0)
        return Q(g[0], g[1], h)

    rev = list(reversed(prof))
    face = [pt(0, 0), pt(L, 0)] + [pt(s, h) for s, h in rev]
    back = [pt(0, 0, True), pt(L, 0, True)] + [pt(s, h, True) for s, h in rev]
    cv.erase(back)
    cv.erase(face)
    # Top wall thickness rim + stepped block joints across thickness
    pen.stroke([pt(s, h, True) for s, h in prof], W0 * 0.65, taper=(0.03, 0.03), wobble=0.2)
    pen.line(pt(0, prof[0][1]), pt(0, prof[0][1], True), W0 * 0.65, sketch=0, overshoot=0.25)
    pen.line(pt(L, prof[-1][1]), pt(L, prof[-1][1], True), W0 * 0.65, sketch=0, overshoot=0.25)
    for (s0, h0), (s1, h1) in zip(prof, prof[1:]):
        if abs(h1 - h0) > 2.2:
            pen.stroke([pt(s1, max(h0, h1)), pt(s1, max(h0, h1), True)], W0 * 0.45, taper=(0.05, 0.05), wobble=0.1)

    if shade == "dark":
        pen.hatch(face, 80 + rng.normal(0, 3), rng.uniform(1.6, 2.0), W0 * 0.72, coverage=(0.75, 1.0))
    elif shade == "light":
        pen.hatch(face, 80, rng.uniform(3.0, 4.0), W0 * 0.45, coverage=(0.25, 0.6))

    for (s0, s1, h0, h1) in openings:
        if prof_min(prof, s0 - 1, s1 + 1) > h1 + 2.2:
            sm = (s0 + s1) * 0.5
            if arched and (s1 - s0) > 3.2:
                q = [pt(s0, h0), pt(s1, h0), pt(s1, h1 * 0.76 + h0 * 0.24),
                     pt(sm, h1 + 1.4), pt(s0, h1 * 0.76 + h0 * 0.24)]
                pen.blot(q, jitter=0.08)
                # stone arch voussoir rim
                pen.stroke([pt(s0 - 0.6, h1 * 0.74 + h0 * 0.26), pt(sm, h1 + 2.0), pt(s1 + 0.6, h1 * 0.74 + h0 * 0.26)],
                           W0 * 0.5, taper=(0.15, 0.15), smooth=1, wobble=0.08)
            else:
                pen.blot([pt(s0, h0), pt(s1, h0), pt(s1, h1), pt(s0, h1)], jitter=0.08)
                pen.line(pt(s0 - 1.0, h1 + 0.7), pt(s1 + 1.0, h1 + 0.7), W0 * 0.58, sketch=0)  # lintel

    if marks and shade != "dark":
        # Masonry stone courses & subtle corner quoins
        for _ in range(rng.integers(3, 8)):
            s = rng.uniform(2, max(3, L - 6))
            top = prof_at(prof, s)
            if top < 6:
                continue
            h = top - rng.uniform(2, min(14, top - 2))
            ln = rng.uniform(3.0, 6.0)
            pen.stroke([pt(s, h), pt(s + ln, h)], W0 * 0.44, taper=(0.25, 0.25), wobble=0.1)
            if rng.random() < 0.55:
                pen.stroke([pt(s + ln * 0.5, h), pt(s + ln * 0.5, h - 2.6)], W0 * 0.38, taper=(0.15, 0.15))
        # Subtle corner quoin blocks on lit corners
        for s_c, sgn in ((0.0, 1.0), (L, -1.0)):
            ht = prof_at(prof, s_c)
            if ht > 10:
                for hh in np.linspace(3.0, min(ht - 3.0, 40.0), max(2, int(min(ht, 40) / 6.5))):
                    ql = rng.uniform(2.0, 3.4)
                    pen.stroke([pt(s_c, hh), pt(s_c + sgn * ql, hh)], W0 * 0.42, taper=(0.05, 0.3), wobble=0.05)

    pen.stroke([pt(s, h) for s, h in prof], W0, taper=(0.03, 0.03), wobble=0.22)
    pen.line(pt(0, 0), pt(0, prof[0][1]))
    pen.line(pt(L, 0), pt(L, prof[-1][1]))
    pen.line(pt(0, 0), pt(L, 0), W0 * 1.15)


# --------------------------------------------------------------------------- roofs

def draw_roof(pen, S, S2, u0, u1, state, hi_broken, Lr):
    """S(u, v) is the visible slope (v=0 eave, v=1 ridge); S2 the far slope."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    span = u1 - u0
    step = 7.0 / Lr

    def rafter(p, q):
        pen.beam(p, q, thick=1.7)

    if state == "rafters":
        us = np.arange(u0 + step * rng.uniform(0.2, 0.8), u1, step)
        for u in us:
            if rng.random() < 0.8:
                rafter(S2(u, 1), S2(u, 1 - (1 if rng.random() < 0.6 else rng.uniform(0.3, 0.8))))
        ra = u0 if hi_broken else u0 + span * rng.uniform(0, 0.3)
        rb = u1 if not hi_broken else u1 - span * rng.uniform(0, 0.3)
        rafter(S(ra, 1), S(rb, 1))
        for u in us:
            if rng.random() < 0.85:
                if rng.random() < 0.65:
                    rafter(S(u, 0), S(u, 1))
                elif rng.random() < 0.5:
                    rafter(S(u, 0), S(u, rng.uniform(0.3, 0.8)))
                else:
                    rafter(S(u, 1), S(u, rng.uniform(0.2, 0.7)))
        return

    # partial roof: intact end ue, ragged shingle courses toward the broken end
    ue, ub = (u0, u1) if hi_broken else (u1, u0)
    sign = -1 if hi_broken else 1
    rows = int(rng.integers(5, 9))
    vs = np.linspace(0, 1, rows + 1)
    base = rng.uniform(0.05, 0.3)
    row_u = [ub + sign * span * min(0.9, base + rng.uniform(0, 0.3) + (rng.random() < 0.2) * 0.2)
             for _ in range(rows)]
    for u in np.arange(ub + sign * step * 0.5, ub + sign * span, sign * step):
        if (u - min(row_u)) * sign < 0 or rng.random() < 0.3:
            continue
        rafter(S(u, 0), S(u, 1 if rng.random() < 0.6 else rng.uniform(0.4, 0.9)))
    path = []
    for i in range(rows):
        path += [S(row_u[i], vs[i]), S(row_u[i], vs[i + 1])]
    poly = [S(ue, 0)] + path + [S(ue, 1)]
    cv.erase(poly)
    spacing = 2.1 / Lr
    for i in range(rows):
        lo, hi = sorted((ue, row_u[i]))
        for u in np.arange(lo + spacing * rng.uniform(0.2, 0.8), hi, spacing):
            if rng.random() < 0.88:
                a = S(u + rng.normal(0, spacing * 0.14), vs[i] + 0.01)
                b = S(u + rng.normal(0, spacing * 0.14), vs[i + 1] - 0.01)
                pen.stroke([b, a], W0 * 0.56, taper=(0.15, 0.3), wobble=0.1)
        if 0 < i:
            pen.stroke([S(ue, vs[i]), S(row_u[i] if sign * (row_u[i] - row_u[i - 1]) < 0 else row_u[i - 1], vs[i])],
                       W0 * 0.48, taper=(0.05, 0.2), wobble=0.18)
    pen.stroke(path, W0 * 0.9, taper=(0.02, 0.02), wobble=0.15)
    pen.line(S(ue, 0), S(row_u[0], 0), W0 * 1.2, sketch=0)
    pen.line(S(ue, 1), S(row_u[-1], 1), W0 * 1.0, sketch=0)
    pen.line(S(ue, 0), S(ue, 1), W0 * 1.1)


# --------------------------------------------------------------------------- houses & keeps

def plan_house(rng, k=1.0, severity=None, keep=False):
    sev = rng.uniform(0.18, 0.95) if severity is None else severity
    if keep:
        W = rng.uniform(28, 40) * k
        D = W * rng.uniform(0.88, 1.12)
        H, R, ridge_x, roof = rng.uniform(52, 78) * k, 0.0, True, "none"
    else:
        ridge_x = rng.random() < 0.6
        if ridge_x:
            W, D = rng.uniform(55, 90) * k, rng.uniform(30, 42) * k
        else:
            W, D = rng.uniform(32, 44) * k, rng.uniform(50, 80) * k
        H = rng.uniform(20, 28) * k
        R = (D if ridge_x else W) / 2 * rng.uniform(0.75, 1.15)
        p = [0.45, 0.3, 0.25] if sev < 0.45 else [0.12, 0.35, 0.53]
        roof = str(rng.choice(["partial", "rafters", "none"], p=p))
    t = rng.uniform(2.8, 3.8) * max(k, 0.85) if keep else rng.uniform(2.4, 3.4) * max(k, 0.8)
    Lr = W if ridge_x else D
    hi_broken = rng.random() < 0.5
    roof_u = protect = None
    if roof != "none":
        span = rng.uniform(0.35, 0.7) if roof == "partial" else rng.uniform(0.4, 0.9)
        roof_u = (0.0, span) if hi_broken else (1 - span, 1.0)
        protect = (roof_u[0] * Lr - 3, roof_u[1] * Lr + 3)
    allw = (-1e9, 1e9)
    g_lo = allw if roof_u and roof_u[0] == 0 else None
    g_hi = allw if roof_u and roof_u[1] == 1 else None

    def dp(L, gable, prot):
        return damage_profile(rng, L, H, R, gable, sev, prot)

    if ridge_x:
        walls = dict(front=dp(W, False, protect), back=dp(W, False, protect),
                     left=dp(D, True, g_lo), right=dp(D, True, g_hi))
    else:
        walls = dict(left=dp(D, False, protect), right=dp(D, False, protect),
                     front=dp(W, True, g_lo), back=dp(W, True, g_hi))

    def windows(L, n, door):
        ops = []
        ww, wh, sill = rng.uniform(4.5, 6.5) * k, rng.uniform(6, 9) * k, H * rng.uniform(0.3, 0.42)
        slots = np.linspace(0, L, n + 2)[1:-1] + rng.normal(0, L * 0.03, n)
        di = int(rng.integers(0, n)) if door else -1
        for i, s in enumerate(slots):
            if i == di:
                ops.append((s - ww * 0.7, s + ww * 0.7, 0.0, min(H * 0.72, wh * 1.9)))
            else:
                ops.append((s - ww / 2, s + ww / 2, sill, sill + wh))
        return ops

    def slits(L):
        ops = []
        for frac in (0.30, 0.58, 0.82):
            for s in rng.choice(np.linspace(L * 0.22, L * 0.78, 3), size=int(rng.integers(1, 3)), replace=False):
                ops.append((s - 1.15, s + 1.15, H * frac, H * frac + rng.uniform(5.2, 8.2)))
        return ops

    if keep:
        ops = dict(front=slits(W) + [(W * 0.5 - 4.2, W * 0.5 + 4.2, 0, 12.0)] * (rng.random() < 0.7),
                   right=slits(D))
    else:
        ops = dict(front=windows(W, max(1, int(W / 22)), True), right=windows(D, max(1, int(D / 28)), False))
    return dict(W=W, D=D, H=H, R=R, t=t, k=k, sev=sev, ridge_x=ridge_x, roof=roof, roof_u=roof_u,
                hi_broken=hi_broken, walls=walls, ops=ops, keep=keep,
                chimney=(not keep) and rng.random() < 0.48,
                ivy=rng.random() < 0.45)


def _chimney(pen, Q, x0, y0, cw, hc):
    """Tapered medieval stone chimney with wider hearth base and weathered flue cap."""
    # Broader stone hearth base tapering up into chimney stack
    hb = hc * 0.42
    box(pen, Q, x0 - 1.2, y0 - 1.2, x0 + cw + 1.2, y0 + cw + 1.2, 0, hb, inset=1.0)
    box(pen, Q, x0, y0, x0 + cw, y0 + cw, hb, hc)
    # Dark chimney flue opening at top
    pen.blot([Q(x0 + 0.8, y0 + 0.8, hc), Q(x0 + cw - 0.8, y0 + 0.8, hc),
              Q(x0 + cw - 0.8, y0 + cw - 0.8, hc), Q(x0 + 0.8, y0 + cw - 0.8, hc)], jitter=0.1)


def draw_house(pen, P, X0, Y0, pl, exterior=True):
    rng, W0 = pen.rng, pen.width
    W, D, H, R, t, k = pl["W"], pl["D"], pl["H"], pl["R"], pl["t"], pl["k"]
    wl, ops = pl["walls"], pl["ops"]

    def Q(x, y, h=0.0):
        return P(X0 + x, Y0 + y, h)

    # Back exterior shrubs (west/north sides) drawn BEFORE walls so walls cleanly occlude them
    if exterior:
        house_exterior_back(pen, P, X0, Y0, pl)

    # Stepped foundation plinth on keeps
    if pl["keep"]:
        box(pen, P, X0 - 2.2 * k, Y0 - 2.2 * k, X0 + W + 2.2 * k, Y0 + D + 2.2 * k, 0, 4.0 * k, inset=1.0 * k)

    # far walls (inner faces visible over the broken near walls)
    draw_wall(pen, Q, (0, D - t), (W, D - t), wl["back"], (0, t), shade="light")
    draw_wall(pen, Q, (t, 0), (t, D), wl["left"], (-t, 0), shade="dark")

    # Surviving timber floor-joist stubs poking out of a ruined keep's inner wall
    if pl["keep"] and rng.random() < 0.7:
        for frac in (0.36, 0.66):
            hf = H * frac
            for s in np.linspace(t + 4, W - t - 4, max(2, int(W / 10))):
                if prof_at(wl["back"], s) > hf + 4 and rng.random() < 0.6:
                    pen.beam(Q(s, D - t, hf), Q(s, D - t - rng.uniform(3.5, 8.5), hf - rng.uniform(0, 1.5)), 1.3)

    cw = rng.uniform(6, 8) * k if pl["chimney"] else 0.0
    # interior debris & collapsed roof timber (drawn BEFORE chimney & kept clear of it)
    if pl["sev"] > 0.25 and W > 2 * t + 14 and D > 2 * t + 10:
        x_lo = t + (cw + 5.0 if (pl["chimney"] and pl["ridge_x"]) else 6.0)
        y_hi = D - t - (cw + 5.0 if (pl["chimney"] and not pl["ridge_x"]) else 5.0)
        if x_lo < W - t - 6 and t + 4 < y_hi:
            for _ in range(rng.integers(1, 3)):
                cx, cy = Q(rng.uniform(x_lo, W - t - 6), rng.uniform(t + 4, y_hi))
                pen.rubble(cx, cy, rng.uniform(5, 9) * k, rng.uniform(2.5, 4.0) * k, rmax=2.5)
    if pl["chimney"]:
        hc = H + R + rng.uniform(-5, 6) * k
        if pl["ridge_x"]:
            _chimney(pen, Q, t, D / 2 - cw / 2, cw, hc)
        else:
            _chimney(pen, Q, W / 2 - cw / 2, D - t - cw, cw, hc)
    if pl["roof"] != "none" and pl["roof"] != "partial" and rng.random() < 0.55 and W > 2 * t + 10:
        x = rng.uniform(t + 4, W - t - 4)
        hb = prof_at(wl["back"], x)
        pen.beam(Q(x, D - t, min(hb, H) - 1), Q(x + rng.uniform(-12, 12), rng.uniform(t, D * 0.5), 0), 1.6)

    if pl["ridge_x"]:
        eave = H - 2 * R / (D / 2)
        S = lambda u, v: _lerp(Q(u * W, -2, eave), Q(u * W, D / 2, H + R), v)
        S2 = lambda u, v: _lerp(Q(u * W, D + 2, eave), Q(u * W, D / 2, H + R), v)
    else:
        eave = H - 2 * R / (W / 2)
        S = lambda u, v: _lerp(Q(W + 2, u * D, eave), Q(W / 2, u * D, H + R), v)
        S2 = lambda u, v: _lerp(Q(-2, u * D, eave), Q(W / 2, u * D, H + R), v)
    Lr = W if pl["ridge_x"] else D

    arched = pl["keep"] or pl.get("arched", False)
    draw_wall(pen, Q, (W, 0), (W, D), wl["right"], (-t, 0), shade="dark", openings=ops["right"], arched=arched)
    if pl["roof"] != "none" and not pl["ridge_x"]:
        draw_roof(pen, S, S2, *pl["roof_u"], pl["roof"], pl["hi_broken"], Lr)
    draw_wall(pen, Q, (0, 0), (W, 0), wl["front"], (0, t), shade=None, openings=ops["front"], arched=arched)
    if pl["roof"] != "none" and pl["ridge_x"]:
        draw_roof(pen, S, S2, *pl["roof_u"], pl["roof"], pl["hi_broken"], Lr)

    # Corner buttresses on keeps for unmistakable fortress silhouette
    if pl["keep"]:
        bw = 3.2 * k
        for bx, by in ((-bw * 0.6, -bw * 0.6), (W - bw * 0.4, -bw * 0.6), (W - bw * 0.4, D - bw * 0.4)):
            hm = min(prof_at(wl["front"], max(0, min(W, bx))), H * 0.62)
            if hm > 10:
                box(pen, Q, bx, by, bx + bw, by + bw, 0, hm, inset=0.6 * k)

    # Creeping ivy climbing a ruined front corner
    if pl.get("ivy"):
        sx = rng.uniform(2, W * 0.35) if rng.random() < 0.5 else rng.uniform(W * 0.65, W - 2)
        hm = min(prof_at(wl["front"], sx) - 1.5, H * 0.72)
        if hm > 7:
            pts = [Q(sx + math.sin(i * 0.9) * 1.4, -0.3, i * 2.8) for i in range(max(3, int(hm / 2.8)))]
            _ivy(pen, pts)

    if exterior:
        house_exterior_front(pen, P, X0, Y0, pl)


def house_exterior_back(pen, P, X0, Y0, pl):
    """Exterior foliage tucked behind the west or north walls (drawn BEFORE the building so walls occlude it)."""
    rng = pen.rng
    W, D, k = pl["W"], pl["D"], pl["k"]
    plinth = 2.6 * k if pl["keep"] else 0.0
    if rng.random() < 0.45:
        br = rng.uniform(5.2, 8.2) * k
        if rng.random() < 0.70:
            # West flank shrub peeking out from behind the left wall
            bx = -plinth - br * 0.55 - rng.uniform(1.0, 3.0) * k
            by = rng.uniform(D * 0.08, D * 0.48)
        else:
            # North-east shoulder peeking from behind the far-right corner
            bx = W + plinth + rng.uniform(1.5, 4.0) * k
            by = D + plinth + rng.uniform(1.0, 3.5) * k
        pen.bush(*P(X0 + bx, Y0 + by), br, outline=True)


def house_exterior_front(pen, P, X0, Y0, pl, y_pad=0.0):
    """Exterior rubble, tufts, shrubs & fallen timbers strictly in front of south & east walls."""
    rng = pen.rng
    W, D, H, k, wl = pl["W"], pl["D"], pl["H"], pl["k"], pl["walls"]
    plinth = (2.8 * k if pl["keep"] else 0.0) + y_pad
    items = []

    def weighted_s(prof, L):
        sc = np.linspace(5, max(6, L - 5), 24)
        wts = np.array([max(0.0, H - prof_at(prof, s)) + 1.0 for s in sc]) ** 2
        return float(rng.choice(sc, p=wts / wts.sum()))

    n_piles = int(rng.integers(1, 3) + (pl["sev"] > 0.6))
    for _ in range(n_piles):
        rx = rng.uniform(6.5, 13.0) * k * (0.6 + pl["sev"])
        if rng.random() < 0.62:
            x = weighted_s(wl["front"], W)
            y = -plinth - rx * 0.48 - rng.uniform(2.8, 6.0) * k
        else:
            x = W + plinth + rx * 0.58 + rng.uniform(2.5, 5.5) * k
            y = min(D - 6.0, weighted_s(wl["right"], D))
        items.append((P.depth(X0 + x, Y0 + y), "rubble", (x, y, rx)))
    for _ in range(rng.integers(3, 7)):
        if rng.random() < 0.62:
            x, y = rng.uniform(0.0, W + 3.0), -plinth - rng.uniform(2.2, 9.5) * k
        else:
            x, y = W + plinth + rng.uniform(2.5, 9.5) * k, rng.uniform(-2.0, D - 5.0)
        items.append((P.depth(X0 + x, Y0 + y), "tuft", (x, y, 0)))
    if rng.random() < 0.48:
        br = rng.uniform(4.8, 7.6) * k
        # Strictly in front of SE corner or front-left SW ground (NEVER on west or NE wall!)
        if rng.random() < 0.60:
            bx, by = W + plinth + br * 0.55 + 2.2 * k, -plinth - br * 0.45 - 2.0 * k
        else:
            bx, by = rng.uniform(-1.0, W * 0.22), -plinth - br * 0.65 - 3.5 * k
        items.append((P.depth(X0 + bx, Y0 + by), "bush", (bx, by, br)))
    if rng.random() < 0.35:
        # Project BOTH endpoints in world space strictly south of the front wall so beam never pierces wall!
        ln = rng.uniform(14, 24) * k
        bx0 = rng.uniform(2.0, max(3.0, W - ln * 0.8))
        by0 = -plinth - rng.uniform(7.5, 11.5) * k
        ang = rng.uniform(-0.22, 0.22)
        bx1 = bx0 + math.cos(ang) * ln
        by1 = min(-plinth - 6.0 * k, by0 + math.sin(ang) * ln * 0.35)
        items.append((P.depth(X0 + (bx0 + bx1) * 0.5, Y0 + (by0 + by1) * 0.5), "beam", (bx0, by0, (bx1, by1))))

    for _, kind, (x, y, r) in sorted(items, key=lambda it: -it[0]):
        sx, sy = P(X0 + x, Y0 + y)
        if kind == "rubble":
            pen.rubble(sx, sy, r, r * 0.45)
        elif kind == "tuft":
            pen.tuft(sx, sy, rng.uniform(3.5, 6))
        elif kind == "bush":
            pen.bush(sx, sy, r, outline=True)
        elif kind == "beam":
            pen.beam((sx, sy), P(X0 + r[0], Y0 + r[1]), 1.8)


house_exterior = house_exterior_front


def ruined_cottage(pen, P, rng):
    pl = plan_house(rng)
    draw_house(pen, P, -pl["W"] / 2, -pl["D"] / 2, pl)


def ruined_keep(pen, P, rng):
    pl = plan_house(rng, keep=True, severity=rng.uniform(0.28, 0.9))
    draw_house(pen, P, -pl["W"] / 2, -pl["D"] / 2, pl)


# --------------------------------------------------------------------------- round tower

def ruined_tower(pen, P, rng, X0=0.0, Y0=0.0, k=1.0, exterior=True):
    """Ruined cylindrical stone watchtower with stepped masonry breaches (no banana spikes or rim gaps)."""
    cv, W0 = pen.cv, pen.width
    r, H, t = rng.uniform(14, 20) * k, rng.uniform(44, 76) * k, rng.uniform(2.8, 3.8) * k
    n = 96
    th = np.linspace(0, TAU, n, endpoint=False)
    # Broad, natural stepped masonry collapse profile (never a sharp shark-fin spike!)
    hmin = H * rng.uniform(0.32, 0.62)
    phi = rng.uniform(0, TAU)
    # Gentle cosine plateau with a second smaller breach, clamped so peaks stay broad & solid
    wave = 0.5 + 0.5 * np.cos(th - phi)
    bump = np.clip(wave ** rng.uniform(0.8, 1.5), 0.0, 1.0)
    if rng.random() < 0.55:
        phi2 = phi + rng.uniform(1.5, 4.7)
        bump = np.maximum(0.0, bump - rng.uniform(0.18, 0.38) * np.maximum(0, np.cos(th - phi2)) ** 2)
    noise = (0.7 * np.sin(3 * th + rng.uniform(0, TAU)) + 0.45 * np.sin(5 * th + rng.uniform(0, TAU))) * 2.2
    course = rng.uniform(3.2, 4.4)
    hs = np.clip(np.round((hmin + (H - hmin) * bump + noise) / course) * course, course * 2, H)
    # Optional remnant merlons on surviving flat top wall sections
    if rng.random() < 0.55:
        for i in range(0, n, 10):
            if hs[i] >= H - course * 0.5 and hs[(i + 4) % n] >= H - course * 0.5:
                hs[i:(i + 4)] += course * 0.85

    a = P.az

    def hat(tt):
        return float(np.interp(tt % TAU, np.append(th, TAU), np.append(hs, hs[0])))

    def C(tt, rad, h):
        return P(X0 + rad * math.cos(tt), Y0 + rad * math.sin(tt), h)

    back = np.linspace(a, a + math.pi, 52)
    front = np.linspace(a + math.pi, a + TAU, 52)

    # Rear exterior shrubs drawn BEFORE cylinder right behind left/right tangent so they hug the ground
    if exterior and rng.random() < 0.38:
        tt_b = a + math.pi * rng.choice([rng.uniform(0.92, 0.99), rng.uniform(0.01, 0.08)])
        br = rng.uniform(5.2, 8.2) * k
        pen.bush(*C(tt_b, r + br * 0.55 + 1.5 * k, 0), br, outline=True)

    # Far half: properly closed wall-top strip (eliminates green transparency gaps!) + shadowed inner face
    outer_back_top = [C(tt, r, hat(tt)) for tt in back]
    inner_back_top = [C(tt, r - t, hat(tt)) for tt in back]
    inner_back_floor = [C(tt, r - t, 0) for tt in back[::-1]]
    back_strip = outer_back_top + inner_back_top[::-1]
    face_in = inner_back_top + inner_back_floor
    cv.erase(face_in)
    cv.erase(back_strip)
    pen.hatch(face_in, 80, 1.75, W0 * 0.68, coverage=(0.72, 1.0))
    # Subtle broken wooden floor joists projecting from shadowed inner wall
    if rng.random() < 0.6:
        hj = H * rng.uniform(0.32, 0.52)
        for tt in np.linspace(a + 0.5, a + math.pi - 0.5, 4):
            if hat(tt) > hj + 5:
                p_in = C(tt, r - t, hj)
                p_out = C(tt, r - t - rng.uniform(3.5, 6.5), hj - rng.uniform(0, 1.0))
                pen.beam(p_in, p_out, 1.2)
    pen.stroke(outer_back_top, W0 * 0.78, taper=(0.02, 0.02), wobble=0.15)
    pen.stroke(inner_back_top, W0 * 0.78, taper=(0.02, 0.02), wobble=0.15)
    if rng.random() < 0.7:
        cx, cy = P(X0, Y0 + r * 0.2)
        pen.rubble(cx, cy, r * 0.58, r * 0.24, rmax=2.3)

    # Near half: outer wall body + thick stone wall top rim
    outer_top = [C(tt, r, hat(tt)) for tt in front]
    inner_top = [C(tt, r - t, hat(tt)) for tt in front]
    body = [C(tt, r, 0) for tt in front] + outer_top[::-1]
    strip = outer_top + inner_top[::-1]
    cv.erase(strip)
    cv.erase(body)
    pen.stroke(inner_top, W0 * 0.7, taper=(0.02, 0.02), wobble=0.12)
    # Stone block ticks across the broken wall-top thickness
    for tt in np.linspace(a + math.pi + 0.2, a + TAU - 0.2, 8):
        pen.stroke([C(tt, r, hat(tt)), C(tt, r - t, hat(tt))], W0 * 0.4, taper=(0.05, 0.05), wobble=0.05)

    # East-side cylindrical shadow hatching
    shade = np.linspace(a + TAU - rng.uniform(0.95, 1.25), a + TAU, 28)
    pen.hatch([C(tt, r, 0) for tt in shade] + [C(tt, r, hat(tt)) for tt in shade[::-1]], 82, 1.65,
              W0 * 0.72, coverage=(0.75, 1.0))

    # Recessed arrow slits & stone arched doorway
    for _ in range(rng.integers(2, 5)):
        tt = rng.uniform(a + math.pi + 0.45, a + TAU - 0.85)
        h0 = rng.uniform(0.22, 0.72) * H
        h1 = h0 + rng.uniform(5.0, 7.8)
        if hat(tt) > h1 + 3.5:
            dt = 1.15 / r
            pen.blot([C(tt - dt, r, h0), C(tt + dt, r, h0), C(tt + dt, r, h1), C(tt - dt, r, h1)], jitter=0.05)
    door_tt = None
    if rng.random() < 0.6:
        tt, dt = a + 1.5 * math.pi + rng.uniform(-0.3, 0.25), 4.0 / r
        if hat(tt) > 16:
            door_tt = tt
            arch = [C(tt - dt, r, 0)] + [C(tt + dt * math.cos(q), r, 9.5 + 3.2 * math.sin(q)) for q in np.linspace(math.pi, 0, 7)] + [C(tt + dt, r, 0)]
            pen.blot(arch, jitter=0.08)
            pen.stroke([C(tt + dt * 1.18 * math.cos(q), r, 9.5 + 3.8 * math.sin(q)) for q in np.linspace(math.pi, 0, 7)],
                       W0 * 0.5, taper=(0.1, 0.1), wobble=0.06)

    # Masonry course lines & stone cracks
    for _ in range(rng.integers(3, 7)):
        tt, h = rng.uniform(a + math.pi + 0.25, a + TAU - 0.55), rng.uniform(3, H * 0.78)
        if hat(tt) > h + 3:
            pen.stroke([C(tt, r, h), C(tt + 4.5 / r, r, h)], W0 * 0.44, taper=(0.25, 0.25))

    # Optional creeping ivy on one flank of the tower
    if rng.random() < 0.45:
        tt0 = a + math.pi + rng.uniform(0.3, 0.9)
        hm = min(hat(tt0) - 2, H * 0.65)
        if hm > 10:
            pts = [C(tt0 + math.sin(i * 0.8) * 0.08, r + 0.2, i * 3.0) for i in range(max(3, int(hm / 3.0)))]
            _ivy(pen, pts)

    # Crisp outer silhouette + stepped stone footing line
    pen.stroke(outer_top, W0, taper=(0.02, 0.02), wobble=0.18)
    pen.line(C(a + math.pi, r, 0), C(a + math.pi, r, hat(a + math.pi)))
    pen.line(C(a, r, 0), C(a, r, hat(a)))
    pen.stroke([C(tt, r, 0) for tt in front], W0 * 1.15, taper=(0.05, 0.05))
    pen.stroke([C(tt, r * 1.03, 2.2) for tt in front], W0 * 0.5, taper=(0.15, 0.15), wobble=0.1)

    if not exterior:
        return

    # Front debris, tufts, and leafy shrubs strictly in front of the cylinder base (never clipping rim/door)
    fg = []
    fall = phi + math.pi
    for _ in range(rng.integers(1, 4)):
        tt = fall + rng.normal(0, 0.6)
        if math.sin(tt - a) > -0.24:
            tt = rng.uniform(a + math.pi + 0.28, a + TAU - 0.28)
        rx_r = rng.uniform(5.5, 10.5) * k
        d = r + rx_r * 0.55 + rng.uniform(3.0, 9.0) * k
        wx, wy = X0 + d * math.cos(tt), Y0 + d * math.sin(tt)
        fg.append((P.depth(wx, wy), "rubble", (wx, wy, rx_r)))
    for _ in range(rng.integers(2, 6)):
        tt = rng.uniform(a + math.pi + 0.25, a + TAU - 0.25)
        d = r + rng.uniform(2.5, 6.5) * k
        wx, wy = X0 + d * math.cos(tt), Y0 + d * math.sin(tt)
        fg.append((P.depth(wx, wy), "tuft", (wx, wy, 0.0)))
    if rng.random() < 0.42:
        br = rng.uniform(4.8, 7.6) * k
        cands = [a + math.pi * 1.24, a + math.pi * 1.78]
        if door_tt is not None:
            cands.sort(key=lambda tt: -abs((tt - door_tt + math.pi) % TAU - math.pi))
            tt_s = cands[0]
        else:
            tt_s = float(rng.choice(cands))
        d = r + br * 0.65 + 3.2 * k
        wx, wy = X0 + d * math.cos(tt_s), Y0 + d * math.sin(tt_s)
        fg.append((P.depth(wx, wy), "bush", (wx, wy, br)))
    for _, kind, (wx, wy, rad) in sorted(fg, key=lambda it: -it[0]):
        sx, sy = P(wx, wy)
        if kind == "rubble":
            pen.rubble(sx, sy, rad, rad * 0.45)
        elif kind == "tuft":
            pen.tuft(sx, sy, rng.uniform(3.5, 6))
        else:
            pen.bush(sx, sy, rad, outline=True)


def ruined_tower_any(pen, P, rng):
    if rng.random() < 0.6:
        ruined_tower(pen, P, rng)
    else:
        ruined_keep(pen, P, rng)


# --------------------------------------------------------------------------- village pieces

def ruined_well(pen, P, X, Y):
    rng, W0 = pen.rng, pen.width
    r, h = rng.uniform(5, 7), rng.uniform(4, 6)
    a = P.az

    def C(tt, rad, hh):
        return P(X + rad * math.cos(tt), Y + rad * math.sin(tt), hh)

    front = np.linspace(a + math.pi, a + TAU, 30)
    ring = np.linspace(0, TAU, 40)
    pen.cv.erase([C(tt, r, h) for tt in ring])
    pen.blot([C(tt, r - 1.6, h) for tt in ring])
    body = [C(tt, r, 0) for tt in front] + [C(tt, r, h) for tt in front[::-1]]
    pen.cv.erase(body)
    pen.closed([C(tt, r, h) for tt in ring], W0 * 0.9)
    pen.stroke([C(tt, r, 0) for tt in front], W0, taper=(0.05, 0.05))
    pen.line(C(a + math.pi, r, 0), C(a + math.pi, r, h), sketch=0)
    pen.line(C(a, r, 0), C(a, r, h), sketch=0)
    for _ in range(3):
        tt = rng.uniform(a + math.pi + 0.3, a + TAU - 0.3)
        pen.stroke([C(tt, r, h * 0.5), C(tt + 0.5, r, h * 0.5)], W0 * 0.45, taper=(0.3, 0.3))
    if rng.random() < 0.7:
        ph = rng.uniform(10, 15)
        p1, p2 = C(a + math.pi, r - 0.5, h), C(a, r - 0.5, h)
        t1 = (p1[0] + rng.normal(0, 0.8), p1[1] - ph)
        t2 = (p2[0] + rng.normal(0, 0.8), p2[1] - ph * rng.uniform(0.5, 1.0))
        pen.beam(p1, t1, 1.5)
        pen.beam(p2, t2, 1.5)
        if rng.random() < 0.5:
            pen.beam(t1, t2, 1.3)


def ruined_fence(pen, P, A, B):
    rng, W0 = pen.rng, pen.width
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = max(2, int(np.hypot(*(B - A)) / 7))
    hp = rng.uniform(6, 8)
    posts = []
    for i in range(n + 1):
        g = A + (B - A) * i / n
        if rng.random() < 0.75:
            base = P(*g)
            top = (base[0] + rng.normal(0, 1.0), base[1] - hp * rng.uniform(0.6, 1.0))
            posts.append((i, base, top))
    for (i, b0, t0), (j, b1, t1) in zip(posts, posts[1:]):
        if j == i + 1 and rng.random() < 0.7:
            for f in (0.35, 0.75):
                p = _lerp(b0, t0, f)
                q = _lerp(b1, t1, f)
                if rng.random() < 0.25:
                    q = (q[0], b1[1])
                pen.line(p, q, W0 * 0.6, sketch=0)
    for _, b, t in posts:
        pen.beam(b, t, 1.4)


def plan_ruined_church(rng, k=0.78):
    """Plan a ruined medieval village parish church: stone nave + broken west bell tower."""
    sev = rng.uniform(0.32, 0.78)
    W = rng.uniform(54, 68) * k
    D = rng.uniform(24, 30) * k
    H = rng.uniform(19, 25) * k
    R = (D / 2) * rng.uniform(0.95, 1.25)
    t = rng.uniform(2.6, 3.4) * max(k, 0.82)
    roof = str(rng.choice(["partial", "rafters", "none"], p=[0.40, 0.38, 0.22]))
    hi_broken = rng.random() < 0.65  # tend to preserve roof toward tower end or east end
    roof_u = protect = None
    if roof != "none":
        span = rng.uniform(0.36, 0.65)
        roof_u = (0.0, span) if hi_broken else (1 - span, 1.0)
        protect = (roof_u[0] * W - 3, roof_u[1] * W + 3)
    allw = (-1e9, 1e9)
    g_lo = allw if roof_u and roof_u[0] == 0 else None
    g_hi = allw if roof_u and roof_u[1] == 1 else None
    walls = dict(
        front=damage_profile(rng, W, H, R, False, sev, protect),
        back=damage_profile(rng, W, H, R, False, sev, protect),
        left=damage_profile(rng, D, H, R, True, sev * 0.7, g_lo),
        right=damage_profile(rng, D, H, R, True, sev, g_hi),
    )
    # Tall gothic lancet windows + arched portal along the nave south face
    n_win = max(3, int(W / (14 * k)))
    slots = np.linspace(0, W, n_win + 2)[1:-1]
    ww, wh, sill = 3.4 * k, 9.0 * k, H * 0.26
    front_ops = []
    for i, s in enumerate(slots):
        if i == 0:
            front_ops.append((s - ww * 0.75, s + ww * 0.75, 0.0, min(H * 0.72, wh * 1.45)))
        else:
            front_ops.append((s - ww * 0.5, s + ww * 0.5, sill, sill + wh))
    right_ops = [(D * 0.5 - ww * 0.65, D * 0.5 + ww * 0.65, sill * 0.9, sill * 0.9 + wh * 1.15)]
    tw = rng.uniform(16, 20) * k
    th = rng.uniform(52, 68) * k
    tower_state = str(rng.choice(["steeple_broken", "steeple_intact", "breached"], p=[0.42, 0.28, 0.30]))
    return dict(
        W=W, D=D, H=H, R=R, t=t, k=k, sev=sev, ridge_x=True, roof=roof, roof_u=roof_u,
        hi_broken=hi_broken, walls=walls, ops=dict(front=front_ops, right=right_ops),
        keep=False, arched=True, chimney=False, ivy=rng.random() < 0.70,
        tw=tw, th=th, tower_state=tower_state,
    )


def draw_ruined_church(pen, P, X0, Y0, pl, exterior=True):
    """Draw a ruined village church: weathered/broken west bell tower with crooked cross + ruined gothic nave."""
    rng, W0 = pen.rng, pen.width
    W, D, H, k = pl["W"], pl["D"], pl["H"], pl["k"]
    tw, th = pl["tw"], pl["th"]
    tx0, tx1 = X0 - tw * 0.85, X0 + tw * 0.15
    ty0, ty1 = Y0 + (D - tw) * 0.5, Y0 + (D + tw) * 0.5
    tx, ty = (tx0 + tx1) * 0.5, (ty0 + ty1) * 0.5

    # 1. Ruined west bell tower (strictly west/behind the nave in az=28° view)
    state = pl["tower_state"]
    if state == "breached":
        # Open breached stone belfry shell with stepped masonry collapse & gothic belfry lancets
        t_sev = rng.uniform(0.32, 0.58)
        tt = 2.6 * max(k, 0.85)
        bw = tw * 0.22
        b_ops = [(tw * 0.30 - bw * 0.5, tw * 0.30 + bw * 0.5, th * 0.56, th * 0.76),
                 (tw * 0.70 - bw * 0.5, tw * 0.70 + bw * 0.5, th * 0.56, th * 0.76)]
        r_ops = [(tw * 0.50 - bw * 0.55, tw * 0.50 + bw * 0.55, th * 0.56, th * 0.76)]
        tpl = dict(
            W=tw, D=tw, H=th, R=0.0, t=tt, k=k, sev=t_sev, ridge_x=True, roof="none", roof_u=None,
            hi_broken=True, keep=True, arched=True, chimney=False, ivy=rng.random() < 0.55,
            walls=dict(
                front=damage_profile(rng, tw, th, 0.0, False, t_sev),
                back=damage_profile(rng, tw, th, 0.0, False, t_sev),
                left=damage_profile(rng, tw, th, 0.0, False, t_sev),
                right=damage_profile(rng, tw, th, 0.0, False, t_sev),
            ),
            ops=dict(front=b_ops, right=r_ops),
        )
        draw_house(pen, P, tx0, ty0, tpl, exterior=False)
    else:
        # Standing stone bell tower with cracked belfry & intact or shattered steeple + crooked cross!
        box(pen, P, tx0 - 1.3 * k, ty0 - 1.3 * k, tx1 + 1.3 * k, ty1 + 1.3 * k, 0, th * 0.22)
        box(pen, P, tx0, ty0, tx1, ty1, 0, th)
        pen.stroke([P(tx0, ty0, th * 0.65), P(tx1, ty0, th * 0.65)], W0 * 0.52, taper=(0.05, 0.05))
        # Twin arched gothic belfry openings on south face
        bh = th * 0.19
        bw = tw * 0.18
        for off in (-tw * 0.20, tw * 0.20):
            bx = tx + off
            pen.blot([P(bx - bw * 0.5, ty0, th * 0.72), P(bx + bw * 0.5, ty0, th * 0.72),
                      P(bx + bw * 0.5, ty0, th * 0.72 + bh * 0.72), P(bx, ty0, th * 0.72 + bh * 1.08),
                      P(bx - bw * 0.5, ty0, th * 0.72 + bh * 0.72)], jitter=0.05)
        # Masonry cracks & creeping ivy up the ruined tower
        pen.stroke([P(tx0 + tw * 0.28, ty0, th * 0.58), P(tx0 + tw * 0.35, ty0, th * 0.42),
                    P(tx0 + tw * 0.26, ty0, th * 0.28)], W0 * 0.44, taper=(0.1, 0.7), wobble=0.14)
        if state == "steeple_intact":
            apex = th + rng.uniform(22, 30) * k
            pyramid(pen, P, tx0 - 1.0 * k, ty0 - 1.0 * k, tx1 + 1.0 * k, ty1 + 1.0 * k, th, apex)
            # Slightly leaning iron cross atop the old weathered steeple
            ax, ay = P(tx, ty, apex)
            lean = rng.uniform(-1.6, 1.8)
            ch = 8.5 * max(k, 0.85)
            pen.stroke([(ax, ay), (ax + lean, ay - ch)], W0 * 0.66, taper=(0.0, 0.15), wobble=0)
            pen.stroke([(ax + lean * 0.68 - ch * 0.34, ay - ch * 0.68 + lean * 0.15),
                        (ax + lean * 0.68 + ch * 0.34, ay - ch * 0.68 - lean * 0.15)],
                       W0 * 0.58, taper=(0.1, 0.1), wobble=0)
        else:
            # Shattered / snapped skeletal timber steeple frame sticking out of the belfry!
            apex = th + rng.uniform(16, 24) * k
            for cx_c, cy_c in ((tx0, ty0), (tx1, ty0), (tx1, ty1), (tx0, ty1)):
                frac = rng.uniform(0.52, 0.96)
                px = cx_c + (tx - cx_c) * frac
                py = cy_c + (ty - cy_c) * frac
                pz = th + (apex - th) * frac
                pen.beam(P(cx_c, cy_c, th), P(px, py, pz), 1.35)
            # Partial broken shingle skirt on the west side of the spire
            sh_top = th + (apex - th) * 0.42
            px_a = tx0 + (tx - tx0) * 0.42
            py_a = ty0 + (ty - ty0) * 0.42
            skirt = [P(tx0 - 0.8, ty0 - 0.8, th), P(tx1 + 0.8, ty0 - 0.8, th),
                     P(tx1 - (tx1 - tx) * 0.38, ty0 + (ty - ty0) * 0.38, sh_top), P(px_a, py_a, sh_top)]
            pen.cv.erase(skirt)
            pen.hatch(skirt, 74, 2.0, W0 * 0.46, coverage=(0.45, 0.85))
            pen.closed(skirt, W0 * 0.72, wobble=0.14)
            # Crooked broken cross leaning from the snapped peak
            ax, ay = P(tx, ty, th + (apex - th) * 0.78)
            pen.stroke([(ax, ay), (ax + 2.4, ay - 6.8)], W0 * 0.58, taper=(0.0, 0.2), wobble=0)
            pen.stroke([(ax - 0.8, ay - 4.6), (ax + 4.0, ay - 3.6)], W0 * 0.52, taper=(0.1, 0.1), wobble=0)

    # 2. Ruined gothic stone nave with arched lancets & south buttresses
    draw_house(pen, P, X0, Y0, pl, exterior=False)
    for frac in (0.20, 0.52, 0.82):
        bx = X0 + W * frac
        bh_b = min(H * 0.68, prof_at(pl["walls"]["front"], W * frac) - 2.0)
        if bh_b > 5.0:
            box(pen, P, bx - 1.2 * k, Y0 - 2.0 * k, bx + 1.2 * k, Y0 + 0.4, 0, bh_b, inset=0.4 * k)
    if exterior:
        house_exterior_front(pen, P, X0, Y0, pl, y_pad=2.5 * k)


def _ruined_track(pen, P, rng, pts_world, half_w=3.0):
    """Faint broken overgrown village track along world polyline `pts_world`."""
    W0 = pen.width
    pts_w = np.asarray(pts_world, float)
    tang = np.gradient(pts_w, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    nrm /= np.maximum(1e-6, np.hypot(nrm[:, 0], nrm[:, 1]))[:, None]
    for sgn in (-1.0, 1.0):
        edge = [P(*(p + n * (sgn * half_w))) for p, n in zip(pts_w, nrm)]
        i = 0
        while i < len(edge) - 3:
            j = min(len(edge), i + int(rng.integers(5, 13)))
            if rng.random() < 0.65:
                pen.stroke(edge[i:j], W0 * 0.42, taper=(0.35, 0.35), wobble=0.45)
            i = j + int(rng.integers(2, 5))


def ruined_village(pen, P, rng):
    """Ruined village in three varied shapes: wide (E-W), tall (screen-vertical N-S), or round/crossroads."""
    shape = str(rng.choice(["wide", "tall", "round"], p=[0.34, 0.33, 0.33]))
    if shape == "wide":
        L = rng.uniform(160, 210)
        xs = np.linspace(-L / 2, L / 2, 40)
        ys = rng.uniform(6, 14) * np.sin(xs / L * math.pi * rng.uniform(1, 1.8) + rng.uniform(0, TAU))
        _ruined_track(pen, P, rng, np.stack([xs, ys], axis=1))
    elif shape == "tall":
        L = rng.uniform(155, 195)
        ys = np.linspace(-L / 2, L / 2, 40)
        xs = -0.44 * ys + rng.uniform(6, 12) * np.sin(ys / L * math.pi * rng.uniform(1, 1.6) + rng.uniform(0, TAU))
        _ruined_track(pen, P, rng, np.stack([xs, ys], axis=1))
    else:
        L1 = rng.uniform(120, 155)
        xs1 = np.linspace(-L1 / 2, L1 / 2, 30)
        ys1 = rng.uniform(3, 8) * np.sin(xs1 / L1 * math.pi + rng.uniform(0, TAU))
        L2 = rng.uniform(115, 150)
        ys2 = np.linspace(-L2 / 2, L2 / 2, 30)
        xs2 = -0.44 * ys2 + rng.uniform(3, 8) * np.sin(ys2 / L2 * math.pi + rng.uniform(0, TAU))
        _ruined_track(pen, P, rng, np.stack([xs1, ys1], axis=1))
        _ruined_track(pen, P, rng, np.stack([xs2, ys2], axis=1))

    placed = []
    items = []
    # ~55% of ruined villages have a ruined parish church centerpiece!
    if rng.random() < 0.55:
        cpl = plan_ruined_church(rng, k=rng.uniform(0.72, 0.84))
        if shape == "wide":
            cx, cy = rng.uniform(-34, 18), rng.uniform(10, 34)
        elif shape == "tall":
            cy = rng.uniform(18, 52)
            cx = -0.44 * cy + rng.uniform(-22, 16) - cpl["W"] * 0.4
        else:
            cx, cy = rng.uniform(-32, 10), rng.uniform(14, 42)
        cbox = (cx - cpl["tw"] - 12, cy - 12, cx + cpl["W"] + 10, cy + cpl["D"] + 14)
        placed.append((cpl, (cx, cy), cbox))
        c_fp = (cx - cpl["tw"] * 0.90, cy - 2.5, cx + cpl["W"] + 1.5, cy + cpl["D"] + 1.5)
        items.append((c_fp, ("church", (cpl, cx, cy))))

    n = int(rng.integers(3, 6)) + (1 if items else 0)
    for _ in range(320):
        if len(placed) >= n:
            break
        k = rng.uniform(0.62, 0.85)
        pl = plan_house(rng, k=k)
        if shape == "wide":
            X, Y = rng.uniform(-90, 50), rng.uniform(-45, 30)
        elif shape == "tall":
            cy_h = rng.uniform(-82, 68)
            cx_h = -0.44 * cy_h + rng.uniform(-48, 48)
            X, Y = cx_h - pl["W"] / 2, cy_h - pl["D"] / 2
        else:
            ang = rng.uniform(0, TAU)
            rad = rng.uniform(22, 74)
            cx_h = rad * math.cos(ang) * 0.92 - 0.22 * rad * math.sin(ang)
            cy_h = rad * math.sin(ang) * 1.02
            X, Y = cx_h - pl["W"] / 2, cy_h - pl["D"] / 2
        b_box = (X - 8, Y - 10, X + pl["W"] + 8, Y + pl["D"] + 8)
        if all(b_box[2] < b[0] or b_box[0] > b[2] or b_box[3] < b[1] or b_box[1] > b[3] for _, _, b in placed):
            placed.append((pl, (X, Y), b_box))
            items.append(((X, Y, X + pl["W"], Y + pl["D"]), ("house", (pl, X, Y))))

    bx0 = min(b[0] for _, _, b in placed)
    bx1 = max(b[2] for _, _, b in placed)
    by0 = min(b[1] for _, _, b in placed)
    by1 = max(b[3] for _, _, b in placed)

    # Actual tight building footprints for accurate clearance & 3D occlusion sorting
    bldg_fps = [it[0] for it in items]

    def free(x, y, m=6):
        return all(not (fp[0] - m < x < fp[2] + m and fp[1] - m < y < fp[3] + m) for fp in bldg_fps)

    for _ in range(40):
        x, y = rng.uniform(bx0, bx1), rng.uniform(by0, by1)
        if free(x, y, 12):
            if rng.random() < 0.72:
                wfp = (x - 7.0, y - 7.0, x + 7.0, y + 7.0)
                bldg_fps.append(wfp)
                items.append((wfp, ("well", (x, y))))
            break
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(30):
            x, y = rng.uniform(bx0 - 8, bx1 + 8), rng.uniform(by0 - 8, by1 + 8)
            ang = rng.choice([0, math.pi / 2]) + rng.normal(0, 0.1)
            fl = rng.uniform(22, 42)
            x2, y2 = x + math.cos(ang) * fl, y + math.sin(ang) * fl
            if free(x, y, 5) and free(x2, y2, 5) and free((x + x2) / 2, (y + y2) / 2, 5):
                ffp = (min(x, x2) - 1.0, min(y, y2) - 1.0, max(x, x2) + 1.0, max(y, y2) + 1.0)
                items.append((ffp, ("fence", ((x, y), (x2, y2)))))
                break
    # Scatter rubble, tufts & shrubs strictly outside building footprints (sorted via 3D box DAG!)
    for _ in range(int(rng.integers(8, 14))):
        kind = str(rng.choice(["rubble", "rubble", "tuft", "tuft", "tuft", "bush"]))
        rad = rng.uniform(5.5, 9.5) if kind == "rubble" else (rng.uniform(4.8, 7.4) if kind == "bush" else 3.0)
        for _ in range(18):
            x, y = rng.uniform(bx0 - 10, bx1 + 10), rng.uniform(by0 - 10, by1 + 10)
            if free(x, y, rad + 3.5):
                items.append(((x - rad, y - rad * 0.6, x + rad, y + rad * 0.6), (kind, (x, y, rad))))
                break
    if rng.random() < 0.65:
        for _ in range(int(rng.integers(1, 3))):
            for _ in range(18):
                x, y = rng.uniform(bx0 - 10, bx1 + 10), rng.uniform(by0 - 10, by1 + 10)
                if free(x, y, 8.0):
                    items.append(((x - 4.5, y - 4.5, x + 4.5, y + 4.5), ("tree", (x, y))))
                    break

    for kind, args in sort_boxes(P, items):
        if kind == "house":
            draw_house(pen, P, args[1], args[2], args[0], exterior=False)
        elif kind == "church":
            draw_ruined_church(pen, P, args[1], args[2], args[0], exterior=False)
        elif kind == "well":
            ruined_well(pen, P, *args)
        elif kind == "fence":
            ruined_fence(pen, P, *args)
        elif kind == "rubble":
            pen.rubble(*P(args[0], args[1]), args[2], args[2] * 0.45)
        elif kind == "tuft":
            pen.tuft(*P(args[0], args[1]), rng.uniform(3.5, 6))
        elif kind == "bush":
            pen.bush(*P(args[0], args[1]), args[2], outline=True)
        elif kind == "tree":
            x, y = P(*args)
            pen.dead_tree(x, y, rng.uniform(18, 30), lean=rng.normal(0, 0.15))
