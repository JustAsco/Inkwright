"""Ruined buildings: cottages, square keeps, round towers, village clusters."""
import math

import numpy as np

from .ink import TAU, clip_halfplane


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def prof_at(prof, s):
    return float(np.interp(s, [p[0] for p in prof], [p[1] for p in prof]))


def prof_min(prof, s0, s1):
    return min([prof_at(prof, s0), prof_at(prof, s1)] + [h for s, h in prof if s0 <= s <= s1])


# --------------------------------------------------------------------------- walls

def damage_profile(rng, L, H, R, gable, severity, protect=None):
    """Top edge of a wall as [(s, h), ...]: intact (flat or gable) with stepped breaches."""
    gable = gable and R > 0
    top = H + (R if gable else 0.0)

    def intact(s):
        return H + (R * (1 - np.abs(2 * s / L - 1)) if gable else 0.0)

    ss = np.linspace(0, L, max(3, int(L / rng.uniform(3.0, 4.5)) + 1))
    if gable:
        ss = np.union1d(ss, [L / 2])
    n = len(ss)
    cap = np.full(n, 1e9)
    for _ in range(int(rng.integers(0, 2)) + (severity > 0.35) + (severity > 0.7)):
        c, hw = rng.uniform(-0.15 * L, 1.15 * L), rng.uniform(0.2, 0.55) * L
        floor = H * rng.uniform(0.05, max(0.12, 0.8 - 0.6 * severity))
        f = np.clip(1 - np.abs(ss - c) / hw, 0, 1) ** (1 / rng.uniform(1, 3))
        cap = np.minimum(cap, top + 2 - f * (top + 2 - floor))
    if severity > 0.75 and rng.random() < 0.5:
        cap = np.minimum(cap, H * rng.uniform(0.2, 0.55))
    if protect is not None:
        cap[(ss >= protect[0]) & (ss <= protect[1])] = 1e9
    vals = intact(ss)
    broken = cap < vals - 0.5
    course = rng.uniform(2.5, 3.8)
    hb = np.round((cap + rng.normal(0, 1.6, n)) / course) * course
    h = np.where(broken, np.clip(np.minimum(hb, vals), 1.5, None), vals)
    pts = [(ss[0], h[0])]
    for i in range(1, n):
        if (broken[i] or broken[i - 1]) and abs(h[i] - h[i - 1]) > 0.4 and rng.random() < 0.55:
            pts.append((ss[i], h[i - 1]))
        pts.append((ss[i], h[i]))
    return pts


def draw_wall(pen, Q, a, b, prof, far, shade=None, openings=(), marks=True):
    """One wall face from ground point a to b with top profile; `far` = thickness vector."""
    rng, cv, W0 = pen.rng, pen.cv, pen.width
    a, b, far = np.asarray(a, float), np.asarray(b, float), np.asarray(far, float)
    L = float(np.hypot(*(b - a)))
    u = (b - a) / L

    def pt(s, h, f=False):
        g = a + u * s + (far if f else 0.0)
        return Q(g[0], g[1], h)

    rev = list(reversed(prof))
    face = [pt(0, 0), pt(L, 0)] + [pt(s, h) for s, h in rev]
    back = [pt(0, 0, True), pt(L, 0, True)] + [pt(s, h, True) for s, h in rev]
    cv.erase(back)
    cv.erase(face)
    # top surface of the wall (thickness)
    pen.stroke([pt(s, h, True) for s, h in prof], W0 * 0.65, taper=(0.03, 0.03), wobble=0.25)
    pen.line(pt(0, prof[0][1]), pt(0, prof[0][1], True), W0 * 0.65, sketch=0, overshoot=0.3)
    pen.line(pt(L, prof[-1][1]), pt(L, prof[-1][1], True), W0 * 0.65, sketch=0, overshoot=0.3)

    if shade == "dark":
        pen.hatch(face, 80 + rng.normal(0, 4), rng.uniform(1.4, 1.9), W0 * 0.75, coverage=(0.75, 1.0))
    elif shade == "light":
        pen.hatch(face, 80, rng.uniform(3.0, 4.0), W0 * 0.45, coverage=(0.25, 0.6))

    for (s0, s1, h0, h1) in openings:
        if prof_min(prof, s0 - 1, s1 + 1) > h1 + 2.5:
            pen.blot([pt(s0, h0), pt(s1, h0), pt(s1, h1), pt(s0, h1)])
            pen.line(pt(s0 - 1, h1 + 0.8), pt(s1 + 1, h1 + 0.8), W0 * 0.6, sketch=0)  # lintel

    if marks and shade != "dark":
        for _ in range(rng.integers(2, 7)):
            s = rng.uniform(2, L - 6)
            top = prof_at(prof, s)
            if top < 6:
                continue
            h = top - rng.uniform(2, min(10, top - 2))
            ln = rng.uniform(2.5, 5)
            pen.stroke([pt(s, h), pt(s + ln, h)], W0 * 0.45, taper=(0.3, 0.3), wobble=0.1)
            if rng.random() < 0.5:
                pen.stroke([pt(s + ln * 0.5, h), pt(s + ln * 0.5, h - 2.5)], W0 * 0.4, taper=(0.2, 0.2))

    pen.stroke([pt(s, h) for s, h in prof], W0, taper=(0.03, 0.03), wobble=0.25)
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

    # partial roof: intact end ue, ragged courses toward the broken end
    ue, ub = (u0, u1) if hi_broken else (u1, u0)
    sign = -1 if hi_broken else 1
    rows = int(rng.integers(5, 9))
    vs = np.linspace(0, 1, rows + 1)
    base = rng.uniform(0.05, 0.3)
    row_u = [ub + sign * span * min(0.9, base + rng.uniform(0, 0.3) + (rng.random() < 0.2) * 0.2)
             for _ in range(rows)]
    # exposed rafters in the collapsed part
    for u in np.arange(ub + sign * step * 0.5, ub + sign * span, sign * step):
        if (u - min(row_u)) * sign < 0 or rng.random() < 0.3:
            continue
        rafter(S(u, 0), S(u, 1 if rng.random() < 0.6 else rng.uniform(0.4, 0.9)))
    path = []
    for i in range(rows):
        path += [S(row_u[i], vs[i]), S(row_u[i], vs[i + 1])]
    poly = [S(ue, 0)] + path + [S(ue, 1)]
    cv.erase(poly)
    spacing = 1.9 / Lr
    for i in range(rows):
        lo, hi = sorted((ue, row_u[i]))
        for u in np.arange(lo + spacing * rng.uniform(0.2, 0.8), hi, spacing):
            if rng.random() < 0.9:
                a = S(u + rng.normal(0, spacing * 0.15), vs[i] + 0.01)
                b = S(u + rng.normal(0, spacing * 0.15), vs[i + 1] - 0.01)
                pen.stroke([b, a], W0 * 0.6, taper=(0.15, 0.3), wobble=0.1)
        if 0 < i:
            pen.stroke([S(ue, vs[i]), S(row_u[i] if sign * (row_u[i] - row_u[i - 1]) < 0 else row_u[i - 1], vs[i])],
                       W0 * 0.5, taper=(0.05, 0.2), wobble=0.2)
    pen.stroke(path, W0 * 0.9, taper=(0.02, 0.02), wobble=0.15)
    pen.line(S(ue, 0), S(row_u[0], 0), W0 * 1.2, sketch=0)
    pen.line(S(ue, 1), S(row_u[-1], 1), W0 * 1.0, sketch=0)
    pen.line(S(ue, 0), S(ue, 1), W0 * 1.1)


# --------------------------------------------------------------------------- houses

def plan_house(rng, k=1.0, severity=None, keep=False):
    sev = rng.uniform(0.15, 1.0) if severity is None else severity
    if keep:
        W = rng.uniform(26, 38) * k
        D = W * rng.uniform(0.85, 1.15)
        H, R, ridge_x, roof = rng.uniform(50, 80) * k, 0.0, True, "none"
    else:
        ridge_x = rng.random() < 0.6
        if ridge_x:
            W, D = rng.uniform(55, 90) * k, rng.uniform(30, 42) * k
        else:
            W, D = rng.uniform(32, 44) * k, rng.uniform(50, 80) * k
        H = rng.uniform(20, 28) * k
        R = (D if ridge_x else W) / 2 * rng.uniform(0.75, 1.15)
        p = [0.45, 0.3, 0.25] if sev < 0.45 else [0.1, 0.35, 0.55]
        roof = str(rng.choice(["partial", "rafters", "none"], p=p))
    t = rng.uniform(2.4, 3.4) * max(k, 0.8)
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
        for frac in (0.3, 0.6, 0.85):
            for s in rng.choice(np.linspace(L * 0.2, L * 0.8, 3), size=int(rng.integers(0, 3)), replace=False):
                ops.append((s - 1.1, s + 1.1, H * frac, H * frac + rng.uniform(5, 8)))
        return ops

    if keep:
        ops = dict(front=slits(W) + [(W * 0.5 - 4, W * 0.5 + 4, 0, 11)] * (rng.random() < 0.6), right=slits(D))
    else:
        ops = dict(front=windows(W, max(1, int(W / 22)), True), right=windows(D, max(1, int(D / 28)), False))
    return dict(W=W, D=D, H=H, R=R, t=t, k=k, sev=sev, ridge_x=ridge_x, roof=roof, roof_u=roof_u,
                hi_broken=hi_broken, walls=walls, ops=ops, keep=keep,
                chimney=(not keep) and rng.random() < 0.4)


def _chimney(pen, Q, x0, y0, cw, hc):
    x1, y1 = x0 + cw, y0 + cw
    front = [Q(x0, y0, 0), Q(x1, y0, 0), Q(x1, y0, hc), Q(x0, y0, hc)]
    side = [Q(x1, y0, 0), Q(x1, y1, 0), Q(x1, y1, hc), Q(x1, y0, hc)]
    top = [Q(x0, y0, hc), Q(x1, y0, hc), Q(x1, y1, hc), Q(x0, y1, hc)]
    for poly in (front, side, top):
        pen.cv.erase(poly)
    pen.hatch(side, 82, 1.6, pen.width * 0.7, coverage=(0.8, 1.0))
    for poly in (front, side, top):
        pen.closed(poly, pen.width * 0.85, overshoot=0.6)
    pen.blot([Q(x0 + 0.8, y0 + 0.8, hc), Q(x1 - 0.8, y0 + 0.8, hc), Q(x1 - 0.8, y1 - 0.8, hc), Q(x0 + 0.8, y1 - 0.8, hc)])


def draw_house(pen, P, X0, Y0, pl, exterior=True):
    rng, W0 = pen.rng, pen.width
    W, D, H, R, t, k = pl["W"], pl["D"], pl["H"], pl["R"], pl["t"], pl["k"]
    wl, ops = pl["walls"], pl["ops"]

    def Q(x, y, h=0.0):
        return P(X0 + x, Y0 + y, h)

    # far walls (inner faces visible over the broken near walls)
    draw_wall(pen, Q, (0, D - t), (W, D - t), wl["back"], (0, t), shade="light")
    draw_wall(pen, Q, (t, 0), (t, D), wl["left"], (-t, 0), shade="dark")

    if pl["chimney"]:
        cw = rng.uniform(6, 8) * k
        hc = H + R + rng.uniform(-6, 6) * k
        if pl["ridge_x"]:
            _chimney(pen, Q, t, D / 2 - cw / 2, cw, hc)
        else:
            _chimney(pen, Q, W / 2 - cw / 2, D - t - cw, cw, hc)

    # interior debris
    if pl["sev"] > 0.3 and W > 2 * t + 14 and D > 2 * t + 10:
        for _ in range(rng.integers(1, 3)):
            cx, cy = Q(rng.uniform(t + 6, W - t - 6), rng.uniform(t + 4, D - t - 4))
            pen.rubble(cx, cy, rng.uniform(5, 10) * k, rng.uniform(2.5, 4.5) * k, rmax=2.6)
    if pl["roof"] != "partial" and rng.random() < 0.6 and W > 2 * t + 10:
        x = rng.uniform(t + 4, W - t - 4)
        hb = prof_at(wl["back"], x)
        pen.beam(Q(x, D - t, min(hb, H) - 1), Q(x + rng.uniform(-12, 12), rng.uniform(t, D * 0.5), 0), 1.7)

    if pl["ridge_x"]:
        eave = H - 2 * R / (D / 2)
        S = lambda u, v: _lerp(Q(u * W, -2, eave), Q(u * W, D / 2, H + R), v)
        S2 = lambda u, v: _lerp(Q(u * W, D + 2, eave), Q(u * W, D / 2, H + R), v)
    else:
        eave = H - 2 * R / (W / 2)
        S = lambda u, v: _lerp(Q(W + 2, u * D, eave), Q(W / 2, u * D, H + R), v)
        S2 = lambda u, v: _lerp(Q(-2, u * D, eave), Q(W / 2, u * D, H + R), v)
    Lr = W if pl["ridge_x"] else D

    draw_wall(pen, Q, (W, 0), (W, D), wl["right"], (-t, 0), shade="dark", openings=ops["right"])
    if pl["roof"] != "none" and not pl["ridge_x"]:
        draw_roof(pen, S, S2, *pl["roof_u"], pl["roof"], pl["hi_broken"], Lr)
    draw_wall(pen, Q, (0, 0), (W, 0), wl["front"], (0, t), shade=None, openings=ops["front"])
    if pl["roof"] != "none" and pl["ridge_x"]:
        draw_roof(pen, S, S2, *pl["roof_u"], pl["roof"], pl["hi_broken"], Lr)

    if exterior:
        house_exterior(pen, P, X0, Y0, pl)


def house_exterior(pen, P, X0, Y0, pl):
    rng = pen.rng
    W, D, H, k, wl = pl["W"], pl["D"], pl["H"], pl["k"], pl["walls"]
    items = []

    def weighted_s(prof, L):
        sc = np.linspace(4, L - 4, 24)
        wts = np.array([max(0.0, H - prof_at(prof, s)) + 1.0 for s in sc]) ** 2
        return float(rng.choice(sc, p=wts / wts.sum()))

    n_piles = int(rng.integers(1, 3) + (pl["sev"] > 0.6))
    for _ in range(n_piles):
        if rng.random() < 0.6:
            x, y = weighted_s(wl["front"], W), -rng.uniform(3, 7) * k
        else:
            x, y = W + rng.uniform(3, 7) * k, weighted_s(wl["right"], D)
        rx = rng.uniform(7, 14) * k * (0.6 + pl["sev"])
        items.append((P.depth(X0 + x, Y0 + y), "rubble", (x, y, rx)))
    for _ in range(rng.integers(3, 8)):
        if rng.random() < 0.6:
            x, y = rng.uniform(-4, W + 4), -rng.uniform(1, 9)
        else:
            x, y = W + rng.uniform(1, 9), rng.uniform(-2, D)
        items.append((P.depth(X0 + x, Y0 + y), "tuft", (x, y, 0)))
    for _ in range(rng.integers(0, 3)):
        corner = [(-3, -3), (W + 3, -3), (W + 3, D - 4), (-4, D * 0.3)][rng.integers(0, 4)]
        items.append((P.depth(X0 + corner[0], Y0 + corner[1]), "bush", (*corner, rng.uniform(5, 9) * k)))
    if rng.random() < 0.35:
        items.append((P.depth(X0 + W * 0.5, Y0 - 8), "beam", (rng.uniform(0, W * 0.6), -rng.uniform(5, 10), 0)))

    for _, kind, (x, y, r) in sorted(items, key=lambda it: -it[0]):
        sx, sy = P(X0 + x, Y0 + y)
        if kind == "rubble":
            pen.rubble(sx, sy, r, r * 0.45)
        elif kind == "tuft":
            pen.tuft(sx, sy, rng.uniform(3.5, 6))
        elif kind == "bush":
            pen.bush(sx, sy, r)
        elif kind == "beam":
            ln = rng.uniform(14, 24) * k
            a = rng.uniform(-0.4, 0.4)
            pen.beam((sx, sy), (sx + math.cos(a) * ln, sy + math.sin(a) * ln * 0.4), 1.8)


def ruined_cottage(pen, P, rng):
    pl = plan_house(rng)
    draw_house(pen, P, -pl["W"] / 2, -pl["D"] / 2, pl)


def ruined_keep(pen, P, rng):
    pl = plan_house(rng, keep=True, severity=rng.uniform(0.3, 1.0))
    draw_house(pen, P, -pl["W"] / 2, -pl["D"] / 2, pl)


# --------------------------------------------------------------------------- round tower

def ruined_tower(pen, P, rng, X0=0.0, Y0=0.0, k=1.0):
    cv, W0 = pen.cv, pen.width
    r, H, t = rng.uniform(13, 20) * k, rng.uniform(45, 85) * k, rng.uniform(2.6, 3.6) * k
    n = 120
    th = np.linspace(0, TAU, n, endpoint=False)
    hmin = H * rng.uniform(0.25, 0.6)
    phi = rng.uniform(0, TAU)
    bump = np.maximum(0, np.cos(th - phi)) ** rng.uniform(1.5, 5)
    noise = sum(rng.normal(0, 1) * np.sin(f * th + rng.uniform(0, TAU)) for f in (3, 5, 9)) * 1.5
    course = rng.uniform(2.6, 3.6)
    hs = np.clip(np.round((hmin + (H - hmin) * bump + noise) / course) * course, 3, None)
    a = P.az

    def hat(tt):
        return float(np.interp(tt % TAU, np.append(th, TAU), np.append(hs, hs[0])))

    def C(tt, rad, h):
        return P(X0 + rad * math.cos(tt), Y0 + rad * math.sin(tt), h)

    back = np.linspace(a, a + math.pi, 60)
    front = np.linspace(a + math.pi, a + TAU, 60)
    # far half: rim and inner face
    inner_back_top = [C(tt, r - t, hat(tt)) for tt in back]
    inner_back_floor = [C(tt, r - t, 0) for tt in back[::-1]]
    face_in = inner_back_top + inner_back_floor
    cv.erase(face_in)
    pen.hatch(face_in, 80, 1.7, W0 * 0.7, coverage=(0.7, 1.0))
    pen.stroke([C(tt, r, hat(tt)) for tt in back], W0 * 0.7, taper=(0.02, 0.02))
    pen.stroke(inner_back_top, W0 * 0.8, taper=(0.02, 0.02))
    if rng.random() < 0.7:
        cx, cy = P(X0, Y0 + r * 0.2)
        pen.rubble(cx, cy, r * 0.6, r * 0.25, rmax=2.4)
    # near half
    outer_top = [C(tt, r, hat(tt)) for tt in front]
    inner_top = [C(tt, r - t, hat(tt)) for tt in front]
    body = [C(tt, r, 0) for tt in front] + outer_top[::-1]
    strip = outer_top + inner_top[::-1]
    cv.erase(strip)
    cv.erase(body)
    pen.stroke(inner_top, W0 * 0.7, taper=(0.02, 0.02))
    shade = np.linspace(a + TAU - rng.uniform(0.9, 1.3), a + TAU, 30)
    pen.hatch([C(tt, r, 0) for tt in shade] + [C(tt, r, hat(tt)) for tt in shade[::-1]], 82, 1.6,
              W0 * 0.75, coverage=(0.75, 1.0))
    # windows / arrow slits
    for _ in range(rng.integers(1, 5)):
        tt = rng.uniform(a + math.pi + 0.5, a + TAU - 0.9)
        h0 = rng.uniform(0.2, 0.75) * H
        h1 = h0 + rng.uniform(5, 8)
        if hat(tt) > h1 + 3:
            dt = 1.2 / r
            pen.blot([C(tt - dt, r, h0), C(tt + dt, r, h0), C(tt + dt, r, h1), C(tt - dt, r, h1)])
    if rng.random() < 0.5:
        tt, dt = a + 1.5 * math.pi + rng.uniform(-0.3, 0.3), 4.0 / r
        pen.blot([C(tt - dt, r, 0)] + [C(tt + dt * math.cos(q), r, 10 + 3 * math.sin(q)) for q in np.linspace(math.pi, 0, 7)] + [C(tt + dt, r, 0)])
    for _ in range(rng.integers(2, 6)):
        tt, h = rng.uniform(a + math.pi + 0.3, a + TAU - 0.6), rng.uniform(3, H * 0.8)
        if hat(tt) > h + 3:
            pen.stroke([C(tt, r, h), C(tt + 4 / r, r, h)], W0 * 0.45, taper=(0.3, 0.3))
    pen.stroke(outer_top, W0, taper=(0.02, 0.02), wobble=0.2)
    pen.line(C(a + math.pi, r, 0), C(a + math.pi, r, hat(a + math.pi)))
    pen.line(C(a, r, 0), C(a, r, hat(a)))
    pen.stroke([C(tt, r, 0) for tt in front], W0 * 1.15, taper=(0.05, 0.05))
    # debris on the collapsed side
    fall = phi + math.pi
    for _ in range(rng.integers(1, 4)):
        tt = fall + rng.normal(0, 0.6)
        d = r + rng.uniform(4, 12) * k
        x, y = P(X0 + d * math.cos(tt), Y0 + d * math.sin(tt))
        if math.sin(tt - a) < 0.2:  # roughly in front
            pen.rubble(x, y, rng.uniform(6, 12) * k, rng.uniform(3, 5) * k)
    for _ in range(rng.integers(2, 6)):
        tt = rng.uniform(a + math.pi, a + TAU)
        x, y = C(tt, r + rng.uniform(1, 5), 0)
        pen.tuft(x, y + 1, rng.uniform(3.5, 6))
    if rng.random() < 0.5:
        x, y = C(a + math.pi * rng.choice([1.15, 1.85]), r + 3, 0)
        pen.bush(x, y + 2, rng.uniform(6, 10) * k)


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
                    q = (q[0], b1[1])  # sagged to the ground
                pen.line(p, q, W0 * 0.6, sketch=0)
    for _, b, t in posts:
        pen.beam(b, t, 1.4)


def ruined_village(pen, P, rng):
    n = int(rng.integers(3, 6))
    placed = []
    for _ in range(200):
        if len(placed) >= n:
            break
        k = rng.uniform(0.6, 0.85)
        pl = plan_house(rng, k=k)
        X, Y = rng.uniform(-90, 50), rng.uniform(-55, 35)
        box = (X - 8, Y - 10, X + pl["W"] + 8, Y + pl["D"] + 8)
        if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for _, _, b in placed):
            placed.append((pl, (X, Y), box))
    items = [(P.depth(X + pl["W"] / 2, Y + pl["D"] / 2), "house", (pl, X, Y)) for pl, (X, Y), _ in placed]

    def free(x, y, m=6):
        return all(not (b[0] - m < x < b[2] + m and b[1] - m < y < b[3] + m) for _, _, b in placed)

    for _ in range(40):
        x, y = rng.uniform(-70, 80), rng.uniform(-50, 50)
        if free(x, y, 10):
            if rng.random() < 0.7:
                items.append((P.depth(x, y), "well", (x, y)))
            break
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(30):
            x, y = rng.uniform(-90, 90), rng.uniform(-60, 60)
            ang = rng.choice([0, math.pi / 2]) + rng.normal(0, 0.1)
            L = rng.uniform(25, 50)
            x2, y2 = x + math.cos(ang) * L, y + math.sin(ang) * L
            if free(x, y, 3) and free(x2, y2, 3) and free((x + x2) / 2, (y + y2) / 2, 3):
                items.append((P.depth((x + x2) / 2, (y + y2) / 2), "fence", ((x, y), (x2, y2))))
                break
    for _ in range(int(rng.integers(4, 9))):
        x, y = rng.uniform(-90, 100), rng.uniform(-60, 60)
        if free(x, y, 2):
            kind = rng.choice(["rubble", "tuft", "tuft", "bush"])
            items.append((P.depth(x, y), str(kind), (x, y)))
    if rng.random() < 0.5:
        x, y = rng.uniform(-90, 100), rng.uniform(-60, 60)
        if free(x, y, 4):
            items.append((P.depth(x, y), "tree", (x, y)))

    for _, kind, args in sorted(items, key=lambda it: -it[0]):
        if kind == "house":
            draw_house(pen, P, args[1], args[2], args[0])
        elif kind == "well":
            ruined_well(pen, P, *args)
        elif kind == "fence":
            ruined_fence(pen, P, *args)
        elif kind == "rubble":
            pen.rubble(*P(*args), rng.uniform(5, 10), rng.uniform(2.5, 4))
        elif kind == "tuft":
            pen.tuft(*P(*args), rng.uniform(3.5, 6))
        elif kind == "bush":
            pen.bush(*P(*args), rng.uniform(5, 9))
        elif kind == "tree":
            x, y = P(*args)
            pen.dead_tree(x, y, rng.uniform(18, 30), lean=rng.normal(0, 0.15))
