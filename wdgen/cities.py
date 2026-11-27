"""Urban settlements, intact castles, big villages and fortified fantasy cities.

Adds:
- Townhouses (narrow 2-3 storey medieval urban houses, jettied half-timbering, crow-step gables,
  row-house terraces, L-shaped corner inns/taverns)
- Town Parts (civic guildhalls/clocktowers, market squares with awnings & fountains, grand town
  cathedrals, courtyard merchant manors, tower & post windmills, timber-crane warehouses)
- City Walls (modular intact crenellated E-W/N-S wall runs, twin-towered gatehouses & watchtowers)
- Intact Castles (wide, deep, hexagonal & concentric stone fortresses with battlements, conical
  turrets, portcullis gatehouses, banners & royal keeps)
- Big Villages (20-28 houses along sprawling crossroads, high-streets + lanes, or wide ring greens,
  with church, windmill, well, fences & trees)
- Cities (~20-26 densely clustered overlapping townhouses & civic landmarks wrapped in fortified
  city walls, bastion towers, gatehouses and rising citadels / cathedrals)
"""
import math

import numpy as np

from .castles import _capped_turret, _curtain_seg, _square_tower, crenellate, curtain
from .flora import tree
from .houses import (
    _draw_bell_tower,
    _draw_cross,
    _draw_road_strip,
    _house_item,
    _overlaps,
    _render,
    _trees_around,
    _tufts,
    chapel,
    chapel_footprint,
    draw_faces,
    draw_house,
    fence,
    footprint,
    house_faces,
    plan_chapel,
    plan_house,
    well,
)
from .ink import TAU, ellipse_pts
from .proj import sort_boxes
from .ruins import (
    _ruined_track,
    damage_profile,
    draw_house as draw_ruined_house,
    draw_ruined_church,
    plan_house as plan_ruined_house,
    plan_ruined_church,
    ruined_fence,
    ruined_tower,
    ruined_well,
)
from .solids import box, cone, cylinder, pyramid


# --------------------------------------------------------------------------- shared intact fortress helpers

def _draw_flag(pen, P, x, y, z, h=10.5, w_scale=0.66, flip=False):
    """Crisp upright medieval banner pennant fluttering from a flagpole at world (x, y, z)."""
    W0 = pen.width
    ax, ay = P(x, y, z)
    top_y = ay - h
    pen.stroke([(ax, ay), (ax, top_y)], W0 * w_scale, taper=(0.0, 0.15), wobble=0.0)
    sgn = -1.0 if flip else 1.0
    bw = h * 0.72
    bh = h * 0.34
    p0 = (ax, top_y + 0.6)
    p1 = (ax + sgn * bw * 0.55, top_y + 0.2)
    p2 = (ax + sgn * bw, top_y + bh * 0.30)
    p3 = (ax + sgn * bw * 0.72, top_y + bh * 0.56)
    p4 = (ax + sgn * bw, top_y + bh * 0.88)
    p5 = (ax + sgn * bw * 0.52, top_y + bh * 0.78)
    p6 = (ax, top_y + bh)
    poly = [p0, p1, p2, p3, p4, p5, p6]
    pen.cv.erase(poly)
    pen.closed(poly, W0 * (w_scale * 0.72), wobble=0.06)
    # subtle stripe/fold line along pennant
    pen.stroke([(ax + sgn * 1.2, top_y + bh * 0.50), (ax + sgn * bw * 0.60, top_y + bh * 0.52)],
               W0 * (w_scale * 0.44), taper=(0.15, 0.35), wobble=0.04)


def _draw_masonry_courses(pen, P, a, b, h0, h1, rng, n=4, lit=True):
    """Subtle sparse ashlar stone course lines along an intact wall face."""
    if not lit:
        return
    W0 = pen.width
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    if L < 8.0 or h1 - h0 < 6.0:
        return
    u = (b - a) / L
    for _ in range(n):
        s = rng.uniform(2.0, max(2.5, L - 5.5))
        sl = rng.uniform(2.6, min(6.5, L - s - 1.0))
        hz = rng.uniform(h0 + 2.5, h1 - 2.5)
        g0 = a + u * s
        g1 = a + u * (s + sl)
        pen.stroke([P(g0[0], g0[1], hz), P(g1[0], g1[1], hz)], W0 * 0.36, taper=(0.25, 0.25), wobble=0.05)


def _draw_slits(pen, P, a, b, z_mid, n, slit_h=4.4, crosslet=False):
    """Crisp dark vertical arrow-slit embrasures along a stone fortification wall."""
    W0 = pen.width
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    if L < 8.0 or n <= 0:
        return
    u = (b - a) / L
    for s in np.linspace(0.0, L, n + 2)[1:-1]:
        g = a + u * s
        p0, p1 = P(g[0], g[1], z_mid - slit_h * 0.5), P(g[0], g[1], z_mid + slit_h * 0.5)
        pen.stroke([p0, p1], W0 * 0.92, taper=(0.08, 0.08), wobble=0.02)
        if crosslet:
            xm, ym = P(g[0], g[1], z_mid + slit_h * 0.06)
            pen.stroke([(xm - 1.5, ym), (xm + 1.5, ym)], W0 * 0.62, taper=(0.15, 0.15), wobble=0.0)


def intact_curtain(pen, P, a, b, H, t, rng, outer_side=None, crenel=True, slits=True, buttresses=False):
    """Unbroken intact crenellated axis-aligned stone curtain wall from world `a` to `b`."""
    W0 = pen.width
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    if L < 3.0:
        return
    if abs(by - ay) < 1e-6:  # East-West wall
        x0, x1 = min(ax, bx), max(ax, bx)
        y0, y1 = ay - t * 0.5, ay + t * 0.5
        box(pen, P, x0, y0, x1, y1, 0.0, H)
        # subtle walkway molding line & masonry courses on sunlit south wall
        pen.stroke([P(x0, y0, H - 1.6), P(x1, y0, H - 1.6)], W0 * 0.52, taper=(0.04, 0.04), wobble=0.08)
        _draw_masonry_courses(pen, P, (x0, y0), (x1, y0), 0.0, H - 2.0, rng, n=max(2, int(L / 14)))
        if slits and L > 16.0:
            _draw_slits(pen, P, (x0, y0), (x1, y0), H * 0.56, max(1, int(L / 16)), slit_h=min(5.0, H * 0.24),
                        crosslet=(H > 22.0))
        if buttresses and L > 34.0:
            for bx_p in np.linspace(x0 + 12.0, x1 - 12.0, max(2, int(L / 26))):
                box(pen, P, bx_p - 1.8, y0 - 2.2, bx_p + 1.8, y0 + 0.3, 0.0, H * 0.74, inset=0.45)
        if crenel:
            mt = t * 0.46
            # Back wall merlons sit along north edge; front wall merlons along south edge
            my0, my1 = (y1 - mt, y1) if outer_side == "north" else (y0, y0 + mt)
            period = 5.8
            mw, mh = 2.8, 3.0
            k = 0
            while x0 + 1.2 + k * period + mw < x1 - 0.8:
                mx0 = x0 + 1.2 + k * period
                box(pen, P, mx0, my0, mx0 + mw, my1, H, H + mh, w=W0 * 0.66)
                k += 1
    else:  # North-South wall
        y0, y1 = min(ay, by), max(ay, by)
        x0, x1 = ax - t * 0.5, ax + t * 0.5
        box(pen, P, x0, y0, x1, y1, 0.0, H)
        pen.stroke([P(x1, y0, H - 1.6), P(x1, y1, H - 1.6)], W0 * 0.48, taper=(0.04, 0.04), wobble=0.08)
        if slits and L > 16.0:
            _draw_slits(pen, P, (x1, y0), (x1, y1), H * 0.56, max(1, int(L / 16)), slit_h=min(4.6, H * 0.22))
        if buttresses and L > 34.0:
            for by_p in np.linspace(y1 - 12.0, y0 + 12.0, max(2, int(L / 26))):
                box(pen, P, x1 - 0.3, by_p - 1.8, x1 + 2.2, by_p + 1.8, 0.0, H * 0.74, inset=0.45)
        if crenel:
            mt = t * 0.46
            mx0, mx1 = (x0, x0 + mt) if outer_side == "west" else (x1 - mt, x1)
            period = 5.8
            mw, mh = 2.8, 3.0
            # Draw north-to-south (back-to-front!) so merlons occlude cleanly
            ys_m = []
            k = 0
            while y0 + 1.2 + k * period + mw < y1 - 0.8:
                ys_m.append(y0 + 1.2 + k * period)
                k += 1
            for my0 in reversed(ys_m):
                box(pen, P, mx0, my0, mx1, my0 + mw, H, H + mh, w=W0 * 0.66)


def intact_curtain_seg(pen, P, a, b, H, t, rng, show_inner=False, crenel=True, slits=True):
    """Arbitrary-angle intact crenellated stone wall segment `a -> b` around (0, 0)."""
    W0 = pen.width
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    if L < 3.0:
        return
    u = (b - a) / L
    n_in = np.array([-u[1], u[0]])
    if float(n_in @ (-(a + b) * 0.5)) < 0:
        n_in = -n_in
    n_out = -n_in

    if show_inner:
        p0, p1 = a + n_in * t, b + n_in * t
        thick = -n_in * t
        shaded = n_in[0] > 0.28
        light_shade = (not shaded) and (n_in[0] > -0.15)
    else:
        p0, p1 = a, b
        thick = n_in * t
        shaded = n_out[0] > 0.28
        light_shade = False

    def Wp(s, h, off=(0.0, 0.0)):
        g = p0 + u * s + np.asarray(off)
        return P(g[0], g[1], h)

    # 1. Wall body front face + top walkway deck
    face = [Wp(0, 0), Wp(L, 0), Wp(L, H), Wp(0, H)]
    topf = [Wp(0, H), Wp(L, H), Wp(L, H, thick), Wp(0, H, thick)]
    pen.cv.erase(face)
    pen.cv.erase(topf)
    if shaded:
        pen.hatch(face, 82, 1.65, W0 * 0.68, coverage=(0.76, 1.0))
    elif light_shade:
        pen.hatch(face, 82, 2.35, W0 * 0.44, coverage=(0.45, 0.85))
    else:
        _draw_masonry_courses(pen, P, p0, p1, 0.0, H - 2.0, rng, n=max(2, int(L / 14)))
    if slits and not show_inner and L > 16.0:
        _draw_slits(pen, P, p0, p1, H * 0.56, max(1, int(L / 16)), slit_h=min(4.8, H * 0.22), crosslet=(H > 22))
    pen.closed(face, W0 * 0.84, overshoot=0.35, wobble=0.18)
    pen.closed(topf, W0 * 0.78, overshoot=0.25, wobble=0.16)
    pen.stroke([Wp(0, H - 1.5), Wp(L, H - 1.5)], W0 * 0.46, taper=(0.05, 0.05), wobble=0.06)

    # 2. Crenellation merlons along the outer edge (sorted back-to-front by P.depth!)
    if crenel:
        period, mw, mh = 5.8, 2.8, 2.9
        m_off = thick * 0.55 if show_inner else (0.0, 0.0)
        m_th = thick * 0.45
        merlons = []
        k = 0
        while 1.2 + k * period + mw < L - 0.8:
            s0 = 1.2 + k * period
            mid_g = p0 + u * (s0 + mw * 0.5) + np.asarray(m_off)
            merlons.append((P.depth(mid_g[0], mid_g[1]), s0))
            k += 1
        for _, s0 in sorted(merlons, key=lambda it: -it[0]):
            s1 = s0 + mw
            mf = [Wp(s0, H, m_off), Wp(s1, H, m_off), Wp(s1, H + mh, m_off), Wp(s0, H + mh, m_off)]
            mt = [Wp(s0, H + mh, m_off), Wp(s1, H + mh, m_off),
                  Wp(s1, H + mh, np.asarray(m_off) + m_th), Wp(s0, H + mh, np.asarray(m_off) + m_th)]
            pen.cv.erase(mf)
            pen.cv.erase(mt)
            if shaded:
                pen.hatch(mf, 82, 1.7, W0 * 0.58, coverage=(0.7, 1.0))
            pen.closed(mf, W0 * 0.64, wobble=0.08)
            pen.closed(mt, W0 * 0.58, wobble=0.08)


def intact_round_tower(pen, P, rng, X, Y, r, H, cap="cone", flag=True, machicolations=True):
    """Intact round medieval stone fortress tower (conical spire roof OR crenellated battlement crown)."""
    W0 = pen.width
    a = P.az
    # Flared base plinth
    cylinder(pen, P, X, Y, r * 1.16, 0.0, min(6.0, H * 0.14), top=False, w=W0 * 0.88)
    C = cylinder(pen, P, X, Y, r, min(4.5, H * 0.10), H, top=(cap == "crenel"), w=W0 * 0.90)

    # Horizontal stone stringcourse band
    hm = H * 0.58
    pen.stroke([C(t, hm) for t in np.linspace(a + math.pi, a + TAU, 22)], W0 * 0.52, taper=(0.05, 0.05), wobble=0.06)
    # Crisp dark arrow slits / arched windows on lit front face
    n_w = 2 if H > 34 else 1
    for j in range(n_w):
        hz = H * (0.34 + 0.34 * j) if n_w == 2 else H * 0.48
        t_ang = a + 1.5 * math.pi + rng.uniform(-0.35, 0.25)
        dt = min(0.18, 1.4 / max(4.0, r))
        q = [C(t_ang - dt, hz), C(t_ang + dt, hz), C(t_ang + dt, hz + 4.4),
             C(t_ang, hz + 5.6), C(t_ang - dt, hz + 4.4)]
        pen.blot(q, jitter=0.04)
    # Subtle masonry ticks
    for _ in range(max(2, int(H / 16))):
        t_ang = a + 1.5 * math.pi + rng.uniform(-0.75, 0.35)
        hz = rng.uniform(7.0, max(8.0, H - 5.0))
        pen.stroke([C(t_ang, hz), C(t_ang + 3.2 / max(5.0, r), hz)], W0 * 0.36, taper=(0.25, 0.25), wobble=0.05)

    if cap == "cone":
        if machicolations and r >= 7.5:
            rb = r * 1.14
            for t in np.linspace(a + math.pi + 0.16, a + TAU - 0.16, 9):
                pen.stroke([P(X + r * math.cos(t), Y + r * math.sin(t), H - 2.6),
                            P(X + rb * math.cos(t), Y + rb * math.sin(t), H)],
                           W0 * 0.52, taper=(0.2, 0.05), wobble=0.04)
            cylinder(pen, P, X, Y, rb, H, H + 3.2, top=False, w=W0 * 0.80)
            h_roof = H + 3.2
            r_roof = rb * 1.12
        else:
            h_roof = H
            r_roof = r * 1.20
        apex = h_roof + r * rng.uniform(1.85, 2.55)
        cone(pen, P, X, Y, r_roof, h_roof, apex, broken=0.0)
        if flag:
            _draw_flag(pen, P, X, Y, apex, h=max(7.5, min(11.5, r * 0.92)), flip=(rng.random() < 0.25))
    else:
        rb = r * 1.16
        if machicolations and r >= 7.5:
            for t in np.linspace(a + math.pi + 0.18, a + TAU - 0.18, 9):
                pen.stroke([P(X + r * math.cos(t), Y + r * math.sin(t), H - 2.5),
                            P(X + rb * math.cos(t), Y + rb * math.sin(t), H)],
                           W0 * 0.50, taper=(0.2, 0.05), wobble=0.04)
        cylinder(pen, P, X, Y, rb, H, H + 3.4, top=True, w=W0 * 0.82)
        # Evenly spaced stone merlons around the circular battlement rim (sorted back-to-front!)
        n_m = max(6, int(round(rb * 0.65)))
        ts = np.linspace(0.0, TAU, n_m, endpoint=False) + 0.2
        m_items = []
        for t in ts:
            mx, my = X + (rb - 1.1) * math.cos(t), Y + (rb - 1.1) * math.sin(t)
            m_items.append((P.depth(mx, my), mx, my))
        for _, mx, my in sorted(m_items, key=lambda it: -it[0]):
            box(pen, P, mx - 1.4, my - 1.4, mx + 1.4, my + 1.4, H + 3.2, H + 6.2, w=W0 * 0.62)
        if flag:
            _draw_flag(pen, P, X, Y, H + 3.4, h=max(9.5, min(13.5, r * 1.05)), flip=(rng.random() < 0.25))


def intact_square_tower(pen, P, rng, X, Y, s, H, cap="crenel", flag=True):
    """Intact square stone watchtower / bastion tower with battlements OR steep pyramidal roof."""
    W0 = pen.width
    x0, x1, y0, y1 = X - s, X + s, Y - s, Y + s
    # Stepped plinth
    box(pen, P, x0 - 1.4, y0 - 1.4, x1 + 1.4, y1 + 1.4, 0.0, min(5.5, H * 0.14), inset=0.4)
    box(pen, P, x0, y0, x1, y1, min(4.5, H * 0.10), H, top=(cap == "crenel"))
    # Stringcourse + quoins + arrow slits
    hm = H * 0.58
    pen.stroke([P(x0, y0, hm), P(x1, y0, hm)], W0 * 0.52, taper=(0.04, 0.04), wobble=0.06)
    pen.stroke([P(x1, y0, hm), P(x1, y1, hm)], W0 * 0.48, taper=(0.04, 0.04), wobble=0.06)
    _draw_masonry_courses(pen, P, (x0, y0), (x1, y0), 6.0, H - 3.0, rng, n=max(2, int(H / 14)))
    n_sl = 2 if s > 9.0 else 1
    _draw_slits(pen, P, (x0, y0), (x1, y0), H * 0.74, n_sl, slit_h=4.6, crosslet=True)
    if H > 34.0:
        _draw_slits(pen, P, (x0, y0), (x1, y0), H * 0.38, 1, slit_h=4.2)
    _draw_slits(pen, P, (x1, y0), (x1, y1), H * 0.74, 1, slit_h=4.4)

    sp = s * 1.14
    if cap == "pyramid":
        # Corbelled cornice + steep tiled pyramidal roof + flagpole
        for kx in np.linspace(-s * 0.8, s * 0.8, 4):
            pen.stroke([P(X + kx, y0, H - 2.4), P(X + kx * 1.08, Y - sp, H)], W0 * 0.48, taper=(0.2, 0.05), wobble=0)
        box(pen, P, X - sp, Y - sp, X + sp, Y + sp, H, H + 3.0, top=False, w=W0 * 0.78)
        apex = H + 3.0 + s * rng.uniform(1.85, 2.65)
        pyramid(pen, P, X - sp * 1.06, Y - sp * 1.06, X + sp * 1.06, Y + sp * 1.06, H + 3.0, apex)
        if flag:
            _draw_flag(pen, P, X, Y, apex, h=max(8.0, min(11.5, s * 0.95)), flip=(rng.random() < 0.25))
    else:
        for kx in np.linspace(-s * 0.8, s * 0.8, 4):
            pen.stroke([P(X + kx, y0, H - 2.4), P(X + kx * 1.08, Y - sp, H)], W0 * 0.48, taper=(0.2, 0.05), wobble=0)
        box(pen, P, X - sp, Y - sp, X + sp, Y + sp, H, H + 3.2, top=True, w=W0 * 0.82)
        # Crisp battlement merlons around all 4 parapet edges sorted back-to-front
        mw = 2.6
        coords = []
        pts_m = [
            (X - sp, Y + sp - mw), (X - mw * 0.5, Y + sp - mw), (X + sp - mw, Y + sp - mw),
            (X - sp, Y - mw * 0.5), (X + sp - mw, Y - mw * 0.5),
            (X - sp, Y - sp), (X - mw * 0.5, Y - sp), (X + sp - mw, Y - sp),
        ]
        for mx0, my0 in pts_m:
            coords.append(((mx0, my0, mx0 + mw, my0 + mw), (mx0, my0)))
        for mx0, my0 in sort_boxes(P, coords):
            box(pen, P, mx0, my0, mx0 + mw, my0 + mw, H + 3.2, H + 6.2, w=W0 * 0.62)
        if flag:
            _draw_flag(pen, P, X, Y, H + 3.2, h=max(9.5, min(13.5, s * 1.1)), flip=(rng.random() < 0.25))


def intact_gatehouse(pen, P, rng, gx, gy, gw, H, t, style="round", drawbridge=True, flags=True):
    """Intact fortified medieval twin-towered gatehouse with iron portcullis, parapet & flanking towers."""
    W0 = pen.width
    gh = H * 1.18
    gt = max(t * 1.45, 7.5)

    # Optional short cobbled/timber approach bridge south of the gate arch
    if drawbridge:
        bw = gw * 0.46
        bl = rng.uniform(11.0, 16.0)
        bridge = [P(gx - bw, gy - gt * 0.5), P(gx + bw, gy - gt * 0.5),
                  P(gx + bw * 1.08, gy - gt * 0.5 - bl), P(gx - bw * 1.08, gy - gt * 0.5 - bl)]
        pen.cv.erase(bridge)
        for f in np.linspace(0.18, 0.85, 4):
            pen.stroke([P(gx - bw, gy - gt * 0.5 - bl * f), P(gx + bw, gy - gt * 0.5 - bl * f)],
                       W0 * 0.36, taper=(0.15, 0.15), wobble=0.1)
        pen.stroke([bridge[0], bridge[3]], W0 * 0.62, taper=(0.05, 0.35), wobble=0.12)
        pen.stroke([bridge[1], bridge[2]], W0 * 0.62, taper=(0.05, 0.35), wobble=0.12)

    tr = max(6.2, gw * 0.44)
    th = gh * rng.uniform(1.16, 1.32)
    off = gw * 0.5 + tr * 0.72
    cap = "cone" if (style == "round" and rng.random() < 0.72) else ("pyramid" if (style == "square" and rng.random() < 0.45) else "crenel")

    # 1. West flanking tower (strictly west/behind central gate block in az=28°)
    if style == "round":
        intact_round_tower(pen, P, rng, gx - off, gy, tr, th, cap=cap, flag=flags)
    else:
        intact_square_tower(pen, P, rng, gx - off, gy, tr * 0.90, th, cap=cap, flag=flags)

    # 2. Central gatehouse block + dark portcullis archway + overhanging machicolated parapet
    bx0, bx1 = gx - gw * 0.56, gx + gw * 0.56
    by0, by1 = gy - gt * 0.5, gy + gt * 0.5
    box(pen, P, bx0, by0, bx1, by1, 0.0, gh)
    # Dark pointed/rounded stone gateway arch
    ah = min(gh * 0.64, gw * 1.02)
    aw = gw * 0.36
    arch = [
        P(gx - aw, by0, 0.0), P(gx - aw, by0, ah * 0.62),
        P(gx - aw * 0.65, by0, ah * 0.90), P(gx, by0, ah),
        P(gx + aw * 0.65, by0, ah * 0.90), P(gx + aw, by0, ah * 0.62),
        P(gx + aw, by0, 0.0),
    ]
    pen.blot(arch, jitter=0.06)
    # Crisp stone arch surround rim
    pen.stroke(arch[1:-1], W0 * 0.68, taper=(0.15, 0.15), wobble=0.06)
    # Raised iron portcullis spikes & cross-bars across upper opening
    for px in np.linspace(gx - aw * 0.68, gx + aw * 0.68, 4):
        p_top = P(px, by0, ah * 0.84)
        p_bot = P(px, by0, ah * 0.44)
        pen.cv.erase([(p_top[0] - 0.55, p_top[1]), (p_top[0] + 0.55, p_top[1]),
                      (p_bot[0] + 0.55, p_bot[1]), (p_bot[0] - 0.55, p_bot[1])])
    for hz in (ah * 0.54, ah * 0.70):
        l0, l1 = P(gx - aw * 0.82, by0, hz), P(gx + aw * 0.82, by0, hz)
        pen.cv.erase([(l0[0], l0[1] - 0.45), (l1[0], l1[1] - 0.45), (l1[0], l1[1] + 0.45), (l0[0], l0[1] + 0.45)])
    # Small guardhouse slit above arch + crenellated overhang
    _draw_slits(pen, P, (bx0, by0), (bx1, by0), gh * 0.82, 2, slit_h=3.4)
    box(pen, P, bx0 - 0.8, by0 - 0.8, bx1 + 0.8, by1 + 0.8, gh, gh + 2.6, w=W0 * 0.78)
    for mx in np.linspace(bx0 - 0.2, bx1 - 2.4, 3):
        box(pen, P, mx, by0 - 0.8, mx + 2.5, by0 + 1.4, gh + 2.6, gh + 5.4, w=W0 * 0.60)

    # 3. East flanking tower (strictly east/in front of central gate block)
    if style == "round":
        intact_round_tower(pen, P, rng, gx + off, gy, tr, th, cap=cap, flag=flags)
    else:
        intact_square_tower(pen, P, rng, gx + off, gy, tr * 0.90, th, cap=cap, flag=flags)


def intact_keep(pen, P, rng, X0, Y0, W, D, H, style="turreted", flag=True):
    """Commanding intact royal castle keep / donjon with corner turrets OR grand gabled roof."""
    W0 = pen.width
    X1, Y1 = X0 + W, Y0 + D
    cx, cy = (X0 + X1) * 0.5, (Y0 + Y1) * 0.5

    # Stepped battered stone plinth base
    box(pen, P, X0 - 2.0, Y0 - 2.0, X1 + 2.0, Y1 + 2.0, 0.0, min(7.0, H * 0.14), inset=0.9)
    # Main keep masonry block
    box(pen, P, X0, Y0, X1, Y1, min(5.5, H * 0.10), H, top=(style == "turreted"))

    # Exterior vertical stone buttress pilasters on lit south wall
    for frac in (0.06, 0.94):
        bx = X0 + W * frac
        box(pen, P, bx - 1.6, Y0 - 1.6, bx + 1.6, Y0 + 0.4, 0.0, H * 0.82, inset=0.4)

    # Horizontal story stringcourses
    for frac in (0.42, 0.76):
        hz = H * frac
        pen.stroke([P(X0, Y0, hz), P(X1, Y0, hz)], W0 * 0.56, taper=(0.04, 0.04), wobble=0.08)
        pen.stroke([P(X1, Y0, hz), P(X1, Y1, hz)], W0 * 0.50, taper=(0.04, 0.04), wobble=0.08)

    # Grand arched entrance portal + paired gothic upper windows
    dw, dh = min(7.0, W * 0.18), min(11.0, H * 0.24)
    pen.blot([P(cx - dw * 0.5, Y0, 0.0), P(cx - dw * 0.5, Y0, dh * 0.66),
              P(cx, Y0, dh), P(cx + dw * 0.5, Y0, dh * 0.66), P(cx + dw * 0.5, Y0, 0.0)], jitter=0.05)
    n_win = max(2, int(W / 14))
    for hz in (H * 0.54, H * 0.84):
        for wx in np.linspace(X0 + 6.5, X1 - 6.5, n_win):
            ww, wh = 2.2, 5.2
            pen.blot([P(wx - ww * 0.5, Y0, hz), P(wx + ww * 0.5, Y0, hz),
                      P(wx + ww * 0.5, Y0, hz + wh * 0.72), P(wx, Y0, hz + wh),
                      P(wx - ww * 0.5, Y0, hz + wh * 0.72)], jitter=0.04)
    for hz in (H * 0.54, H * 0.84):
        _draw_slits(pen, P, (X1, Y0), (X1, Y1), hz + 2.2, max(1, int(D / 14)), slit_h=4.6)
    _draw_masonry_courses(pen, P, (X0, Y0), (X1, Y0), 8.0, H - 4.0, rng, n=max(4, int(W * H / 260)))

    if style == "turreted":
        # Overhanging battlement cornice + inner hipped/gabled roof + 4 corner turrets!
        box(pen, P, X0 - 1.0, Y0 - 1.0, X1 + 1.0, Y1 + 1.0, H, H + 3.2, top=True, w=W0 * 0.82)
        ts = max(4.2, min(6.6, min(W, D) * 0.18))
        th_extra = rng.uniform(8.0, 14.0)
        capped = rng.random() < 0.65

        # Draw roof + 4 corner turrets strictly back-to-front!
        k_items = []
        has_roof = rng.random() < 0.72
        if has_roof:
            k_items.append(((X0 + ts * 0.6, Y0 + ts * 0.6, X1 - ts * 0.6, Y1 - ts * 0.6), ("roof", None)))
        else:
            # Crisp main-parapet merlons along the 4 edges between the corner turrets
            mw = 2.6
            for mx in np.linspace(X0 + ts + 1.5, X1 - ts - mw - 1.5, max(2, int((W - 2 * ts) / 6.0))):
                k_items.append(((mx, Y1 - mw + 0.5, mx + mw, Y1 + 1.0), ("merlon", (mx, Y1 - mw + 0.8))))
                k_items.append(((mx, Y0 - 1.0, mx + mw, Y0 + mw - 0.5), ("merlon", (mx, Y0 - 0.8))))
            for my in np.linspace(Y0 + ts + 1.5, Y1 - ts - mw - 1.5, max(2, int((D - 2 * ts) / 6.0))):
                k_items.append(((X0 - 1.0, my, X0 + mw - 0.5, my + mw), ("merlon", (X0 - 0.8, my))))
                k_items.append(((X1 - mw + 0.5, my, X1 + 1.0, my + mw), ("merlon", (X1 - mw + 0.8, my))))
        for cx_t, cy_t in ((X0 + 0.5, Y1 - 0.5), (X1 - 0.5, Y1 - 0.5),
                           (X0 + 0.5, Y0 + 0.5), (X1 - 0.5, Y0 + 0.5)):
            k_items.append(((cx_t - ts, cy_t - ts, cx_t + ts, cy_t + ts), ("turret", (cx_t, cy_t))))
        for kind_k, arg in sort_boxes(P, k_items):
            if kind_k == "roof":
                rh = min(W, D) * 0.36
                pyramid(pen, P, X0 + ts * 0.7, Y0 + ts * 0.7, X1 - ts * 0.7, Y1 - ts * 0.7, H + 3.2, H + 3.2 + rh)
                if flag:
                    _draw_flag(pen, P, cx, cy, H + 3.2 + rh, h=12.5)
            elif kind_k == "merlon":
                mx, my = arg
                box(pen, P, mx, my, mx + 2.6, my + 2.4, H + 3.2, H + 5.8, w=W0 * 0.58)
            else:
                tx, ty = arg
                if capped:
                    box(pen, P, tx - ts * 0.78, ty - ts * 0.78, tx + ts * 0.78, ty + ts * 0.78,
                        H + 2.0, H + th_extra, top=False, w=W0 * 0.76)
                    pyramid(pen, P, tx - ts * 0.94, ty - ts * 0.94, tx + ts * 0.94, ty + ts * 0.94,
                            H + th_extra, H + th_extra + ts * 1.9)
                else:
                    box(pen, P, tx - ts * 0.78, ty - ts * 0.78, tx + ts * 0.78, ty + ts * 0.78,
                        H + 2.0, H + th_extra, top=True, w=W0 * 0.78)
                    tm = [((mx, my, mx + 2.0, my + 2.0), (mx, my))
                          for mx in (tx - ts * 0.78, tx + ts * 0.78 - 2.0)
                          for my in (ty - ts * 0.78, ty + ts * 0.78 - 2.0)]
                    for mx, my in sort_boxes(P, tm):
                        box(pen, P, mx, my, mx + 2.0, my + 2.0, H + th_extra, H + th_extra + 2.4, w=W0 * 0.56)
        if flag and not has_roof:
            _draw_flag(pen, P, cx, cy, H + 3.2, h=14.0)
    else:
        # Grand gabled palace-keep with solid closed gable masonry, steep tiled roof, chimney & royal pennant
        ridge_x = W >= D
        R = (D if ridge_x else W) * rng.uniform(0.45, 0.58)
        # Solid masonry gable end triangles closing the roof cleanly (no open notch under ridge!)
        if ridge_x:
            g_east = [P(X1, Y0, H), P(X1, Y1, H), P(X1, cy, H + R)]
            pen.cv.erase(g_east)
            pen.hatch(g_east, 82, 1.65, W0 * 0.68, coverage=(0.76, 1.0))
            pen.closed(g_east, W0 * 0.84, wobble=0.14)
        else:
            g_south = [P(X0, Y0, H), P(X1, Y0, H), P(cx, Y0, H + R)]
            pen.cv.erase(g_south)
            pen.closed(g_south, W0 * 0.84, wobble=0.14)
        st = dict(kind="house", W=W, D=D, H=H, R=R, ridge_x=ridge_x, floors=0,
                  timber=False, plinth=False, quoins=False, arch_door=False,
                  dormer=ridge_x and W > 34, chimney=False, smoke=False, leanto=False,
                  cross=False, roof="tile", win=(3.0, 4.5))
        roof_only = [f for f in house_faces(X0, Y0, W, D, H, R, ridge_x, o=2.0) if f["kind"] == "roof"]
        draw_faces(pen, P, roof_only, st)
        # Tall slender stone chimney + central roof pennant
        ch_x = X0 + W * 0.24 if ridge_x else X0 + W * 0.62
        ch_y = Y0 + D * 0.36 if ridge_x else Y0 + D * 0.28
        box(pen, P, ch_x - 2.1, ch_y - 2.1, ch_x + 2.1, ch_y + 2.1, H + R * 0.58, H + R + 5.2)
        if flag:
            _draw_flag(pen, P, cx, cy, H + R, h=12.5)


# --------------------------------------------------------------------------- townhouses

def _draw_crow_step_gable(pen, P, a, b, H, R, n_steps=4):
    """Add medieval stepped masonry gable merlons along a front gable wall `a -> b`."""
    W0 = pen.width
    a, b = np.asarray(a, float), np.asarray(b, float)
    steps = []
    for i in range(n_steps):
        f0 = i / n_steps * 0.5
        f1 = (i + 1) / n_steps * 0.5
        h_s = H + R * (f1 * 2.0)
        g0 = a + (b - a) * f0
        g1 = a + (b - a) * f1
        steps += [P(g0[0], g0[1], H + R * (f0 * 2.0)), P(g0[0], g0[1], h_s), P(g1[0], g1[1], h_s)]
    # Mirror down right half
    for i in range(n_steps - 1, -1, -1):
        f0 = 1.0 - (i + 1) / n_steps * 0.5
        f1 = 1.0 - i / n_steps * 0.5
        h_s = H + R * ((1.0 - f0) * 2.0)
        g0 = a + (b - a) * f0
        g1 = a + (b - a) * f1
        steps += [P(g0[0], g0[1], h_s), P(g1[0], g1[1], h_s), P(g1[0], g1[1], H + R * ((1.0 - f1) * 2.0))]
    pen.stroke(steps, W0 * 0.78, taper=(0.05, 0.05), wobble=0.08)


def plan_townhouse(rng, scale=1.0, force_ridge_x=None):
    """Plan a tall, narrow medieval urban townhouse (2-3 storeys, jetty/crow-step/dormer/chimney)."""
    floors = int(rng.choice([2, 3], p=[0.52, 0.48]))
    ridge_x = (rng.random() < 0.42) if force_ridge_x is None else force_ridge_x
    if ridge_x:
        W = rng.uniform(24, 35) * scale
        D = rng.uniform(17, 23) * scale
    else:
        W = rng.uniform(17, 24) * scale
        D = rng.uniform(24, 34) * scale
    H = (rng.uniform(20, 26) if floors == 2 else rng.uniform(27, 34)) * scale
    span = (D if ridge_x else W) * 0.5
    R = span * rng.uniform(1.12, 1.48)
    timber = rng.random() < 0.58
    jetty = timber and rng.random() < 0.56 and scale >= 0.75
    crow_step = (not timber) and (not ridge_x) and rng.random() < 0.52
    return dict(
        kind="townhouse", W=W, D=D, H=H, R=R, ridge_x=ridge_x, floors=floors,
        timber=timber, jetty=jetty, crow_step=crow_step,
        plinth=(not timber), quoins=(not timber) and rng.random() < 0.65,
        arch_door=rng.random() < 0.55,
        dormer=ridge_x and rng.random() < 0.58 and scale >= 0.75,
        chimney=rng.random() < 0.80,
        smoke=rng.random() < 0.16,
        leanto=False, cross=rng.random() < 0.68, roof="tile",
        win=(rng.uniform(2.8, 3.6) * scale, rng.uniform(4.2, 5.2) * scale),
        scale=scale,
    )


def draw_townhouse(pen, P, X0, Y0, pl):
    """Draw a tall medieval urban townhouse with optional overhanging jetty, crow-step gable & shopfront."""
    rng, W0 = pen.rng, pen.width
    W, D, H, R = pl["W"], pl["D"], pl["H"], pl["R"]
    sc = pl.get("scale", 1.0)

    if pl.get("jetty"):
        # Stone/plaster recessed ground floor + overhanging half-timbered upper stories!
        h_g = H * (0.42 if pl["floors"] == 2 else 0.32)
        ov = 1.6 * sc
        gx0, gy0, gx1, gy1 = X0 + ov, Y0 + ov, X0 + W - ov, Y0 + D - ov
        box(pen, P, gx0, gy0, gx1, gy1, 0.0, h_g, top=False)
        # Ground-floor arched shop door & shuttered window on lit south facade
        gw = gx1 - gx0
        ds = gx0 + gw * rng.uniform(0.32, 0.68)
        dw = min(5.4 * sc, gw * 0.34)
        dh = min(8.2 * sc, h_g * 0.78)
        pen.blot([P(ds - dw * 0.5, gy0, 0.0), P(ds - dw * 0.5, gy0, dh * 0.70),
                  P(ds, gy0, dh), P(ds + dw * 0.5, gy0, dh * 0.70), P(ds + dw * 0.5, gy0, 0.0)], jitter=0.05)
        # Corbel timber brackets supporting the overhanging jetty beam
        for bx in np.linspace(gx0 + 1.2, gx1 - 1.2, max(3, int(gw / 6))):
            pen.stroke([P(bx, gy0, h_g - 2.4 * sc), P(bx, Y0, h_g)], W0 * 0.52, taper=(0.15, 0.05), wobble=0.04)
        # Draw upper stories cleanly with elevated projection wrapper so windows/timbers sit at z = h_g!
        P_up = lambda x, y, h=0.0: P(x, y, h + h_g)
        P_up.az, P_up.ca, P_up.sa, P_up.ce, P_up.se = P.az, P.ca, P.sa, P.ce, P.se
        _draw_elevated_house_body(pen, P_up, X0, Y0, W, D, H - h_g, R, pl)
    else:
        # Full-height 2-3 storey townhouse with extra third-floor window band when floors == 3
        draw_faces(pen, P, house_faces(X0, Y0, W, D, H, R, pl["ridge_x"], o=1.5 * sc),
                   dict(pl, floors=min(2, pl["floors"])))
        if pl["floors"] >= 3:
            # Crisp 3rd-floor row of windows tucked just below the eaves on the sunlit south face
            ww, wh = pl["win"]
            hh = H * 0.74
            n = max(1, int((W - 4.0) / (9.5 * sc)))
            for s in np.linspace(X0 + 3.5 * sc, X0 + W - 3.5 * sc, n):
                pen.blot([P(s - ww * 0.45, Y0, hh), P(s + ww * 0.45, Y0, hh),
                          P(s + ww * 0.45, Y0, hh + wh * 0.9), P(s - ww * 0.45, Y0, hh + wh * 0.9)], jitter=0.05)

    if pl.get("crow_step") and not pl["ridge_x"]:
        _draw_crow_step_gable(pen, P, (X0, Y0), (X0 + W, Y0), H, R, n_steps=4)

    if pl.get("chimney"):
        if pl["ridge_x"]:
            cx, cy = X0 + W * rng.uniform(0.18, 0.82), Y0 + D * rng.uniform(0.32, 0.44)
            hb = H + R * (1.0 - abs(cy - (Y0 + D * 0.5)) / (D * 0.5))
        else:
            cx, cy = X0 + W * rng.uniform(0.56, 0.68), Y0 + D * rng.uniform(0.22, 0.78)
            hb = H + R * (1.0 - abs(cx - (X0 + W * 0.5)) / (W * 0.5))
        cw = 1.9 * sc
        ctop = H + R + rng.uniform(3.2, 6.4) * sc
        box(pen, P, cx - cw, cy - cw, cx + cw, cy + cw, hb - 2.2 * sc, ctop)
        box(pen, P, cx - cw * 1.25, cy - cw * 1.25, cx + cw * 1.25, cy + cw * 1.25, ctop, ctop + 1.2 * sc)


def _draw_elevated_house_body(pen, P_up, X0, Y0, W, D, H, R, pl):
    """Draw the cantilevered upper half-timbered storeys + roof of a jettied townhouse (no ground door!)."""
    rng, W0 = pen.rng, pen.width
    sc = pl.get("scale", 1.0)
    faces = house_faces(X0, Y0, W, D, H, R, pl["ridge_x"], o=1.5 * sc)
    # Use draw_faces with floors=-1 (no door!) then stamp neat timber-framed upper window rows
    draw_faces(pen, P_up, faces, dict(pl, floors=-1, dormer=pl.get("dormer", False)))
    # Crisp upper-story windows on lit south wall
    rows = [H * 0.34] if pl["floors"] <= 2 else [H * 0.16, H * 0.58]
    ww, wh = pl["win"]
    n = max(1, int((W - 4.0 * sc) / (9.0 * sc)))
    for hh in rows:
        for s in np.linspace(X0 + 3.8 * sc, X0 + W - 3.8 * sc, n):
            pen.blot([P_up(s - ww * 0.45, Y0, hh), P_up(s + ww * 0.45, Y0, hh),
                      P_up(s + ww * 0.45, Y0, hh + wh * 0.92), P_up(s - ww * 0.45, Y0, hh + wh * 0.92)], jitter=0.05)
            pen.stroke([P_up(s - ww * 0.6, Y0, hh - 0.3), P_up(s + ww * 0.6, Y0, hh - 0.3)],
                       W0 * 0.40, taper=(0.1, 0.1), wobble=0)


def townhouse(pen, P, rng):
    """Modular medieval urban townhouses: tall single, attached row-house pair/triple, or L-corner inn."""
    mode = str(rng.choice(["single", "row_pair", "row_triple", "corner_l"], p=[0.36, 0.28, 0.18, 0.18]))
    items = []
    if mode == "single":
        pl = plan_townhouse(rng, scale=1.0)
        X0, Y0 = -pl["W"] * 0.5, -pl["D"] * 0.5
        items.append(((X0, Y0, X0 + pl["W"], Y0 + pl["D"]), (X0, Y0, pl)))
    elif mode in ("row_pair", "row_triple"):
        n_row = 2 if mode == "row_pair" else 3
        along_x = rng.random() < 0.72
        pls = [plan_townhouse(rng, scale=0.92 if n_row == 3 else 0.96, force_ridge_x=(not along_x if rng.random() < 0.75 else None))
               for _ in range(n_row)]
        if along_x:
            tot_w = sum(p["W"] for p in pls)
            cur_x = -tot_w * 0.5
            for pl in pls:
                y0 = -pl["D"] * 0.5 + rng.uniform(-1.5, 1.5)
                items.append(((cur_x, y0, cur_x + pl["W"], y0 + pl["D"]), (cur_x, y0, pl)))
                cur_x += pl["W"] - 0.2  # flush party wall!
        else:
            tot_d = sum(p["D"] for p in pls)
            cur_y = -tot_d * 0.5
            for pl in pls:
                x0 = -pl["W"] * 0.5 + rng.uniform(-1.5, 1.5)
                items.append(((x0, cur_y, x0 + pl["W"], cur_y + pl["D"]), (x0, cur_y, pl)))
                cur_y += pl["D"] - 0.2
    else:  # L-shaped corner townhouse / urban tavern
        pl1 = plan_townhouse(rng, scale=0.96, force_ridge_x=True)
        pl2 = plan_townhouse(rng, scale=0.92, force_ridge_x=False)
        pl2["H"] = pl1["H"] * rng.uniform(0.84, 0.96)
        pl2["jetty"] = False
        X1, Y1 = -pl1["W"] * 0.5, 0.0
        # Wing 2 projects south from west OR east end of wing 1 (strictly disjoint footprints!)
        west_wing = rng.random() < 0.55
        X2 = X1 if west_wing else X1 + pl1["W"] - pl2["W"]
        Y2 = Y1 - pl2["D"] * 0.88
        items.append(((X1, Y1, X1 + pl1["W"], Y1 + pl1["D"]), (X1, Y1, pl1)))
        items.append(((X2, Y2, X2 + pl2["W"], Y1), (X2, Y2, pl2)))

    for X0, Y0, pl in sort_boxes(P, items):
        draw_townhouse(pen, P, X0, Y0, pl)


# --------------------------------------------------------------------------- town parts (civic & urban landmarks)

def _draw_clocktower(pen, P, rng, tx0, tx1, ty0, ty1, z0, th, spire_h, flag=True):
    """Civic stone guildhall clocktower with clock dial, twin belfry arches, steep spire & banner flag."""
    W0 = pen.width
    tx, ty = (tx0 + tx1) * 0.5, (ty0 + ty1) * 0.5
    tw, td = tx1 - tx0, ty1 - ty0
    if z0 <= 0.5:
        box(pen, P, tx0 - 1.4, ty0 - 1.4, tx1 + 1.4, ty1 + 1.4, 0.0, th * 0.22)
    box(pen, P, tx0, ty0, tx1, ty1, z0, th, top=False)
    for frac in (0.60, 0.96):
        hz = z0 + (th - z0) * frac
        pen.stroke([P(tx0, ty0, hz), P(tx1, ty0, hz)], W0 * 0.54, taper=(0.04, 0.04), wobble=0.06)
        pen.stroke([P(tx1, ty0, hz), P(tx1, ty1, hz)], W0 * 0.48, taper=(0.04, 0.04), wobble=0.06)
    # Circular civic clock dial with hands on sunlit south face
    oc_z = z0 + (th - z0) * 0.44
    ox, oy = P(tx, ty0, oc_z)
    oc_r = min(3.6, tw * 0.27)
    ring = ellipse_pts(ox, oy, oc_r, oc_r * 1.05, 16)
    pen.cv.erase(ring)
    pen.closed(ring, W0 * 0.62, wobble=0.04)
    pen.stroke([(ox, oy), (ox, oy - oc_r * 0.72)], W0 * 0.45, taper=(0, 0), wobble=0)
    pen.stroke([(ox, oy), (ox + oc_r * 0.56, oy + oc_r * 0.22)], W0 * 0.45, taper=(0, 0), wobble=0)
    # Paired belfry arches near top
    bz0 = z0 + (th - z0) * 0.68
    bh = (th - z0) * 0.22
    bw = min(2.2, tw * 0.17)
    for off in (-tw * 0.20, tw * 0.20):
        bx = tx + off
        pen.blot([P(bx - bw * 0.5, ty0, bz0), P(bx + bw * 0.5, ty0, bz0),
                  P(bx + bw * 0.5, ty0, bz0 + bh * 0.72), P(bx, ty0, bz0 + bh * 1.06),
                  P(bx - bw * 0.5, ty0, bz0 + bh * 0.72)], jitter=0.04)
    ebw = min(2.4, td * 0.20)
    pen.blot([P(tx1, ty - ebw * 0.5, bz0), P(tx1, ty + ebw * 0.5, bz0),
              P(tx1, ty + ebw * 0.5, bz0 + bh * 0.72), P(tx1, ty, bz0 + bh * 1.06),
              P(tx1, ty - ebw * 0.5, bz0 + bh * 0.72)], jitter=0.04)
    apex = th + spire_h
    pyramid(pen, P, tx0 - 1.2, ty0 - 1.2, tx1 + 1.2, ty1 + 1.2, th, apex)
    if flag:
        _draw_flag(pen, P, tx, ty, apex, h=9.5)


def town_guildhall(pen, P, rng):
    """Grand medieval guildhall / town hall with ground arcade, dormers & soaring clocktower spire."""
    W0 = pen.width
    W = rng.uniform(54, 68)
    D = rng.uniform(26, 32)
    H = rng.uniform(22, 26)
    R = (D * 0.5) * rng.uniform(1.15, 1.38)
    X0, Y0 = -W * 0.5, -D * 0.5
    X1, Y1 = X0 + W, Y0 + D

    # Rear clocktower rising behind or centrally out of the guildhall
    central_tower = rng.random() < 0.58
    ts = rng.uniform(7.5, 9.2)
    if not central_tower:
        # Tower attached to the west/north-west side (strictly behind main hall)
        tx0, tx1 = X0 - ts * 1.45, X0 + ts * 0.55
        ty0, ty1 = Y0 + D * 0.5 - ts, Y0 + D * 0.5 + ts
        th = H + rng.uniform(32, 46)
        _draw_clocktower(pen, P, rng, tx0, tx1, ty0, ty1, 0.0, th, rng.uniform(22, 32), flag=True)

    # Main 2-storey stone civic hall
    pl = dict(kind="house", W=W, D=D, H=H, R=R, ridge_x=True, floors=2,
              timber=False, plinth=True, quoins=True, arch_door=True,
              dormer=False, chimney=False, smoke=False, leanto=False, cross=True,
              roof="tile", win=(3.2, 5.0))
    # Draw base hall with ground floor suppressed so we can draw a crisp stone arcade colonnade!
    draw_faces(pen, P, house_faces(X0, Y0, W, D, H, R, True, o=1.8), dict(pl, floors=0))
    # Horizontal stringcourse separating ground arcade & upper guild chamber
    h_arc = H * 0.44
    pen.stroke([P(X0, Y0, h_arc), P(X1, Y0, h_arc)], W0 * 0.62, taper=(0.04, 0.04), wobble=0.08)
    pen.stroke([P(X1, Y0, h_arc), P(X1, Y1, h_arc)], W0 * 0.54, taper=(0.04, 0.04), wobble=0.08)
    # Arched ground-floor arcade bays along sunlit south wall
    n_bays = int(rng.integers(4, 6))
    bay_w = (W - 8.0) / n_bays
    for bi in range(n_bays):
        bx = X0 + 4.0 + (bi + 0.5) * bay_w
        aw = bay_w * 0.36
        ah = h_arc * 0.80
        pen.blot([P(bx - aw, Y0, 0.0), P(bx - aw, Y0, ah * 0.62), P(bx, Y0, ah),
                  P(bx + aw, Y0, ah * 0.62), P(bx + aw, Y0, 0.0)], jitter=0.05)
    # Tall upper-floor arched guildhall windows
    for bi in range(n_bays):
        bx = X0 + 4.0 + (bi + 0.5) * bay_w
        ww, wh = 3.2, 6.2
        hz = h_arc + 3.2
        pen.blot([P(bx - ww * 0.5, Y0, hz), P(bx + ww * 0.5, Y0, hz),
                  P(bx + ww * 0.5, Y0, hz + wh * 0.72), P(bx, Y0, hz + wh),
                  P(bx - ww * 0.5, Y0, hz + wh * 0.72)], jitter=0.04)
        pen.stroke([P(bx - ww * 0.7, Y0, hz - 0.3), P(bx + ww * 0.7, Y0, hz - 0.3)],
                   W0 * 0.42, taper=(0.1, 0.1), wobble=0)

    # Twin small chimneys on roof ends
    for fx in (0.14, 0.86):
        box(pen, P, X0 + W * fx - 2.0, Y0 + D * 0.38 - 2.0, X0 + W * fx + 2.0, Y0 + D * 0.38 + 2.0,
            H + R * 0.55, H + R + 4.5)

    # Central soaring clocktower / belfry rising right out of the hall ridge!
    if central_tower:
        z_b = H + R * 0.35
        th = z_b + rng.uniform(26, 36)
        sp_h = rng.uniform(20, 30)
        _draw_clocktower(pen, P, rng, -ts * 0.88, ts * 0.88, -ts * 0.88, ts * 0.88, z_b, th, sp_h, flag=True)


def _draw_market_stall(pen, P, rng, X0, Y0, W, D, ridge_x=True):
    """Small wooden medieval market stall with timber counter & striped canvas awning."""
    W0 = pen.width
    X1, Y1 = X0 + W, Y0 + D
    hc = 3.6
    ht_lo, ht_hi = 8.4, 11.2
    # Solid wooden counter chest at bottom
    box(pen, P, X0 + 0.6, Y0 + 0.6, X1 - 0.6, Y1 - 0.6, 0.0, hc, w=W0 * 0.68)
    # 4 corner awning posts
    for px, py, hz in ((X0 + 0.8, Y1 - 0.8, ht_hi), (X1 - 0.8, Y1 - 0.8, ht_hi),
                       (X0 + 0.8, Y0 + 0.8, ht_lo), (X1 - 0.8, Y0 + 0.8, ht_lo)):
        pen.stroke([P(px, py, hc), P(px, py, hz)], W0 * 0.56, taper=(0.0, 0.05), wobble=0.04)
    # Sloped striped canvas canopy sloping forward toward south
    ov = 1.2
    c0, c1 = (X0 - ov, Y0 - ov, ht_lo), (X1 + ov, Y0 - ov, ht_lo)
    c2, c3 = (X1 + ov, Y1 + ov, ht_hi), (X0 - ov, Y1 + ov, ht_hi)
    poly = [P(*c0), P(*c1), P(*c2), P(*c3)]
    pen.cv.erase(poly)
    # Crisp alternating awning stripes
    n_st = max(3, int(W / 4.2))
    for k in range(n_st):
        if k % 2 == 0:
            u0, u1 = k / n_st, (k + 1) / n_st
            sq = [
                P(c0[0] + (c1[0] - c0[0]) * u0, c0[1], ht_lo),
                P(c0[0] + (c1[0] - c0[0]) * u1, c0[1], ht_lo),
                P(c3[0] + (c2[0] - c3[0]) * u1, c3[1], ht_hi),
                P(c3[0] + (c2[0] - c3[0]) * u0, c3[1], ht_hi),
            ]
            pen.hatch(sq, 72, 1.5, W0 * 0.48, coverage=(0.75, 1.0))
    pen.closed(poly, W0 * 0.78, wobble=0.10)
    # Scalloped valance fringe along front awning edge
    pen.stroke([poly[0], poly[1]], W0 * 1.05, taper=(0.05, 0.05), wobble=0.16)


def _draw_fountain(pen, P, rng, X, Y, R=10.5):
    """Tiered carved stone plaza fountain with water basin, central column & tiny arcing jets."""
    W0 = pen.width
    # Outer low stone basin rim
    cylinder(pen, P, X, Y, R, 0.0, 4.4, w=W0 * 0.85)
    cx, cy = P(X, Y, 4.4)
    water = ellipse_pts(cx, cy, R * 0.78, R * 0.78 * P.se, 20)
    pen.cv.erase(water)
    pen.closed(water, W0 * 0.56, wobble=0.06)
    # Subtle water ripples
    for rr in (0.45, 0.65):
        pen.stroke(ellipse_pts(cx, cy, R * rr, R * rr * P.se, 14, 0.3, 2.4), W0 * 0.30, taper=(0.3, 0.3), wobble=0.06)
    # Central stone pedestal + upper bowl + finial
    cylinder(pen, P, X, Y, R * 0.24, 4.0, 12.0, w=W0 * 0.72)
    cylinder(pen, P, X, Y, R * 0.48, 11.5, 13.8, w=W0 * 0.72)
    cylinder(pen, P, X, Y, R * 0.16, 13.8, 18.2, w=W0 * 0.64)
    # Delicate curving water spray arcs dropping from upper tier into lower basin
    top_pt = P(X, Y, 17.5)
    for ang in (-0.8, 0.2, 2.3, 3.6):
        land = P(X + R * 0.56 * math.cos(ang), Y + R * 0.56 * math.sin(ang), 4.6)
        mid = ((top_pt[0] + land[0]) * 0.5, min(top_pt[1], land[1]) - 3.2)
        pen.stroke([top_pt, mid, land], W0 * 0.34, taper=(0.25, 0.55), smooth=2, wobble=0.08)


def town_market(pen, P, rng):
    """Bustling medieval market square: flagstone plaza, central fountain/well, 4-6 striped stalls & cargo crates."""
    W0 = pen.width
    R_sq = rng.uniform(48, 56)
    # Subtle broken stone paving border outlining the square
    sq_pts = [P(-R_sq, -R_sq * 0.82), P(R_sq, -R_sq * 0.82), P(R_sq, R_sq * 0.82), P(-R_sq, R_sq * 0.82)]
    for i in range(4):
        a, b = sq_pts[i], sq_pts[(i + 1) % 4]
        mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        pen.stroke([a, mid], W0 * 0.42, taper=(0.35, 0.45), wobble=0.25)
        pen.stroke([mid, b], W0 * 0.42, taper=(0.45, 0.35), wobble=0.25)
    for _ in range(12):
        px, py = rng.uniform(-R_sq * 0.78, R_sq * 0.78), rng.uniform(-R_sq * 0.60, R_sq * 0.60)
        pen.stroke([P(px, py), P(px + rng.uniform(3.5, 7.0), py)], W0 * 0.30, taper=(0.35, 0.35), wobble=0.06)

    items, boxes = [], []
    if rng.random() < 0.70:
        f_r = rng.uniform(9.2, 11.4)
        fbox = (-f_r - 1.5, -f_r - 1.5, f_r + 1.5, f_r + 1.5)
        items.append((fbox, ("fountain", (0.0, 0.0, f_r))))
        boxes.append((-f_r - 3.5, -f_r - 3.5, f_r + 3.5, f_r + 3.5))
        c_rad = f_r + 3.5
    else:
        wbox = (-6.8, -5.2, 6.8, 5.2)
        items.append((wbox, ("well", (0.0, 0.0))))
        boxes.append((-8.5, -7.2, 8.5, 7.2))
        c_rad = 9.0

    # Guaranteed 4-6 striped market stalls framing the central landmark cleanly!
    n_stalls = int(rng.integers(5, 7))
    angs = np.linspace(0.0, TAU, n_stalls, endpoint=False) + rng.uniform(0.12, 0.48)
    for ang in angs:
        sw, sd = rng.uniform(14.0, 18.5), rng.uniform(9.5, 11.8)
        for dist_mul in (0.64, 0.74, 0.82):
            dist_x = max(c_rad + sw * 0.5 + 3.5, R_sq * dist_mul)
            dist_y = max(c_rad + sd * 0.5 + 3.5, R_sq * 0.82 * dist_mul)
            cx = dist_x * math.cos(ang)
            cy = dist_y * math.sin(ang)
            x0, y0 = cx - sw * 0.5, cy - sd * 0.5
            sbox = (x0 - 1.0, y0 - 1.0, x0 + sw + 1.0, y0 + sd + 1.0)
            if not _overlaps(sbox, boxes, 1.5):
                boxes.append(sbox)
                items.append((sbox, ("stall", (x0, y0, sw, sd))))
                break

    # 2-4 wooden goods crates tucked neatly beside stalls
    for _ in range(int(rng.integers(2, 5))):
        for _ in range(25):
            cx, cy = rng.uniform(-R_sq * 0.68, R_sq * 0.68), rng.uniform(-R_sq * 0.52, R_sq * 0.52)
            cbox = (cx - 2.4, cy - 2.2, cx + 2.4, cy + 2.2)
            if not _overlaps(cbox, boxes, 1.0):
                boxes.append(cbox)
                items.append((cbox, ("crate", (cx - 1.8, cy - 1.8, cx + 1.8, cy + 1.8, rng.uniform(3.2, 4.6)))))
                break

    for kind, args in sort_boxes(P, items):
        if kind == "fountain":
            _draw_fountain(pen, P, rng, *args)
        elif kind == "well":
            well(pen, P, *args)
        elif kind == "crate":
            box(pen, P, args[0], args[1], args[2], args[3], 0.0, args[4], w=W0 * 0.58)
        else:
            _draw_market_stall(pen, P, rng, *args)


def town_cathedral(pen, P, rng, scale=1.0):
    """Grand urban gothic cathedral / abbey with twin west spires, cruciform transepts & crossing fleche."""
    W0 = pen.width
    W = rng.uniform(62, 74) * scale
    D = rng.uniform(22, 26) * scale
    H = rng.uniform(21, 25) * scale
    R = (D * 0.5) * rng.uniform(1.16, 1.38)
    X0, Y0 = -W * 0.44, -D * 0.5
    ty = Y0 + D * 0.5

    pl = dict(kind="church", W=W, D=D, H=H, R=R, ridge_x=True, floors=1,
              timber=False, plinth=True, quoins=True, arch_door=True, lancet=True,
              dormer=False, chimney=False, smoke=False, leanto=False, cross=False,
              roof="tile", win=(2.8 * scale, 8.2 * scale))

    tr_w = D * rng.uniform(0.76, 0.88)
    tr_x0 = X0 + W * 0.40
    tr_proj = D * rng.uniform(0.42, 0.56)
    tr_st = dict(pl, W=tr_w, D=tr_proj + D * 0.5, H=H * 0.96, R=(tr_w * 0.5) * 1.18,
                 ridge_x=False, floors=1, win=(2.4 * scale, 7.2 * scale))

    # 1. Far north transept wing
    draw_faces(pen, P, house_faces(tr_x0, ty, tr_w, D * 0.5 + tr_proj, tr_st["H"], tr_st["R"], False, o=1.4 * scale),
               dict(tr_st, floors=0))

    # 2. Twin gothic west facade bell-towers (north-west back tower first, then south-west front tower after nave!)
    ts = rng.uniform(6.4, 7.6) * scale
    th = rng.uniform(54, 68) * scale
    sp_h = rng.uniform(28, 40) * scale
    tx0, tx1 = X0 - ts * 1.4, X0 + ts * 0.6
    # NW back spire (strictly north & west of nave!)
    _draw_bell_tower(pen, P, rng, tx0, tx1, Y0 + D - ts * 1.1, Y0 + D + ts * 0.9, 0.0, th, sp_h)

    # 3. Main buttressed gothic nave + east choir cross
    draw_house(pen, P, X0, Y0, pl)
    for frac in (0.16, 0.28, 0.72, 0.88):
        bx = X0 + W * frac
        box(pen, P, bx - 1.3 * scale, Y0 - 2.3 * scale, bx + 1.3 * scale, Y0 + 0.4 * scale, 0, H * 0.78, inset=0.45 * scale)
    _draw_cross(pen, P, X0 + W, ty, H + R, h=7.0 * scale, w_scale=0.62)

    # 4. Near south-west twin bell tower (flanking the west entrance portal!)
    _draw_bell_tower(pen, P, rng, tx0, tx1, Y0 - ts * 0.9, Y0 + ts * 1.1, 0.0, th, sp_h)

    # 5. Near south transept with circular gothic rose window
    draw_faces(pen, P, house_faces(tr_x0, Y0 - tr_proj, tr_w, tr_proj + D * 0.5, tr_st["H"], tr_st["R"], False, o=1.4 * scale),
               tr_st)
    gx, gy = P(tr_x0 + tr_w * 0.5, Y0 - tr_proj, tr_st["H"] + tr_st["R"] * 0.36)
    rr = 3.0 * scale
    ring = ellipse_pts(gx, gy, rr, rr * 1.05, 16)
    pen.cv.erase(ring)
    pen.closed(ring, W0 * 0.58, wobble=0.05)
    pen.stroke([(gx, gy - rr * 0.8), (gx, gy + rr * 0.8)], W0 * 0.32, taper=(0, 0), wobble=0)
    pen.stroke([(gx - rr * 0.8, gy), (gx + rr * 0.8, gy)], W0 * 0.32, taper=(0, 0), wobble=0)

    # 6. Slender crossing spirelet (fleche) rising at the transept intersection
    cx = tr_x0 + tr_w * 0.5
    z_c = H + R * 0.45
    cs = ts * 0.52
    fleche_h = rng.uniform(22, 32) * scale
    pyramid(pen, P, cx - cs, ty - cs, cx + cs, ty + cs, z_c, z_c + fleche_h)
    _draw_cross(pen, P, cx, ty, z_c + fleche_h, h=6.5 * scale, w_scale=0.56)


def town_manor(pen, P, rng):
    """Wealthy urban merchant manor or U/L-shaped coaching inn with courtyard wall & arched gate."""
    W = rng.uniform(46, 58)
    D = rng.uniform(20, 24)
    H = rng.uniform(20, 24)
    R = (D * 0.5) * rng.uniform(1.05, 1.30)
    main_pl = dict(kind="house", W=W, D=D, H=H, R=R, ridge_x=True, floors=2,
                   timber=(rng.random() < 0.45), plinth=True, quoins=True, arch_door=True,
                   dormer=True, chimney=True, smoke=False, leanto=False, cross=True,
                   roof="tile", win=(3.2, 4.8))
    mx0, my0 = -W * 0.5, 6.0

    u_shape = rng.random() < 0.55
    w_w = rng.uniform(17, 21)
    w_d = rng.uniform(22, 28)
    w_h = H * rng.uniform(0.78, 0.90)
    w_pl = dict(main_pl, W=w_w, D=w_d, H=w_h, R=(w_w * 0.5) * 1.15, ridge_x=False,
                dormer=False, chimney=(rng.random() < 0.5))

    items = [((mx0, my0, mx0 + W, my0 + D), ("house", (mx0, my0, main_pl)))]
    if u_shape:
        # Both west and east wings projecting south to frame a courtyard
        items.append(((mx0, my0 - w_d, mx0 + w_w, my0), ("house", (mx0, my0 - w_d, w_pl))))
        items.append(((mx0 + W - w_w, my0 - w_d, mx0 + W, my0), ("house", (mx0 + W - w_w, my0 - w_d, w_pl))))
        # Low courtyard wall + arched carriage entrance connecting the front ends of the two wings!
        wy_f = my0 - w_d + 2.0
        items.append(((mx0 + w_w, wy_f - 2.0, mx0 + W - w_w, wy_f + 2.0), ("cwall", (mx0 + w_w, mx0 + W - w_w, wy_f))))
    else:
        # L-shaped manor with one projecting wing + courtyard well
        west_w = rng.random() < 0.55
        wx0 = mx0 if west_w else mx0 + W - w_w
        items.append(((wx0, my0 - w_d, wx0 + w_w, my0), ("house", (wx0, my0 - w_d, w_pl))))
        well_x = mx0 + W * (0.72 if west_w else 0.28)
        well_y = my0 - w_d * 0.48
        items.append(((well_x - 6, well_y - 5, well_x + 6, well_y + 5), ("well", (well_x, well_y))))

    for kind, args in sort_boxes(P, items):
        if kind == "house":
            draw_house(pen, P, *args)
        elif kind == "well":
            well(pen, P, *args)
        else:
            xa, xb, yf = args
            if xb - xa > 8.0:
                box(pen, P, xa, yf - 1.5, xb, yf + 1.5, 0.0, 9.5)
                cx = (xa + xb) * 0.5
                aw = min(5.2, (xb - xa) * 0.34)
                pen.blot([P(cx - aw, yf - 1.5, 0.0), P(cx - aw, yf - 1.5, 5.2),
                          P(cx, yf - 1.5, 7.6), P(cx + aw, yf - 1.5, 5.2), P(cx + aw, yf - 1.5, 0.0)], jitter=0.05)


def _draw_windmill_sails(pen, P, rng, hx, hy, hz, L=24.0):
    """Draw 4 classic angled lattice canvas windmill sails mounted at hub `(hx, hy, hz)` facing south."""
    W0 = pen.width
    hub = P(hx, hy, hz)
    base_ang = rng.uniform(0.32, 0.92)
    # Sort blades back-to-front visually (simple stroke order)
    for k in range(4):
        ang = base_ang + k * (math.pi * 0.5)
        # Blade direction in the vertical front plane (x, z)
        dx, dz = math.cos(ang), math.sin(ang)
        px, pz = -dz, dx  # perpendicular sail-width vector
        p_tip = P(hx + dx * L, hy, hz + dz * L)
        s0, s1 = L * 0.20, L * 0.96
        sw = L * 0.22
        sail = [
            P(hx + dx * s0, hy, hz + dz * s0),
            P(hx + dx * s1, hy, hz + dz * s1),
            P(hx + dx * s1 + px * sw, hy, hz + dz * s1 + pz * sw),
            P(hx + dx * s0 + px * sw * 0.85, hy, hz + dz * s0 + pz * sw * 0.85),
        ]
        pen.cv.erase(sail)
        # Wooden lattice cross-slats across the sail
        for f in np.linspace(0.15, 0.88, 6):
            sf = s0 + (s1 - s0) * f
            q0 = P(hx + dx * sf, hy, hz + dz * sf)
            q1 = P(hx + dx * sf + px * sw * (0.85 + 0.15 * f), hy, hz + dz * sf + pz * sw * (0.85 + 0.15 * f))
            pen.stroke([q0, q1], W0 * 0.36, taper=(0.05, 0.05), wobble=0.03)
        # Mid longitudinal lattice rail
        m0 = P(hx + dx * s0 + px * sw * 0.45, hy, hz + dz * s0 + pz * sw * 0.45)
        m1 = P(hx + dx * s1 + px * sw * 0.50, hy, hz + dz * s1 + pz * sw * 0.50)
        pen.stroke([m0, m1], W0 * 0.34, taper=(0.05, 0.05), wobble=0.03)
        pen.closed(sail, W0 * 0.62, wobble=0.06)
        # Bold timber main spar from hub to tip
        pen.stroke([hub, p_tip], W0 * 1.05, taper=(0.0, 0.12), wobble=0.04)
    # Dark hub boss cap
    pen.blot(ellipse_pts(hub[0], hub[1], 2.2, 2.2, 10), jitter=0.04)


def town_windmill(pen, P, rng, scale=1.0):
    """Stone tower windmill or timber post-mill with 4 sweeping lattice canvas sails."""
    W0 = pen.width
    if rng.random() < 0.68:
        # Classic stone tower windmill
        r_b = rng.uniform(11.5, 14.0) * scale
        r_t = r_b * 0.78
        H = rng.uniform(34, 44) * scale
        # Stepped stone base drum + upper tapered tower shaft
        cylinder(pen, P, 0.0, 0.0, r_b * 1.12, 0.0, 5.5 * scale, top=False)
        C1 = cylinder(pen, P, 0.0, 0.0, r_b, 4.5 * scale, H * 0.52, top=False)
        C2 = cylinder(pen, P, 0.0, 0.0, r_t, H * 0.50, H, top=False)
        # Small arched ground door & upper mill window
        a = P.az
        t_d = a + 1.5 * math.pi - 0.18
        pen.blot([C1(t_d - 0.16, 4.5 * scale), C1(t_d + 0.16, 4.5 * scale),
                  C1(t_d + 0.16, 11.0 * scale), C1(t_d, 13.0 * scale), C1(t_d - 0.16, 11.0 * scale)], jitter=0.05)
        pen.blot([C2(t_d - 0.12, H * 0.66), C2(t_d + 0.12, H * 0.66),
                  C2(t_d + 0.12, H * 0.78), C2(t_d - 0.12, H * 0.78)], jitter=0.04)
        # Conical wooden miller's cap
        cone(pen, P, 0.0, 0.0, r_t * 1.18, H, H + r_t * 1.35, broken=0.0)
        # Sweeping 4-blade lattice sails mounted cleanly on the front face
        _draw_windmill_sails(pen, P, rng, 0.0, -r_t * 1.15, H + 1.5 * scale, L=rng.uniform(22, 27) * scale)
    else:
        # Timber post mill perched on a braced octagonal/square trestle roundhouse
        r_b = 11.0 * scale
        cylinder(pen, P, 0.0, 0.0, r_b, 0.0, 10.5 * scale, top=True)
        W, D, H = 17.0 * scale, 19.0 * scale, 20.0 * scale
        R = W * 0.45
        P_up = lambda x, y, h=0.0: P(x, y, h + 10.5 * scale)
        P_up.az, P_up.ca, P_up.sa, P_up.ce, P_up.se = P.az, P.ca, P.sa, P.ce, P.se
        pl = dict(kind="house", W=W, D=D, H=H, R=R, ridge_x=False, floors=1,
                  timber=True, plinth=False, quoins=False, arch_door=False,
                  dormer=False, chimney=False, smoke=False, leanto=False, cross=False,
                  roof="tile", win=(2.6 * scale, 3.6 * scale))
        draw_faces(pen, P_up, house_faces(-W * 0.5, -D * 0.5, W, D, H, R, False, o=1.2 * scale), dict(pl, floors=-1))
        _draw_windmill_sails(pen, P, rng, 0.0, -D * 0.5 - 2.0 * scale, 10.5 * scale + H * 0.85, L=rng.uniform(21, 25) * scale)


def town_warehouse(pen, P, rng):
    """Heavy 2-3 storey medieval merchant warehouse / granary with roof crane hoist & stacked cargo crates."""
    W0 = pen.width
    ridge_x = rng.random() < 0.48
    if ridge_x:
        W, D = rng.uniform(46, 58), rng.uniform(24, 30)
    else:
        W, D = rng.uniform(26, 34), rng.uniform(44, 54)
    H = rng.uniform(21, 26)
    R = (D if ridge_x else W) * 0.5 * rng.uniform(0.95, 1.25)
    X0, Y0 = -W * 0.5, -D * 0.5

    pl = dict(kind="house", W=W, D=D, H=H, R=R, ridge_x=ridge_x, floors=2,
              timber=(rng.random() < 0.42), plinth=True, quoins=True, arch_door=True,
              dormer=False, chimney=False, smoke=False, leanto=(ridge_x and rng.random() < 0.42),
              cross=False, roof="tile", win=(3.2, 4.4))
    draw_house(pen, P, X0, Y0, pl)

    # Projecting timber roof crane beam & hanging cargo bale on BOTH gable-front and eave-front warehouses!
    lx = X0 + W * 0.5
    z_cr = (H + R * 0.72) if (not ridge_x) else (H + 2.8)
    p_in = P(lx, Y0, z_cr)
    p_out = P(lx, Y0 - 7.5, z_cr)
    p_brace = P(lx, Y0, z_cr - 4.0)
    if ridge_x:
        # Small gabled crane-hoist lucarne dormer perched at the front eave
        box(pen, P, lx - 4.5, Y0 - 1.2, lx + 4.5, Y0 + 4.5, H - 1.5, H + 4.2, top=False, w=W0 * 0.68)
        pyramid(pen, P, lx - 5.2, Y0 - 1.8, lx + 5.2, Y0 + 5.2, H + 4.2, H + 8.6)
    pen.beam(p_in, p_out, 1.30)
    pen.stroke([p_brace, P(lx, Y0 - 4.8, z_cr)], W0 * 0.54, taper=(0.05, 0.05), wobble=0)
    pen.stroke([p_out, P(lx, Y0 - 7.5, z_cr - 10.5)], W0 * 0.36, taper=(0, 0), wobble=0.08)
    box(pen, P, lx - 2.3, Y0 - 9.8, lx + 2.3, Y0 - 5.2, z_cr - 14.5, z_cr - 10.5, w=W0 * 0.58)
    # Neat stacked wooden cargo crates along the sunlit front loading bay
    for k in range(int(rng.integers(2, 4))):
        cx = X0 + 4.0 + k * 5.4
        if cx + 4.0 < X0 + W - 3.0:
            box(pen, P, cx, Y0 - 5.0, cx + 4.0, Y0 - 1.2, 0.0, rng.uniform(3.4, 5.2), w=W0 * 0.60)


# --------------------------------------------------------------------------- modular city wall pieces

def city_wall_piece(pen, P, rng, role="ew"):
    """Modular intact stone city wall segments, twin-towered gatehouses & standalone watchtowers."""
    H = rng.uniform(24, 30)
    t = rng.uniform(5.6, 6.8)
    if role == "ew":
        L = rng.uniform(86, 114)
        intact_curtain(pen, P, (-L * 0.5, 0.0), (L * 0.5, 0.0), H, t, rng, outer_side="south",
                       crenel=True, slits=True, buttresses=(rng.random() < 0.55))
    elif role == "ns":
        L = rng.uniform(82, 108)
        intact_curtain(pen, P, (0.0, -L * 0.5), (0.0, L * 0.5), H, t, rng, outer_side="east",
                       crenel=True, slits=True, buttresses=(rng.random() < 0.50))
    elif role == "gate":
        gw = rng.uniform(16, 20)
        style = "round" if rng.random() < 0.62 else "square"
        # Short flanking curtain stubs on both sides so it chains cleanly with wall runs
        stub = rng.uniform(16, 24)
        off = gw * 0.5 + max(6.2, gw * 0.44) * 1.45
        intact_curtain(pen, P, (-off - stub, 0.0), (-off + 2.0, 0.0), H, t, rng, outer_side="south")
        intact_gatehouse(pen, P, rng, 0.0, 0.0, gw, H, t, style=style, drawbridge=(rng.random() < 0.72), flags=True)
        intact_curtain(pen, P, (off - 2.0, 0.0), (off + stub, 0.0), H, t, rng, outer_side="south")
    else:  # standalone watchtower / bastion tower
        if rng.random() < 0.50:
            r = rng.uniform(11.0, 14.5)
            th = rng.uniform(44, 60)
            intact_round_tower(pen, P, rng, 0.0, 0.0, r, th,
                               cap=("cone" if rng.random() < 0.55 else "crenel"), flag=True)
        else:
            s = rng.uniform(10.5, 13.5)
            th = rng.uniform(42, 58)
            intact_square_tower(pen, P, rng, 0.0, 0.0, s, th,
                                cap=("pyramid" if rng.random() < 0.52 else "crenel"), flag=True)


# --------------------------------------------------------------------------- intact castles

def intact_castle(pen, P, rng):
    """Grand intact medieval castle in 4 silhouettes: wide, deep N-S, hexagonal ring, or concentric citadel."""
    shape = str(rng.choice(["wide", "tall", "hex", "concentric"], p=[0.28, 0.24, 0.24, 0.24]))
    H = rng.uniform(25, 31)
    t = rng.uniform(5.2, 6.4)
    round_towers = rng.random() < 0.68
    cap_r = "cone" if rng.random() < 0.68 else "crenel"
    cap_s = "pyramid" if rng.random() < 0.48 else "crenel"

    def draw_tower(X, Y, sc=1.0, flag=True):
        if round_towers:
            intact_round_tower(pen, P, rng, X, Y, rng.uniform(11.5, 13.5) * sc,
                               rng.uniform(42, 52) * sc, cap=cap_r, flag=flag)
        else:
            intact_square_tower(pen, P, rng, X, Y, rng.uniform(10.5, 12.5) * sc,
                                rng.uniform(40, 50) * sc, cap=cap_s, flag=flag)

    if shape == "hex":
        R = rng.uniform(80, 92)
        vs = [(R * math.cos(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3),
               R * math.sin(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3)) for k in range(6)]
        v0, v1, v2, v3, v4, v5 = vs  # SW, SE, E, NE, NW, W

        # Back towers & back ring walls
        draw_tower(*v4, sc=0.88)
        intact_curtain_seg(pen, P, v4, v3, H, t, rng, show_inner=True)
        intact_curtain_seg(pen, P, v5, v4, H, t, rng, show_inner=True)
        draw_tower(*v3, sc=0.88)
        draw_tower(*v5, sc=0.88)

        # Towering central royal keep + optional bailey hall/well inside the ring
        kw, kd, kh = rng.uniform(42, 52), rng.uniform(34, 42), rng.uniform(52, 66)
        kx, ky = -kw * 0.5 + rng.uniform(-5, 5), -kd * 0.5 + rng.uniform(-2, 8)
        k_items = [((kx - 3, ky - 3, kx + kw + 3, ky + kd + 3), ("keep", (kx, ky, kw, kd, kh)))]
        if rng.random() < 0.65:
            wx, wy = rng.uniform(-20, 20), ky - 14.0
            if wy > v0[1] + 12.0:
                k_items.append(((wx - 6, wy - 5, wx + 6, wy + 5), ("well", (wx, wy))))
        for kind_k, arg in sort_boxes(P, k_items):
            if kind_k == "keep":
                intact_keep(pen, P, rng, *arg, style=("turreted" if rng.random() < 0.65 else "gabled"), flag=True)
            else:
                well(pen, P, *arg)

        # Side walls & front towers + south gatehouse
        intact_curtain_seg(pen, P, v3, v2, H, t, rng, show_inner=False)
        intact_curtain_seg(pen, P, v5, v0, H, t, rng, show_inner=False)
        draw_tower(*v2, sc=0.88)
        draw_tower(*v0, sc=0.88)
        intact_curtain_seg(pen, P, v2, v1, H, t, rng, show_inner=False)
        gw = rng.uniform(15, 18)
        y_f = v0[1]
        intact_curtain(pen, P, v0, (-gw * 0.5 - 6.0, y_f), H, t, rng, outer_side="south")
        intact_gatehouse(pen, P, rng, 0.0, y_f, gw, H, t, style=("round" if round_towers else "square"))
        intact_curtain(pen, P, (gw * 0.5 + 6.0, y_f), v1, H, t, rng, outer_side="south")
        draw_tower(*v1, sc=0.88)
        return

    if shape == "concentric":
        # Outer lower curtain wall enclosing a taller inner citadel & royal keep!
        OW, OD = rng.uniform(156, 178), rng.uniform(122, 142)
        IW, ID = OW * 0.58, OD * 0.56
        H_out = rng.uniform(20, 24)
        H_in = rng.uniform(32, 38)
        ox0, oy0, ox1, oy1 = -OW * 0.5, -OD * 0.5, OW * 0.5, OD * 0.5
        ix0, iy0, ix1, iy1 = -IW * 0.5, -ID * 0.5 + 8.0, IW * 0.5, ID * 0.5 + 8.0

        # Outer back wall & towers
        draw_tower(ox0, oy1, sc=0.82)
        intact_curtain(pen, P, (ox0, oy1), (ox1, oy1), H_out, t * 0.9, rng, outer_side="north")
        draw_tower(ox1, oy1, sc=0.82)
        intact_curtain(pen, P, (ox0, oy0), (ox0, oy1), H_out, t * 0.9, rng, outer_side="west")

        # Inner higher citadel back wall & towers
        draw_tower(ix0, iy1, sc=0.96)
        intact_curtain(pen, P, (ix0, iy1), (ix1, iy1), H_in, t, rng, outer_side="north")
        draw_tower(ix1, iy1, sc=0.96)
        intact_curtain(pen, P, (ix0, iy0), (ix0, iy1), H_in, t, rng, outer_side="west")

        # Commanding central keep inside inner citadel
        kw, kd, kh = IW * 0.54, ID * 0.54, rng.uniform(56, 70)
        intact_keep(pen, P, rng, -kw * 0.5, (iy0 + iy1) * 0.5 - kd * 0.5, kw, kd, kh,
                    style=("turreted" if rng.random() < 0.70 else "gabled"), flag=True)

        # Inner citadel east & front walls + inner towers
        intact_curtain(pen, P, (ix1, iy0), (ix1, iy1), H_in, t, rng, outer_side="east")
        draw_tower(ix0, iy0, sc=0.96)
        intact_curtain(pen, P, (ix0, iy0), (ix1, iy0), H_in, t, rng, outer_side="south")
        draw_tower(ix1, iy0, sc=0.96)

        # Outer east & front walls + main gatehouse
        intact_curtain(pen, P, (ox1, oy0), (ox1, oy1), H_out, t * 0.9, rng, outer_side="east")
        draw_tower(ox0, oy0, sc=0.82)
        gw = rng.uniform(16, 19)
        intact_curtain(pen, P, (ox0, oy0), (-gw * 0.5 - 6.0, oy0), H_out, t * 0.9, rng, outer_side="south")
        intact_gatehouse(pen, P, rng, 0.0, oy0, gw, H_out, t * 0.9, style=("round" if round_towers else "square"))
        intact_curtain(pen, P, (gw * 0.5 + 6.0, oy0), (ox1, oy0), H_out, t * 0.9, rng, outer_side="south")
        draw_tower(ox1, oy0, sc=0.82)
        return

    # Wide or deep N-S rectangular intact curtain-wall castle
    if shape == "wide":
        CW, CD = rng.uniform(144, 174), rng.uniform(92, 114)
    else:
        CW, CD = rng.uniform(98, 120), rng.uniform(144, 176)
    x0, y0, x1, y1 = -CW * 0.5, -CD * 0.5, CW * 0.5, CD * 0.5
    mid_towers = (shape == "tall" and rng.random() < 0.72)

    # 1. Back wall & back towers
    draw_tower(x0, y1)
    intact_curtain(pen, P, (x0, y1), (x1, y1), H, t, rng, outer_side="north")
    draw_tower(x1, y1)
    if mid_towers:
        intact_curtain(pen, P, (x0, 0.0), (x0, y1), H, t, rng, outer_side="west")
        draw_tower(x0, 0.0, sc=0.86, flag=False)
        intact_curtain(pen, P, (x0, y0), (x0, 0.0), H, t, rng, outer_side="west")
    else:
        intact_curtain(pen, P, (x0, y0), (x0, y1), H, t, rng, outer_side="west")

    # 2. Bailey interior: intact royal keep + intact great hall / barracks + courtyard well
    b_items, b_fps = [], []
    kw = rng.uniform(40, 52) if shape == "wide" else rng.uniform(36, 46)
    kd = rng.uniform(32, 42)
    kh = rng.uniform(52, 66)
    kx = rng.uniform(x0 + 18, max(x0 + 19, x1 - 18 - kw))
    ky = rng.uniform(y0 + 24, max(y0 + 25, y1 - 16 - kd))
    k_fp = (kx - 3, ky - 3, kx + kw + 3, ky + kd + 3)
    b_fps.append(k_fp)
    b_items.append((k_fp, ("keep", (kx, ky, kw, kd, kh))))

    # Intact bailey great hall / chapel along opposite side of courtyard
    h_pl = dict(kind="house", W=rng.uniform(36, 46), D=rng.uniform(18, 22), H=rng.uniform(16, 20),
                R=11.0, ridge_x=(shape == "tall"), floors=1, timber=False, plinth=True, quoins=True,
                arch_door=True, dormer=False, chimney=True, smoke=False, leanto=False, cross=True,
                roof="tile", win=(3.0, 5.2))
    if not h_pl["ridge_x"]:
        h_pl["W"], h_pl["D"] = h_pl["D"], h_pl["W"]
    h_pl["R"] = (h_pl["D"] if h_pl["ridge_x"] else h_pl["W"]) * 0.55
    for _ in range(30):
        hx = rng.uniform(x0 + 14, max(x0 + 15, x1 - 14 - h_pl["W"]))
        hy = rng.uniform(y0 + 14, max(y0 + 15, y1 - 14 - h_pl["D"]))
        h_fp = (hx - 2, hy - 2, hx + h_pl["W"] + 2, hy + h_pl["D"] + 2)
        if not _overlaps(h_fp, b_fps, 5.0):
            b_fps.append(h_fp)
            b_items.append((h_fp, ("hall", (hx, hy, h_pl))))
            break
    # Optional courtyard well
    for _ in range(25):
        wx, wy = rng.uniform(x0 + 22, x1 - 22), rng.uniform(y0 + 16, y1 - 16)
        wbox = (wx - 6.5, wy - 5.5, wx + 6.5, wy + 5.5)
        if not _overlaps(wbox, b_fps, 3.0):
            b_fps.append(wbox)
            b_items.append((wbox, ("well", (wx, wy))))
            break

    for kind_b, arg in sort_boxes(P, b_items):
        if kind_b == "keep":
            intact_keep(pen, P, rng, *arg, style=("turreted" if rng.random() < 0.62 else "gabled"), flag=True)
        elif kind_b == "hall":
            draw_house(pen, P, *arg)
        else:
            well(pen, P, *arg)

    # 3. East wall, front corner towers & south gatehouse
    if mid_towers:
        intact_curtain(pen, P, (x1, 0.0), (x1, y1), H, t, rng, outer_side="east")
        draw_tower(x1, 0.0, sc=0.86, flag=False)
        intact_curtain(pen, P, (x1, y0), (x1, 0.0), H, t, rng, outer_side="east")
    else:
        intact_curtain(pen, P, (x1, y0), (x1, y1), H, t, rng, outer_side="east")

    gw = rng.uniform(16, 20)
    gx = rng.uniform(-CW * 0.10, CW * 0.10) if CW > 118 else 0.0
    draw_tower(x0, y0)
    intact_curtain(pen, P, (x0, y0), (gx - gw * 0.5 - 7.0, y0), H, t, rng, outer_side="south")
    intact_gatehouse(pen, P, rng, gx, y0, gw, H, t, style=("round" if round_towers else "square"))
    intact_curtain(pen, P, (gx + gw * 0.5 + 7.0, y0), (x1, y0), H, t, rng, outer_side="south")
    draw_tower(x1, y0)


# --------------------------------------------------------------------------- big villages (>= 20 houses!)

def big_village(pen, P, rng):
    """Sprawling 20-28 house big village: crossroads market town, wide high-street + lanes, or double-ring green."""
    shape = str(rng.choice(["crossroads", "high_street", "ring_green"], p=[0.36, 0.34, 0.30]))
    road_pts_list = []
    road_boxes = []
    items, boxes = [], []

    if shape == "crossroads":
        L1 = rng.uniform(270, 315)
        xs1 = np.linspace(-L1 * 0.5, L1 * 0.5, 72)
        ys1 = rng.uniform(7, 15) * np.sin(xs1 / L1 * math.pi * 1.4 + rng.uniform(0, TAU))
        L2 = rng.uniform(230, 275)
        ys2 = np.linspace(-L2 * 0.5, L2 * 0.5, 64)
        xs2 = -0.44 * ys2 + rng.uniform(6, 13) * np.sin(ys2 / L2 * math.pi * 1.3 + rng.uniform(0, TAU))
        road_pts_list = [np.stack([xs1, ys1], axis=1), np.stack([xs2, ys2], axis=1)]
    elif shape == "high_street":
        L = rng.uniform(310, 355)
        xs = np.linspace(-L * 0.5, L * 0.5, 84)
        ph = rng.uniform(0, TAU)
        ys = rng.uniform(10, 20) * np.sin(xs / L * math.pi * 1.6 + ph)
        road_pts_list.append(np.stack([xs, ys], axis=1))
        # Two branching rural side lanes heading north & south off the main high street
        for bx_f, sgn in ((-0.22, 1.0), (0.24, -1.0)):
            bx = L * bx_f
            by0 = float(np.interp(bx, xs, ys))
            ly = np.linspace(by0, by0 + sgn * rng.uniform(90, 118), 28)
            lx = bx - 0.44 * (ly - by0) + rng.uniform(-5, 5) * np.sin(np.linspace(0, math.pi, 28))
            road_pts_list.append(np.stack([lx, ly], axis=1))
    else:  # wide ring_green + 3 radial approach lanes
        rx_g, ry_g = rng.uniform(44, 54), rng.uniform(46, 56)
        ts_g = np.linspace(0.0, TAU, 56)
        gx_r = rx_g * np.cos(ts_g) - 0.16 * ry_g * np.sin(ts_g)
        gy_r = ry_g * np.sin(ts_g)
        road_pts_list.append(np.stack([gx_r, gy_r], axis=1))
        road_pts_list.append(np.stack([np.linspace(rx_g, rx_g + 95, 22), np.zeros(22)], axis=1))
        road_pts_list.append(np.stack([np.linspace(-rx_g - 95, -rx_g, 22), np.zeros(22)], axis=1))
        ys_n = np.linspace(-ry_g - 78, -ry_g, 18)
        road_pts_list.append(np.stack([-0.44 * (ys_n + ry_g) + 0.16 * ry_g, ys_n], axis=1))

    for rpts in road_pts_list:
        _draw_road_strip(pen, P, rng, rpts, half_w=3.2)
        for x, y in rpts:
            road_boxes.append((x - 2.6, y - 4.4, x + 2.6, y + 4.4))

    # Prominent village parish church (~82% of big villages)
    if rng.random() < 0.82:
        for _ in range(90):
            pl_c = plan_chapel(rng)
            if shape == "high_street":
                rpts = road_pts_list[0]
                x = rng.uniform(-100, 90)
                X0 = x - pl_c["W"] * 0.5
                yr = float(np.interp(x, rpts[:, 0], rpts[:, 1]))
                Y0 = yr + 18.0 + pl_c["tr_proj"]
            else:
                X0 = rng.uniform(-55, 18) - pl_c["W"] * 0.4
                Y0 = rng.uniform(28, 72)
            fp = chapel_footprint(pl_c, X0, Y0)
            if not _overlaps(fp, boxes + road_boxes, 2.5):
                boxes.append(fp)
                items.append((fp, ("chapel", (X0, Y0, pl_c))))
                break

    # Central village well on green or near crossroads
    if shape == "ring_green":
        wbox = (-6.5, -5.0, 6.5, 5.0)
        if not _overlaps((-8, -6, 8, 6), boxes + road_boxes, 1):
            items.append((wbox, ("well", (0.0, 0.0))))
            boxes.append((-8, -6, 8, 6))

    # Optional rural tower windmill standing on the outskirts (~48% of big villages)
    has_windmill = False
    if rng.random() < 0.48:
        for _ in range(40):
            ang = rng.uniform(0.15, math.pi - 0.15)
            rad = rng.uniform(118, 155)
            wx, wy = rad * math.cos(ang), rad * math.sin(ang) * 0.72
            wm_box = (wx - 18, wy - 16, wx + 18, wy + 16)
            if not _overlaps(wm_box, boxes + road_boxes, 4.0):
                boxes.append(wm_box)
                items.append((wm_box, ("windmill", (wx, wy))))
                has_windmill = True
                break

    # Guaranteed >= 20 houses (target 22-28!) via road-hugging tiers + organic backfill spiral
    target_houses = int(rng.integers(22, 28))
    house_count = 0

    def _try_place_house(cx, cy, kind=None, gap=2.4):
        nonlocal house_count
        pl = plan_house(rng, kind)
        # Slightly crisper compact scale (~0.88) so 24+ houses read cleanly without sprawling off canvas
        for k_dim in ("W", "D", "H", "R"):
            pl[k_dim] *= 0.88
        pl["win"] = (pl["win"][0] * 0.88, pl["win"][1] * 0.88)
        X0, Y0 = cx - pl["W"] * 0.5, cy - pl["D"] * 0.5
        fp = footprint(pl, X0, Y0)
        if not _overlaps(fp, boxes + road_boxes, gap):
            boxes.append(fp)
            items.append(_house_item(pl, X0, Y0))
            house_count += 1
            return True
        return False

    # Pass 1: Road-hugging primary & secondary tiers
    for attempt in range(1800):
        if house_count >= target_houses:
            break
        if shape == "crossroads":
            rpts = road_pts_list[0 if rng.random() < 0.55 else 1]
            idx = int(rng.integers(4, len(rpts) - 4))
            rx, ry = float(rpts[idx, 0]), float(rpts[idx, 1])
            tier = rng.uniform(16, 30) if attempt < 900 else rng.uniform(32, 74)
            side = int(rng.choice([-1, 1]))
            if rpts is road_pts_list[0]:
                cx, cy = rx + rng.uniform(-8, 8), ry + side * tier
            else:
                cx, cy = rx + side * tier, ry + rng.uniform(-8, 8)
        elif shape == "high_street":
            rpts = road_pts_list[0] if rng.random() < 0.72 else road_pts_list[int(rng.integers(1, len(road_pts_list)))]
            idx = int(rng.integers(3, len(rpts) - 3))
            rx, ry = float(rpts[idx, 0]), float(rpts[idx, 1])
            tier = rng.uniform(16, 32) if attempt < 900 else rng.uniform(34, 72)
            side = int(rng.choice([-1, 1]))
            if rpts is road_pts_list[0]:
                cx, cy = rx + rng.uniform(-8, 8), ry + side * tier
            else:
                cx, cy = rx + side * tier, ry + rng.uniform(-8, 8)
        else:  # ring_green: inner ring (rad 64..92) + outer ring (rad 96..158)
            ang = rng.uniform(0.0, TAU)
            rad = rng.uniform(64, 96) if attempt < 750 else rng.uniform(92, 160)
            cx = rad * math.cos(ang) * 0.98 - 0.18 * rad * math.sin(ang)
            cy = rad * math.sin(ang) * 0.92
        _try_place_house(cx, cy, gap=(2.2 if attempt < 1100 else 1.4))

    # Pass 2: Deterministic expanding-ring guarantee so house_count is ALWAYS >= 20
    ring_r = 45.0
    while house_count < 20 and ring_r < 210.0:
        for ang in np.linspace(0.0, TAU, 28, endpoint=False) + rng.uniform(0, 0.2):
            if house_count >= 21:
                break
            _try_place_house(ring_r * math.cos(ang), ring_r * math.sin(ang) * 0.85, kind="cottage", gap=1.2)
        ring_r += 22.0

    bldg_boxes = list(boxes)
    # Garden fences & village trees
    for _ in range(int(rng.integers(3, 7))):
        b = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
        A, B = (b[0], b[1] - 5.5), (b[2], b[1] - 5.5)
        fbox = (A[0], A[1] - 1, B[0], A[1] + 1)
        if not _overlaps(fbox, boxes + road_boxes, 0):
            items.append((fbox, ("fence", (A, B))))
            boxes.append(fbox)

    bx_all = [b[0] for b in boxes] + [b[2] for b in boxes]
    by_all = [b[1] for b in boxes] + [b[3] for b in boxes]
    occ_boxes = list(boxes) + list(road_boxes)
    # Keep trees tucked organically beside village yards (no stray trees drifting far off into space!)
    for _ in range(int(rng.integers(7, 13))):
        for _ in range(45):
            b_ref = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
            x = rng.uniform(b_ref[0] - 18, b_ref[2] + 18)
            y = rng.uniform(b_ref[1] - 14, b_ref[3] + 14)
            tbox = (x - 4.5, y - 4.5, x + 4.5, y + 4.5)
            if not _overlaps(tbox, occ_boxes, 1.5):
                items.append((tbox, ("tree", (x, y))))
                occ_boxes.append(tbox)
                boxes.append(tbox)
                break
    _tufts(rng, items, bldg_boxes, occ_boxes)

    # Render everything sorted cleanly back-to-front (including optional windmill!)
    for kind, args in sort_boxes(P, items):
        if kind == "house":
            draw_house(pen, P, *args)
        elif kind == "chapel":
            chapel(pen, P, args[0], args[1], rng, args[2])
        elif kind == "windmill":
            P_w = lambda x, y, h=0.0, ox=args[0], oy=args[1]: P(x + ox, y + oy, h)
            P_w.az, P_w.ca, P_w.sa, P_w.ce, P_w.se = P.az, P.ca, P.sa, P.ce, P.se
            town_windmill(pen, P_w, rng, scale=0.82)
        elif kind == "tree":
            tree(pen, *P(*args))
        elif kind == "well":
            well(pen, P, *args)
        elif kind == "fence":
            fence(pen, P, *args)
        elif kind == "tuft":
            pen.tuft(*P(*args), rng.uniform(2.6, 4.4))


# --------------------------------------------------------------------------- cities (~20+ dense overlapping urban buildings + fort/city walls!)

def _pack_dense_urban_buildings(rng, inside_fn, target_n, existing_boxes, scale=0.72,
                                 x_lim=(-115, 115), y_lim=(-90, 90)):
    """Pack `target_n` densely clustered, tightly overlapping medieval townhouses inside `inside_fn(x0,y0,x1,y1)`.

    Combines flush row-house blocks + dense jittered grid scanning + pocket infill so the entire city
    interior fills shoulder-to-shoulder with overlapping rooftops and ZERO awkward empty green lawns!
    """
    placed = []
    occ = list(existing_boxes)

    # 1. Seed flush 2-4 facade row-house street terraces across the city
    for _ in range(180):
        if len(placed) >= int(target_n * 0.56):
            break
        n_row = int(rng.choice([2, 3, 4], p=[0.46, 0.38, 0.16]))
        along_x = rng.random() < 0.72
        pls = [plan_townhouse(rng, scale=scale * rng.uniform(0.92, 1.08),
                              force_ridge_x=(not along_x if rng.random() < 0.75 else None))
               for _ in range(n_row)]
        cx = rng.uniform(x_lim[0] * 0.90, x_lim[1] * 0.90)
        cy = rng.uniform(y_lim[0] * 0.90, y_lim[1] * 0.90)
        cand_items, cand_fps = [], []
        ok = True
        if along_x:
            cur_x = cx - sum(p["W"] for p in pls) * 0.5
            for pl in pls:
                y0 = cy - pl["D"] * 0.5 + rng.uniform(-0.8, 0.8)
                fp = (cur_x, y0, cur_x + pl["W"], y0 + pl["D"])
                if not inside_fn(*fp) or _overlaps(fp, occ, 0.35):
                    ok = False
                    break
                cand_items.append((fp, ("townhouse", (cur_x, y0, pl))))
                cand_fps.append(fp)
                cur_x += pl["W"] + 0.2
        else:
            cur_y = cy - sum(p["D"] for p in pls) * 0.5
            for pl in pls:
                x0 = cx - pl["W"] * 0.5 + rng.uniform(-0.8, 0.8)
                fp = (x0, cur_y, x0 + pl["W"], cur_y + pl["D"])
                if not inside_fn(*fp) or _overlaps(fp, occ, 0.35):
                    ok = False
                    break
                cand_items.append((fp, ("townhouse", (x0, cur_y, pl))))
                cand_fps.append(fp)
                cur_y += pl["D"] + 0.2
        if ok:
            placed.extend(cand_items)
            occ.extend(cand_fps)

    # 2. Systematic shuffled grid sweep so EVERY corner & back-wall pocket fills tightly!
    gx_pts = np.arange(x_lim[0] + 6.0, x_lim[1] - 6.0, 13.0 * scale)
    gy_pts = np.arange(y_lim[0] + 6.0, y_lim[1] - 6.0, 13.0 * scale)
    cells = [(float(gx), float(gy)) for gx in gx_pts for gy in gy_pts]
    rng.shuffle(cells)
    for gx, gy in cells:
        if len(placed) >= target_n:
            break
        for sc_try in (scale * rng.uniform(0.92, 1.08), scale * rng.uniform(0.76, 0.88)):
            pl = plan_townhouse(rng, scale=sc_try)
            cx = gx + rng.uniform(-4.0, 4.0)
            cy = gy + rng.uniform(-4.0, 4.0)
            x0, y0 = cx - pl["W"] * 0.5, cy - pl["D"] * 0.5
            fp = (x0, y0, x0 + pl["W"], y0 + pl["D"])
            if inside_fn(*fp) and not _overlaps(fp, occ, 0.30):
                occ.append(fp)
                placed.append((fp, ("townhouse", (x0, y0, pl))))
                break

    # 3. Extra compact pocket-fill pass to eliminate any lingering green voids
    for _ in range(2200):
        if len(placed) >= target_n + 10:
            break
        pl = plan_townhouse(rng, scale=scale * rng.uniform(0.72, 0.90))
        cx = rng.uniform(*x_lim)
        cy = rng.uniform(*y_lim)
        x0, y0 = cx - pl["W"] * 0.5, cy - pl["D"] * 0.5
        fp = (x0, y0, x0 + pl["W"], y0 + pl["D"])
        if inside_fn(*fp) and not _overlaps(fp, occ, 0.25):
            occ.append(fp)
            placed.append((fp, ("townhouse", (x0, y0, pl))))

    return placed, occ


def _draw_city_interior(pen, P, rng, items):
    """Render all sorted interior/suburban city structures back-to-front."""
    for kind, args in sort_boxes(P, items):
        if kind == "townhouse":
            draw_townhouse(pen, P, *args)
        elif kind == "house":
            draw_house(pen, P, *args)
        elif kind == "keep":
            intact_keep(pen, P, rng, *args, style=("turreted" if rng.random() < 0.72 else "gabled"), flag=True)
        elif kind == "cathedral":
            cx, cy, sc = args
            P_c = lambda x, y, h=0.0, ox=cx, oy=cy: P(x + ox, y + oy, h)
            P_c.az, P_c.ca, P_c.sa, P_c.ce, P_c.se = P.az, P.ca, P.sa, P.ce, P.se
            town_cathedral(pen, P_c, rng, scale=sc)
        elif kind == "guildhall":
            cx, cy, sc = args
            P_g = lambda x, y, h=0.0, ox=cx, oy=cy: P(x + ox, y + oy, h)
            P_g.az, P_g.ca, P_g.sa, P_g.ce, P_g.se = P.az, P.ca, P.sa, P.ce, P.se
            _draw_compact_guildhall(pen, P_g, rng, scale=sc)
        elif kind == "stall":
            _draw_market_stall(pen, P, rng, *args)
        elif kind == "well":
            well(pen, P, *args)
        elif kind == "fence":
            fence(pen, P, *args)
        elif kind == "citadel":
            _draw_inner_citadel(pen, P, rng, *args)
        elif kind == "tower":
            intact_round_tower(pen, P, rng, *args)
        elif kind == "tree":
            if len(args) >= 4:
                tx, ty, th, tkind = args[:4]
                tree(pen, *P(tx, ty), h=th, kind=tkind)
            else:
                tree(pen, *P(*args[:2]))
        elif kind == "tuft":
            pen.tuft(*P(*args), rng.uniform(2.6, 4.2))


def _plan_poor_cottage(rng, scale=0.64, compact=False, force_ridge_x=None):
    """Humble 1-storey thatched hovel / peasant cottage for poor outer city suburbs (contrasts with tall rich inner city)."""
    if compact:
        kind = str(rng.choice(["hovel", "cottage"], p=[0.62, 0.38]))
    else:
        kind = str(rng.choice(["hovel", "cottage", "long"], p=[0.42, 0.46, 0.12]))
    if kind == "hovel":
        W, D, H = rng.uniform(15, 21) * scale, rng.uniform(12, 16) * scale, rng.uniform(6.4, 8.4) * scale
    elif kind == "long":
        W, D, H = rng.uniform(29, 36) * scale, rng.uniform(14, 17) * scale, rng.uniform(7.4, 9.4) * scale
    else:
        W, D, H = rng.uniform(20, 27) * scale, rng.uniform(14, 18) * scale, rng.uniform(7.8, 10.0) * scale
    if force_ridge_x is not None:
        ridge_x = bool(force_ridge_x)
    else:
        ridge_x = rng.random() < (0.80 if kind == "long" else 0.56)
    if not ridge_x:
        W, D = D, W
    span = (D if ridge_x else W) * 0.5
    has_chimney = (kind != "hovel") and (rng.random() < 0.20)
    return dict(
        kind="cottage", W=W, D=D, H=H, R=span * rng.uniform(0.84, 1.08),
        ridge_x=ridge_x, floors=1, timber=(rng.random() < 0.16),
        plinth=False, quoins=False, arch_door=False, dormer=False,
        chimney=has_chimney, chimney_scale=0.54,
        smoke=(has_chimney and rng.random() < 0.42),
        leanto=(ridge_x and kind == "cottage" and not compact and rng.random() < 0.22),
        cross=False, roof=("thatch" if rng.random() < 0.88 else "tile"),
        win=(2.4 * scale, 3.2 * scale),
    )


def _hits_towers(ax, ay, bx, by, tower_pts):
    """True if world box (ax, ay, bx, by) encroaches on any bastion tower / gatehouse circle (tx, ty, rad)."""
    for tx, ty, rad in tower_pts:
        cx = min(max(tx, ax), bx)
        cy = min(max(ty, ay), by)
        if (cx - tx) ** 2 + (cy - ty) ** 2 < rad * rad:
            return True
    return False


def _plan_outer_suburbs(rng, wall_excl_fn, wall_bounds, y_gate, gw, ruined=False, hex_R=None,
                        tower_pts=(), hug_back=(), hug_front=()):
    """Build a dense-to-sparse sprawling poor outer city (~85-115 cottages, >= inner city count!).

    3-zone radial density gradient:
      1. Wall-hugging shantytown collar packed tightly right up against every curtain wall bay (wrapping cleanly
         around protruding bastion towers and the south gatehouse).
      2. Medium-dense mid extramural faubourg lining the winding south gate road + branching side-lanes.
      3. Progressively sparser scattered outer cottages, fences, wells, stalls & scaled trees tapering outward.

    Returns `(roads, back_items, front_items)` split strictly for exact 3D wall/tower occlusion!
    """
    wx0, wy0, wx1, wy1 = wall_bounds
    # Main winding dirt approach road leading south out of the city gate
    road_len = rng.uniform(96, 120)
    ys_r = np.linspace(y_gate - 10.0, y_gate - 10.0 - road_len, 30)
    ph_r = rng.uniform(0, TAU)
    xs_r = -0.30 * (ys_r - (y_gate - 10.0)) * rng.uniform(-0.32, 0.42) + rng.uniform(4.0, 10.5) * np.sin(
        np.linspace(0, math.pi * 1.25, 30) + ph_r
    )
    xs_r -= xs_r[0]
    road_pts = np.stack([xs_r, ys_r], axis=1)
    roads = [road_pts]
    all_road_pts = list(road_pts)

    # 1-2 branching dirt lanes forking cleanly off the outer shoulders of the main gate road
    branch_dirs = [-1, 1] if rng.random() < 0.72 else [int(rng.choice([-1, 1]))]
    for b_dir in branch_dirs:
        j_idx = int(rng.integers(6, 16))
        jx, jy = float(road_pts[j_idx, 0]), float(road_pts[j_idx, 1])
        b_len = rng.uniform(62, 92)
        ts_b = np.linspace(0.0, 1.0, 18)
        bx_r = (jx + b_dir * 3.8) + b_dir * b_len * ts_b
        by_r = jy - b_len * rng.uniform(0.14, 0.38) * ts_b + rng.uniform(-5.5, 5.5) * np.sin(math.pi * ts_b)
        branch_pts = np.stack([bx_r, by_r], axis=1)
        roads.append(branch_pts)
        all_road_pts.extend(branch_pts[2:])

    road_boxes = [(float(rx) - 4.4, float(ry) - 4.4, float(rx) + 4.4, float(ry) + 4.4) for rx, ry in all_road_pts]
    gate_pad = [(-gw * 0.85 - 7.0, y_gate - 20.0, gw * 0.85 + 7.0, y_gate + 4.0)]

    back_items, front_items = [], []
    back_occ, front_occ = list(gate_pad), list(road_boxes) + list(gate_pad)
    back_bldg, front_bldg = [], []

    def blocked(fp):
        return wall_excl_fn(*fp) or _hits_towers(*fp, tower_pts)

    if hex_R is not None:
        h_wall = hex_R * 0.866 + 3.6

        def is_back(fp):
            ax, ay, bx, by = fp
            # Outside N wall (clear of NE corner), outside NW slanted wall, or peeking behind upper NE flank
            if ay > h_wall and bx < hex_R * 0.44 and ax > -hex_R * 0.72:
                return True
            if ay > 4.0 and all(-0.866 * px + 0.5 * py > h_wall for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by))):
                return True
            if ay > 8.0 and by < hex_R * 0.74 and all(
                0.866 * px + 0.5 * py > h_wall + 2.0 for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by))
            ):
                return True
            return False

        def is_front(fp):
            ax, ay, bx, by = fp
            # Strictly south of S wall
            if by < -h_wall:
                return True
            # Strictly outside SE slanted wall (south of E tower) — cleanly in front in az=28°!
            if by < -4.0 and all(0.866 * px - 0.5 * py > h_wall for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by))):
                return True
            # Strictly outside SW slanted wall (south of W tower)
            if by < -8.0 and bx < -hex_R * 0.50 - 4.0 and all(
                -0.866 * px - 0.5 * py > h_wall for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by))
            ):
                return True
            return False
    else:
        def is_back(fp):
            ax, ay, bx, by = fp
            # West wall & NW outer flank (all strictly behind west wall & SW/NW towers!)
            if bx < wx0 - 3.6 and ay > wy0 - 1.0:
                return True
            # North wall (strictly behind north wall, clear of NE corner tower ambiguity)
            if ay > wy1 + 3.6 and bx < wx1 - 10.0:
                return True
            return False

        def is_front(fp):
            ax, ay, bx, by = fp
            # South wall & SW/SE front flanks (all strictly in front of south wall & towers!)
            if by < wy0 - 3.6:
                return True
            # East wall (strictly east of east wall, clear of NE corner tower)
            if ax > wx1 + 3.6 and by < wy1 - 10.0:
                return True
            return False

    def _plan_poor_spec(compact=False, force_ridge_x=None):
        if ruined:
            sc_r = rng.uniform(0.32, 0.42) if compact else rng.uniform(0.34, 0.46)
            pl = plan_ruined_house(rng, k=sc_r, severity=rng.uniform(0.38, 0.88))
            pl["chimney"] = rng.random() < 0.20
            return pl, 1.2
        sc_c = rng.uniform(0.58, 0.69) if compact else rng.uniform(0.60, 0.74)
        pl = _plan_poor_cottage(rng, scale=sc_c, compact=compact, force_ridge_x=force_ridge_x)
        return pl, 2.0

    def _finish_poor(pl, cx, cy):
        x0, y0 = cx - pl["W"] * 0.5, cy - pl["D"] * 0.5
        if ruined:
            fp = (x0 - 1.2, y0 - 1.2, x0 + pl["W"] + 1.2, y0 + pl["D"] + 1.2)
            return fp, ("rhouse", (x0, y0, pl))
        fp = footprint(pl, x0, y0)
        return fp, ("house", (x0, y0, pl))

    def _try_place_back(cx, cy, gap, compact=False, pl=None):
        if pl is None:
            pl, _ = _plan_poor_spec(compact=compact)
        fp, payload = _finish_poor(pl, cx, cy)
        if is_back(fp) and (not blocked(fp)) and (not _overlaps(fp, back_occ, gap)):
            back_occ.append(fp)
            back_bldg.append(fp)
            back_items.append((fp, payload))
            return True
        return False

    def _try_place_front(cx, cy, gap, compact=False, pl=None):
        if pl is None:
            pl, _ = _plan_poor_spec(compact=compact)
        fp, payload = _finish_poor(pl, cx, cy)
        if is_front(fp) and (not blocked(fp)) and (not _overlaps(fp, front_occ, gap)):
            front_occ.append(fp)
            front_bldg.append(fp)
            front_items.append((fp, payload))
            return True
        return False

    # === ZONE 1: WALL-HUGGING SHANTYTOWN COLLAR (dense, huddled flush right against the outer ramparts!) ===
    # Sweep every back & front curtain wall segment in two snug concentric tiers (tier 0 flush against wall!)
    def _sweep_wall_huddle(segs, place_fn, target_add):
        added = 0
        cands = []
        for (ax, ay), (bx, by), nx, ny in segs:
            L = math.hypot(bx - ax, by - ay)
            n_st = max(5, int(L / 11.5))
            for tier in (0, 1):
                for t in np.linspace(0.03, 0.97, n_st):
                    px = ax + (bx - ax) * t + rng.uniform(-2.6, 2.6)
                    py = ay + (by - ay) * t + rng.uniform(-2.6, 2.6)
                    cands.append((tier, px, py, nx, ny))
        t0 = [c for c in cands if c[0] == 0]
        t1 = [c for c in cands if c[0] == 1]
        rng.shuffle(t0)
        rng.shuffle(t1)
        for tier, px, py, nx, ny in t0 + t1:
            if added >= target_add:
                break
            # Orient wall-hugging hovels mostly parallel to the wall face so they press snug against masonry!
            par_rx = (abs(ny) > 0.5) if (tier == 0 and rng.random() < 0.72) else None
            pl, pad = _plan_poor_spec(compact=(tier == 0), force_ridge_x=par_rx)
            half_proj = abs(nx) * (pl["W"] * 0.5 + pad) + abs(ny) * (pl["D"] * 0.5 + pad)
            d = (3.8 + half_proj + rng.uniform(0.15, 2.8)) if tier == 0 else (
                3.8 + half_proj + rng.uniform(13.0, 24.0)
            )
            cx = px + nx * d
            cy = py + ny * d
            if place_fn(cx, cy, gap=(0.35 if tier == 0 else 0.85), compact=(tier == 0), pl=pl):
                added += 1
        return added

    n_back_hug = int(rng.integers(24, 32))
    n_front_hug = int(rng.integers(38, 48))
    _sweep_wall_huddle(hug_back, _try_place_back, n_back_hug)
    _sweep_wall_huddle(hug_front, _try_place_front, n_front_hug)

    # Extra dense pocket-fill right along the walls to pack any gaps between wall-hugging huts
    n_back_mid = len(back_bldg) + int(rng.integers(12, 18))
    for _ in range(900):
        if len(back_bldg) >= n_back_mid:
            break
        if hex_R is not None:
            r_side = rng.random()
            if r_side < 0.45:
                t = rng.uniform(0.06, 0.94)
                ex = (-hex_R) * (1 - t) + (-0.5 * hex_R) * t
                ey = (0.0) * (1 - t) + (0.866 * hex_R) * t
                d = rng.uniform(12.0, 52.0)
                cx, cy = ex - 0.866 * d, ey + 0.5 * d
            elif r_side < 0.80:
                cx = rng.uniform(-hex_R * 0.60, hex_R * 0.38)
                cy = hex_R * 0.866 + rng.uniform(12.0, 48.0)
            else:
                t = rng.uniform(0.14, 0.84)
                ex = (0.5 * hex_R) * (1 - t) + hex_R * t
                ey = (0.866 * hex_R) * (1 - t) + 0.0 * t
                d = rng.uniform(14.0, 44.0)
                cx, cy = ex + 0.866 * d, ey + 0.5 * d
        else:
            if rng.random() < 0.54:
                cx = wx0 - rng.uniform(10.0, 56.0)
                cy = rng.uniform(wy0 + 4.0, wy1 + 32.0)
            else:
                cx = rng.uniform(wx0 - 20.0, wx1 - 16.0)
                cy = wy1 + rng.uniform(10.0, 48.0)
        _try_place_back(cx, cy, gap=rng.uniform(0.8, 1.8), compact=(rng.random() < 0.45))

    # === ZONE 2: MID EXTRAMURAL FAUBOURG & GATE-ROAD RIBBONS (medium-dense along roads & flanks) ===
    road_arr = np.asarray(all_road_pts)
    n_front_mid = len(front_bldg) + int(rng.integers(26, 36))
    for attempt in range(1800):
        if len(front_bldg) >= n_front_mid:
            break
        r_pick = rng.random()
        if r_pick < 0.46:
            # Dense-to-medium ribbons flanking both sides of the main gate road & branching lanes
            idx = int(rng.integers(2, len(road_arr)))
            rx, ry = float(road_arr[idx, 0]), float(road_arr[idx, 1])
            side = int(rng.choice([-1, 1]))
            cx = rx + side * rng.uniform(11.0, 46.0)
            cy = ry + rng.uniform(-12.0, 12.0)
        elif r_pick < 0.74:
            # South & SW extramural belt
            if hex_R is not None and rng.random() < 0.44:
                t = rng.uniform(0.14, 0.96)
                ex = (-hex_R) * (1 - t) + (-0.5 * hex_R) * t
                ey = (0.0) * (1 - t) + (-0.866 * hex_R) * t
                d = rng.uniform(14.0, 62.0)
                cx, cy = ex - 0.866 * d, ey - 0.5 * d
            else:
                cx = rng.uniform(wx0 - 30.0, wx1 + 42.0)
                cy = wy0 - rng.uniform(12.0, 64.0)
        else:
            # East / SE outer shantytown belt
            if hex_R is not None:
                t = rng.uniform(0.06, 0.92)
                ex = (0.5 * hex_R) * (1 - t) + hex_R * t
                ey = (-0.866 * hex_R) * (1 - t) + 0.0 * t
                d = rng.uniform(14.0, 62.0)
                cx, cy = ex + 0.866 * d, ey - 0.5 * d
            else:
                cx = wx1 + rng.uniform(11.0, 62.0)
                cy = rng.uniform(wy0 - 44.0, wy1 - 14.0)
        _try_place_front(cx, cy, gap=(1.2 if attempt < 1100 else 0.75), compact=(rng.random() < 0.40))

    # === ZONE 3: OUTER SCATTERED FRINGE (looser poor farmsteads/hovels tapering organically into countryside!) ===
    n_back_fringe = len(back_bldg) + int(rng.integers(4, 8))
    for _ in range(360):
        if len(back_bldg) >= n_back_fringe:
            break
        if rng.random() < 0.55:
            cx = wx0 - rng.uniform(52.0, 88.0)
            cy = rng.uniform(wy0 + 6.0, wy1 + 48.0)
        else:
            cx = rng.uniform(wx0 - 42.0, wx1 - 24.0)
            cy = wy1 + rng.uniform(44.0, 76.0)
        _try_place_back(cx, cy, gap=4.5, compact=False)

    n_front_fringe = len(front_bldg) + int(rng.integers(9, 15))
    for _ in range(600):
        if len(front_bldg) >= n_front_fringe:
            break
        r_pick = rng.random()
        if r_pick < 0.42:
            # Far end of dirt approach roads & lanes
            idx = int(rng.integers(len(road_arr) // 2, len(road_arr)))
            rx, ry = float(road_arr[idx, 0]), float(road_arr[idx, 1])
            cx = rx + rng.choice([-1, 1]) * rng.uniform(14.0, 62.0)
            cy = ry + rng.uniform(-22.0, 14.0)
        elif r_pick < 0.72:
            cx = rng.uniform(wx0 - 58.0, wx1 + 68.0)
            cy = wy0 - rng.uniform(56.0, 96.0)
        else:
            cx = wx1 + rng.uniform(54.0, 94.0)
            cy = rng.uniform(wy0 - 62.0, wy1 - 18.0)
        _try_place_front(cx, cy, gap=4.2, compact=False)

    # === EXTRAMURAL POOR-QUARTER PROPS (wells, stalls, fences, properly-scaled trees / deadwood & rubble) ===
    if not ruined:
        # 1-2 well-separated outdoor wells + 2 makeshift market stalls along the approach roads
        well_pts = []
        for want_stall in (False, True, True, False):
            if want_stall and rng.random() > 0.72:
                continue
            for _ in range(30):
                idx = int(rng.integers(2, len(road_arr)))
                sx = float(road_arr[idx, 0]) + rng.choice([-1, 1]) * rng.uniform(12.5, 22.0)
                sy = float(road_arr[idx, 1]) + rng.uniform(-4.5, 4.5)
                if want_stall:
                    sbox = (sx - 7, sy - 5, sx + 7, sy + 5)
                    if is_front(sbox) and not blocked(sbox) and not _overlaps(sbox, front_occ, 1.8):
                        front_occ.append(sbox)
                        front_items.append((sbox, ("stall", (sx - 6.5, sy - 4.2, 13.0, 8.4))))
                        break
                else:
                    if any(math.hypot(sx - wx, sy - wy) < 55.0 for wx, wy in well_pts):
                        continue
                    wbox = (sx - 6, sy - 5, sx + 6, sy + 5)
                    if is_front(wbox) and not blocked(wbox) and not _overlaps(wbox, front_occ, 2.2):
                        well_pts.append((sx, sy))
                        front_occ.append(wbox)
                        front_items.append((wbox, ("well", (sx, sy))))
                        break
        # 6-11 rustic garden fences beside outer cottages
        for _ in range(min(int(rng.integers(6, 12)), len(front_bldg))):
            b = front_bldg[int(rng.integers(0, len(front_bldg)))]
            A, B = (b[0], b[1] - 4.0), (b[2], b[1] - 4.0)
            fbox = (A[0], A[1] - 1.2, B[0], A[1] + 1.2)
            if is_front(fbox) and not blocked(fbox) and not _overlaps(fbox, front_occ, 0.7):
                front_occ.append(fbox)
                front_items.append((fbox, ("fence", (A, B))))
        # Properly scaled small urban-outskirts trees tucked around the outer cottages & fringes
        for _ in range(int(rng.integers(10, 16))):
            if not front_bldg:
                break
            b = front_bldg[int(rng.integers(0, len(front_bldg)))]
            tx, ty = rng.uniform(b[0] - 14, b[2] + 14), rng.uniform(b[1] - 12, b[3] + 10)
            tbox = (tx - 4.5, ty - 4.5, tx + 4.5, ty + 4.5)
            if is_front(tbox) and not blocked(tbox) and not _overlaps(tbox, front_occ, 2.4):
                tkind = str(rng.choice(["round", "spruce"], p=[0.66, 0.34]))
                th = rng.uniform(15.0, 21.0) if tkind == "round" else rng.uniform(17.0, 22.5)
                front_occ.append(tbox)
                front_items.append((tbox, ("tree", (tx, ty, th, tkind))))
        for _ in range(int(rng.integers(4, 8))):
            if not back_bldg:
                break
            b = back_bldg[int(rng.integers(0, len(back_bldg)))]
            tx, ty = rng.uniform(b[0] - 14, b[2] + 12), rng.uniform(b[1] - 10, b[3] + 12)
            tbox = (tx - 5.0, ty - 5.0, tx + 5.0, ty + 5.0)
            if is_back(tbox) and not blocked(tbox) and not _overlaps(tbox, back_occ, 3.0):
                tkind = str(rng.choice(["round", "spruce"], p=[0.70, 0.30]))
                th = rng.uniform(15.0, 20.5) if tkind == "round" else rng.uniform(17.0, 22.0)
                back_occ.append(tbox)
                back_items.append((tbox, ("tree", (tx, ty, th, tkind))))
        # Subtle ground grass tufts anchoring the outer fringe cottages
        for _ in range(int(rng.integers(8, 15))):
            if not front_bldg:
                break
            b = front_bldg[int(rng.integers(0, len(front_bldg)))]
            gx, gy = rng.uniform(b[0] - 4, b[2] + 4), b[1] - rng.uniform(4.8, 9.2)
            gbox = (gx - 2.4, gy - 2.4, gx + 2.4, gy + 2.4)
            if is_front(gbox) and not blocked(gbox) and not _overlaps(gbox, front_occ, 0.8):
                front_items.append((gbox, ("tuft", (gx, gy))))
    else:
        # Ruined outer suburbs: broken fences, spaced dead trees, rubble mounds & scrub
        for _ in range(min(6, len(front_bldg))):
            b = front_bldg[int(rng.integers(0, len(front_bldg)))]
            A, B = (b[0], b[1] - 4.2), (b[2], b[1] - 4.2)
            fbox = (A[0], A[1] - 1.2, B[0], A[1] + 1.2)
            if is_front(fbox) and not blocked(fbox) and not _overlaps(fbox, front_occ, 1.0):
                front_occ.append(fbox)
                front_items.append((fbox, ("rfence", (A, B))))
        for _ in range(int(rng.integers(10, 16))):
            if not front_bldg:
                break
            b = front_bldg[int(rng.integers(0, len(front_bldg)))]
            x, y = rng.uniform(b[0] - 16, b[2] + 16), rng.uniform(b[1] - 14, b[3] + 12)
            kind = str(rng.choice(["rubble", "dead_tree", "bush", "tuft"], p=[0.42, 0.22, 0.16, 0.20]))
            rad = rng.uniform(5.0, 8.2) if kind == "rubble" else (rng.uniform(4.6, 6.8) if kind == "bush" else 4.2)
            cbox = (x - rad, y - rad * 0.65, x + rad, y + rad * 0.65)
            gap = 5.5 if kind == "dead_tree" else 2.2
            if is_front(cbox) and not blocked(cbox) and not _overlaps(cbox, front_occ, gap):
                front_occ.append(cbox)
                if kind in ("rubble", "bush"):
                    front_items.append((cbox, (kind, (x, y, rad))))
                elif kind == "dead_tree":
                    front_items.append((cbox, ("dead_tree", (x, y, rng.uniform(13.0, 18.5)))))
                else:
                    front_items.append((cbox, ("tuft", (x, y))))
        for _ in range(int(rng.integers(3, 7))):
            if not back_bldg:
                break
            b = back_bldg[int(rng.integers(0, len(back_bldg)))]
            tx, ty = rng.uniform(b[0] - 15, b[2] + 15), rng.uniform(b[1] - 10, b[3] + 14)
            tbox = (tx - 5.0, ty - 5.0, tx + 5.0, ty + 5.0)
            if is_back(tbox) and not blocked(tbox) and not _overlaps(tbox, back_occ, 5.5):
                back_occ.append(tbox)
                back_items.append((tbox, ("dead_tree", (tx, ty, rng.uniform(13.0, 18.5)))))

    return roads, back_items, front_items


def _draw_compact_guildhall(pen, P, rng, scale=0.80):
    """Scaled urban clocktower guildhall for city centers."""
    W, D, H = 48.0 * scale, 22.0 * scale, 22.0 * scale
    R = D * 0.62
    X0, Y0 = -W * 0.5, -D * 0.5
    pl = dict(kind="house", W=W, D=D, H=H, R=R, ridge_x=True, floors=2,
              timber=False, plinth=True, quoins=True, arch_door=True,
              dormer=False, chimney=False, smoke=False, leanto=False, cross=True,
              roof="tile", win=(2.8 * scale, 4.4 * scale))
    draw_house(pen, P, X0, Y0, pl)
    ts = 7.4 * scale
    z_b = H + R * 0.35
    th = z_b + 30.0 * scale
    sp_h = 26.0 * scale
    _draw_clocktower(pen, P, rng, -ts, ts, -ts, ts, z_b, th, sp_h, flag=True)


def _draw_inner_citadel(pen, P, rng, cx0, cy0, cx1, cy1, H_in, t_in, round_t, cap_r, cap_s, role="keep"):
    """Fortified inner castle / citadel layer inside the middle city: taller stone walls, 4 corner turrets,
    twin-turret barbican gatehouse, and a soaring royal keep, spire-fortress, or grand cathedral inside!"""
    cw_in, cd_in = cx1 - cx0, cy1 - cy0
    cx_mid = (cx0 + cx1) * 0.5
    gw_in = 12.2

    def draw_itower(X, Y, flag=True):
        if round_t:
            intact_round_tower(pen, P, rng, X, Y, 7.6, H_in * 1.62, cap=cap_r, flag=flag)
        else:
            intact_square_tower(pen, P, rng, X, Y, 7.0, H_in * 1.56, cap=cap_s, flag=flag)

    # 1. North back wall + NW/NE turrets + West wall
    draw_itower(cx0, cy1)
    intact_curtain(pen, P, (cx0, cy1), (cx1, cy1), H_in, t_in, rng, outer_side="north")
    draw_itower(cx1, cy1)
    intact_curtain(pen, P, (cx0, cy0), (cx0, cy1), H_in, t_in, rng, outer_side="west")

    # 2. Royal structures inside the inner citadel walls (sorted strictly back-to-front!)
    pad = t_in + 3.5
    ix0, iy0, ix1, iy1 = cx0 + pad, cy0 + pad, cx1 - pad, cy1 - pad
    inner_items, inner_occ = [], []

    if role == "cathedral":
        ccx = cx_mid + rng.uniform(-3.0, 3.0)
        ccy = iy0 + (iy1 - iy0) * 0.54
        sc_c = min(0.76, (ix1 - ix0) / 84.0)
        c_fp = (ccx - 28 * sc_c / 0.76, ccy - 14, ccx + 30 * sc_c / 0.76, ccy + 14)
        inner_occ.append(c_fp)
        inner_items.append((c_fp, ("cathedral", (ccx, ccy, sc_c))))
        # Companion royal spire/belfry turret in an open rear corner if clear
        for tx_c in (ix0 + 8.5, ix1 - 8.5):
            ty_c = iy1 - 8.5
            t_fp = (tx_c - 7.5, ty_c - 7.5, tx_c + 7.5, ty_c + 7.5)
            if not _overlaps(t_fp, inner_occ, 1.5):
                inner_occ.append(t_fp)
                inner_items.append((t_fp, ("tower", (tx_c, ty_c, 6.6, H_in * 2.05, "cone", True))))
                break
    elif role == "spire_keep":
        # Soaring grand round palace keep-tower + companion royal bailey hall inside the inner walls!
        tr = 12.6
        side = int(rng.choice([-1, 1]))
        tx = cx_mid + side * max(12.0, (ix1 - ix0) * 0.19)
        tx = min(max(tx, ix0 + tr + 2.0), ix1 - tr - 2.0)
        ty = iy1 - tr - 3.5
        t_fp = (tx - tr - 1.5, ty - tr - 1.5, tx + tr + 1.5, ty + tr + 1.5)
        inner_occ.append(t_fp)
        inner_items.append((t_fp, ("tower", (tx, ty, tr, rng.uniform(66, 78), "cone", True))))
        # Companion royal keep-annex / bailey hall tucked cleanly on the opposite side of the citadel courtyard
        kw = min(rng.uniform(24, 30), max(20.0, (ix1 - ix0) - 2 * tr - 10.0))
        kd = min(rng.uniform(18, 24), (iy1 - iy0) * 0.56)
        kh = rng.uniform(42, 52)
        kx = (ix0 + 2.5) if side > 0 else (ix1 - kw - 2.5)
        ky = iy1 - kd - rng.uniform(2.5, max(3.0, (iy1 - iy0 - kd) * 0.42))
        k_fp = (kx - 1.2, ky - 1.2, kx + kw + 1.2, ky + kd + 1.2)
        if kx >= ix0 and kx + kw <= ix1 and not _overlaps(k_fp, inner_occ, 1.2):
            inner_occ.append(k_fp)
            inner_items.append((k_fp, ("keep", (kx, ky, kw, kd, kh))))
    else:  # "keep" — towering royal fortress keep + companion watch-spire inside the inner walls!
        kw = min(rng.uniform(42, 50), (ix1 - ix0) * 0.62)
        kd = min(rng.uniform(30, 36), (iy1 - iy0) * 0.72)
        kh = rng.uniform(68, 82)
        kx = cx_mid - kw * 0.5 + rng.uniform(-10, 10)
        kx = min(max(kx, ix0 + 2.0), ix1 - kw - 2.0)
        ky = iy1 - kd - rng.uniform(2.0, max(2.5, (iy1 - iy0 - kd) * 0.38))
        k_fp = (kx - 2.0, ky - 2.0, kx + kw + 2.0, ky + kd + 2.0)
        inner_occ.append(k_fp)
        inner_items.append((k_fp, ("keep", (kx, ky, kw, kd, kh))))
        # Companion tall slender spire turret in the open side of the inner bailey
        tx = (ix1 - 8.5) if (kx + kw * 0.5) <= cx_mid else (ix0 + 8.5)
        ty = iy1 - rng.uniform(9.5, max(10.0, (iy1 - iy0) * 0.45))
        t_fp = (tx - 7.5, ty - 7.5, tx + 7.5, ty + 7.5)
        if not _overlaps(t_fp, inner_occ, 1.5):
            inner_occ.append(t_fp)
            inner_items.append((t_fp, ("tower", (tx, ty, 6.6, H_in * 2.08, "cone", True))))

    _draw_city_interior(pen, P, rng, inner_items)

    # 3. East wall + SW/SE turrets + South wall & twin-turret inner barbican gatehouse
    intact_curtain(pen, P, (cx1, cy0), (cx1, cy1), H_in, t_in, rng, outer_side="east")
    draw_itower(cx0, cy0)
    intact_curtain(pen, P, (cx0, cy0), (cx_mid - gw_in * 0.5 - 4.6, cy0), H_in, t_in, rng, outer_side="south")
    intact_gatehouse(pen, P, rng, cx_mid, cy0, gw_in, H_in, t_in,
                     style=("round" if round_t else "square"), drawbridge=False, flags=True)
    intact_curtain(pen, P, (cx_mid + gw_in * 0.5 + 4.6, cy0), (cx1, cy0), H_in, t_in, rng, outer_side="south")
    draw_itower(cx1, cy0)


def city(pen, P, rng):
    """Fortified medieval fantasy city: rich walled inner city (tall townhouses, keep/cathedral/guildhall) + poor outer suburbs."""
    archetype = str(rng.choice(["walled_citadel", "cathedral_city", "hex_walled", "open_bastion"],
                               p=[0.32, 0.28, 0.24, 0.16]))
    H_w = rng.uniform(19, 24)
    t_w = rng.uniform(4.6, 5.6)
    round_t = rng.random() < 0.72
    cap_r = "cone" if rng.random() < 0.72 else "crenel"
    cap_s = "pyramid" if rng.random() < 0.52 else "crenel"

    def draw_btower(X, Y, sc=0.82, flag=True):
        if round_t:
            intact_round_tower(pen, P, rng, X, Y, 11.2 * sc, 39.0 * sc, cap=cap_r, flag=flag)
        else:
            intact_square_tower(pen, P, rng, X, Y, 10.2 * sc, 37.0 * sc, cap=cap_s, flag=flag)

    if archetype == "hex_walled":
        # Grand polygonal 6-walled rich inner metropolis (~2x area!) + sprawling poor outer suburbs!
        R = rng.uniform(156, 174)
        vs = [(R * math.cos(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3),
               R * math.sin(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3)) for k in range(6)]
        v0, v1, v2, v3, v4, v5 = vs
        mid = lambda a, b: ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        m43, m54, m32, m50, m21 = mid(v4, v3), mid(v5, v4), mid(v3, v2), mid(v5, v0), mid(v2, v1)
        gw = rng.uniform(16, 19)
        y_f = v0[1]

        # Plan dense wall-hugging outer poor suburbs (tight wall exclusion + exact bastion tower circles!)
        h_excl = R * 0.866 + 3.6
        def excl_hex(ax, ay, bx, by):
            for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by), ((ax + bx) * 0.5, (ay + by) * 0.5)):
                if abs(py) < h_excl and (abs(px) * 0.866 + abs(py) * 0.5) < h_excl:
                    return True
            return False

        hex_towers = [(x, y, 10.8) for x, y in (v0, v1, v2, v3, v4, v5)] + [
            (x, y, 9.8) for x, y in (m43, m54, m32, m50, m21)
        ] + [(-gw * 0.5 - 6.5, y_f, 10.5), (gw * 0.5 + 6.5, y_f, 10.5)]
        hug_back = [
            (v4, v3, 0.0, 1.0),
            (v5, v4, -0.866, 0.5),
            (v3, v2, 0.866, 0.5),
        ]
        hug_front = [
            (v0, v1, 0.0, -1.0),
            (v1, v2, 0.866, -0.5),
            (v5, v0, -0.866, -0.5),
        ]

        roads, back_sub, front_sub = _plan_outer_suburbs(
            rng, excl_hex, (-R, y_f, R, -y_f), y_f, gw, ruined=False, hex_R=R,
            tower_pts=hex_towers, hug_back=hug_back, hug_front=hug_front,
        )
        for r_pts in roads:
            _draw_road_strip(pen, P, rng, r_pts, half_w=3.4)
        _draw_city_interior(pen, P, rng, back_sub)

        # Back hex walls (strict decreasing-depth order with mid-wall bastion towers!)
        draw_btower(*v4, sc=0.80)
        intact_curtain_seg(pen, P, v4, m43, H_w, t_w, rng, show_inner=True)
        draw_btower(*m43, sc=0.72, flag=False)
        intact_curtain_seg(pen, P, m43, v3, H_w, t_w, rng, show_inner=True)
        draw_btower(*v3, sc=0.80)
        intact_curtain_seg(pen, P, m54, v4, H_w, t_w, rng, show_inner=True)
        draw_btower(*m54, sc=0.72, flag=False)
        intact_curtain_seg(pen, P, v5, m54, H_w, t_w, rng, show_inner=True)
        draw_btower(*v5, sc=0.80)

        in_R = R * 0.866 - t_w - 2.2
        def inside_hex(x0, y0, x1, y1):
            for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                if abs(cy) > in_R - 1.5:
                    return False
                if abs(cx) * 0.866 + abs(cy) * 0.5 > in_R - 1.5:
                    return False
            return True

        items, occ = [], []
        # Inner Walled Castle / Citadel tier crowning the heart of the hexagonal city!
        H_in = H_w * rng.uniform(1.22, 1.36)
        t_in = t_w * 0.94
        cw_in, cd_in = rng.uniform(82, 94), rng.uniform(56, 64)
        cx_c = rng.uniform(-16, 16)
        cy_c = rng.uniform(22, 34)
        cx0, cy0, cx1, cy1 = cx_c - cw_in * 0.5, cy_c - cd_in * 0.5, cx_c + cw_in * 0.5, cy_c + cd_in * 0.5
        c_role = str(rng.choice(["keep", "cathedral", "spire_keep"], p=[0.46, 0.34, 0.20]))
        cit_fp = (cx0 - 9.5, cy0 - 9.5, cx1 + 9.5, cy1 + 9.5)
        occ.append(cit_fp)
        occ.append((cx_c - 14.0, cy0 - 20.0, cx_c + 14.0, cy0 - 8.5))  # clear apron in front of inner barbican!
        items.append((cit_fp, ("citadel", (cx0, cy0, cx1, cy1, H_in, t_in, round_t, cap_r, cap_s, c_role))))

        # Secondary middle-city civic landmark outside the inner castle walls
        gx_h = rng.uniform(-82, -56) if cx_c > -4 else rng.uniform(56, 82)
        gy_h = rng.uniform(-34, 14)
        g_fp = (gx_h - 21, gy_h - 11, gx_h + 21, gy_h + 11)
        if inside_hex(*g_fp) and not _overlaps(g_fp, occ, 2.5):
            occ.append(g_fp)
            items.append((g_fp, ("guildhall", (gx_h, gy_h, 0.74))))

        th_items, occ = _pack_dense_urban_buildings(
            rng, inside_hex, int(rng.integers(68, 82)), occ, scale=0.72,
            x_lim=(-in_R, in_R), y_lim=(-in_R, in_R),
        )
        items.extend(th_items)
        _draw_city_interior(pen, P, rng, items)

        # Front hex walls (strict decreasing-depth order!)
        intact_curtain_seg(pen, P, v3, m32, H_w, t_w, rng, show_inner=False)
        draw_btower(*m32, sc=0.72, flag=False)
        intact_curtain_seg(pen, P, m32, v2, H_w, t_w, rng, show_inner=False)
        draw_btower(*v2, sc=0.80)
        intact_curtain_seg(pen, P, v5, m50, H_w, t_w, rng, show_inner=False)
        draw_btower(*m50, sc=0.72, flag=False)
        intact_curtain_seg(pen, P, m50, v0, H_w, t_w, rng, show_inner=False)
        draw_btower(*v0, sc=0.80)
        intact_curtain_seg(pen, P, v2, m21, H_w, t_w, rng, show_inner=False)
        draw_btower(*m21, sc=0.72, flag=False)
        intact_curtain_seg(pen, P, m21, v1, H_w, t_w, rng, show_inner=False)
        intact_curtain(pen, P, v0, (-gw * 0.5 - 6.0, y_f), H_w, t_w, rng, outer_side="south")
        intact_gatehouse(pen, P, rng, 0.0, y_f, gw, H_w, t_w, style=("round" if round_t else "square"))
        intact_curtain(pen, P, (gw * 0.5 + 6.0, y_f), v1, H_w, t_w, rng, outer_side="south")
        draw_btower(*v1, sc=0.80)
        _draw_city_interior(pen, P, rng, front_sub)
        return

    if archetype == "open_bastion":
        # Grand crowded urban hill-city (~2x area!) behind a 3-sided bastion wall + sprawling poor cottages outside!
        CW, CD = rng.uniform(268, 302), rng.uniform(198, 224)
        x0, y0, x1, y1 = -CW * 0.5, -CD * 0.5, CW * 0.5, CD * 0.5
        wf_x0, wf_x1 = -CW * 0.44, CW * 0.44
        y_arm = 22.0
        y_mid = (y0 + y_arm) * 0.5
        gw = rng.uniform(16, 19)

        def excl_ob(ax, ay, bx, by):
            return not (bx < wf_x0 - 3.6 or ax > wf_x1 + 3.6 or by < y0 - 3.6 or ay > y1 + 4.0)

        ms_l, ms_r = wf_x0 * 0.52, wf_x1 * 0.52
        ob_towers = [
            (wf_x0, y_arm, 10.5), (wf_x0, y_mid, 10.0), (wf_x0, y0, 11.2),
            (wf_x1, y_arm, 10.5), (wf_x1, y_mid, 10.0), (wf_x1, y0, 11.2),
            (ms_l, y0, 10.2), (ms_r, y0, 10.2),
            (-gw * 0.5 - 6.5, y0, 10.8), (gw * 0.5 + 6.5, y0, 10.8),
        ]
        hug_back = [((wf_x0, y0), (wf_x0, y_arm), -1.0, 0.0), ((wf_x0 + 18, y1), (wf_x1 - 18, y1), 0.0, 1.0)]
        hug_front = [((wf_x0, y0), (wf_x1, y0), 0.0, -1.0), ((wf_x1, y0), (wf_x1, y_arm), 1.0, 0.0)]

        roads, back_sub, front_sub = _plan_outer_suburbs(
            rng, excl_ob, (wf_x0, y0, wf_x1, y1), y0, gw, ruined=False,
            tower_pts=ob_towers, hug_back=hug_back, hug_front=hug_front,
        )
        for r_pts in roads:
            _draw_road_strip(pen, P, rng, r_pts, half_w=3.4)
        _draw_city_interior(pen, P, rng, back_sub)

        draw_btower(wf_x0, y_arm, sc=0.78, flag=False)
        intact_curtain(pen, P, (wf_x0, y_mid), (wf_x0, y_arm), H_w, t_w, rng, outer_side="west")
        draw_btower(wf_x0, y_mid, sc=0.74, flag=False)
        intact_curtain(pen, P, (wf_x0, y0), (wf_x0, y_mid), H_w, t_w, rng, outer_side="west")

        items, occ = [], []
        # Inner Walled Hilltop Castle / Citadel crowning the open-bastion hill-city!
        H_in = H_w * rng.uniform(1.22, 1.36)
        t_in = t_w * 0.94
        cw_in, cd_in = rng.uniform(86, 100), rng.uniform(58, 66)
        cx_c = rng.uniform(-24, 24)
        cy_c = rng.uniform(24, 38)
        cx0, cy0, cx1, cy1 = cx_c - cw_in * 0.5, cy_c - cd_in * 0.5, cx_c + cw_in * 0.5, cy_c + cd_in * 0.5
        c_role = str(rng.choice(["keep", "spire_keep"], p=[0.70, 0.30]))
        cit_fp = (cx0 - 9.5, cy0 - 9.5, cx1 + 9.5, cy1 + 9.5)
        occ.append(cit_fp)
        occ.append((cx_c - 14.0, cy0 - 20.0, cx_c + 14.0, cy0 - 8.5))
        items.append((cit_fp, ("citadel", (cx0, cy0, cx1, cy1, H_in, t_in, round_t, cap_r, cap_s, c_role))))

        # Middle-city guildhall / cathedral flanking below the hilltop citadel
        gx_h = rng.uniform(44, 82) if cx_c < 0 else rng.uniform(-82, -44)
        gy_h = rng.uniform(-34, 8)
        g_fp = (gx_h - 21, gy_h - 11, gx_h + 21, gy_h + 11)
        if not _overlaps(g_fp, occ, 2.5):
            occ.append(g_fp)
            items.append((g_fp, ("guildhall", (gx_h, gy_h, 0.74))))

        if rng.random() < 0.55:
            cx, cy = (rng.uniform(-80, -44) if gx_h > 0 else rng.uniform(44, 80)), rng.uniform(-44, -12)
            c_fp = (cx - 31, cy - 15, cx + 33, cy + 15)
            if not _overlaps(c_fp, occ, 2.5) and wf_x0 + 12 < c_fp[0] and c_fp[2] < wf_x1 - 12:
                occ.append(c_fp)
                items.append((c_fp, ("cathedral", (cx, cy, 0.72))))

        def inside_oval(ax, ay, bx, by):
            if ay < y0 + t_w + 3.5 or ax < wf_x0 + t_w + 2.5 or bx > wf_x1 - t_w - 2.5:
                return False
            for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by)):
                if (px / (CW * 0.48)) ** 2 + (py / (CD * 0.52)) ** 2 > 1.0:
                    return False
            return True

        th_items, occ = _pack_dense_urban_buildings(
            rng, inside_oval, int(rng.integers(56, 70)), occ, scale=0.74,
            x_lim=(wf_x0 + t_w + 3, wf_x1 - t_w - 3), y_lim=(y0 + t_w + 3, y1),
        )
        items.extend(th_items)
        _draw_city_interior(pen, P, rng, items)

        draw_btower(wf_x1, y_arm, sc=0.78, flag=False)
        intact_curtain(pen, P, (wf_x1, y_mid), (wf_x1, y_arm), H_w, t_w, rng, outer_side="east")
        draw_btower(wf_x1, y_mid, sc=0.74, flag=False)
        intact_curtain(pen, P, (wf_x1, y0), (wf_x1, y_mid), H_w, t_w, rng, outer_side="east")

        ms_l, ms_r = wf_x0 * 0.52, wf_x1 * 0.52
        draw_btower(wf_x0, y0, sc=0.84)
        intact_curtain(pen, P, (wf_x0, y0), (ms_l, y0), H_w, t_w, rng, outer_side="south")
        draw_btower(ms_l, y0, sc=0.76, flag=False)
        intact_curtain(pen, P, (ms_l, y0), (-gw * 0.5 - 6.0, y0), H_w, t_w, rng, outer_side="south")
        intact_gatehouse(pen, P, rng, 0.0, y0, gw, H_w, t_w, style=("round" if round_t else "square"))
        intact_curtain(pen, P, (gw * 0.5 + 6.0, y0), (ms_r, y0), H_w, t_w, rng, outer_side="south")
        draw_btower(ms_r, y0, sc=0.76, flag=False)
        intact_curtain(pen, P, (ms_r, y0), (wf_x1, y0), H_w, t_w, rng, outer_side="south")
        draw_btower(wf_x1, y0, sc=0.84)
        _draw_city_interior(pen, P, rng, front_sub)
        return

    # Grand Walled Citadel City OR Cathedral Walled City (~2x area!): rich inner metropolis + sprawling poor outer suburbs!
    CW = rng.uniform(262, 296)
    CD = rng.uniform(192, 218)
    x0, y0, x1, y1 = -CW * 0.5, -CD * 0.5, CW * 0.5, CD * 0.5
    gw = rng.uniform(16, 19)

    def excl_rect(ax, ay, bx, by):
        return not (bx < x0 - 3.6 or ax > x1 + 3.6 or by < y0 - 3.6 or ay > y1 + 3.6)

    nx1, nx2 = x0 + CW / 3.0, x0 + 2.0 * CW / 3.0
    sx_l, sx_r = x0 * 0.52, x1 * 0.52
    rect_towers = [
        (x0, y1, 11.2), (nx1, y1, 10.2), (nx2, y1, 10.2), (x1, y1, 11.2),
        (x0, 0.0, 10.2), (x1, 0.0, 10.2),
        (x0, y0, 11.2), (sx_l, y0, 10.2), (sx_r, y0, 10.2), (x1, y0, 11.2),
        (-gw * 0.5 - 6.5, y0, 10.8), (gw * 0.5 + 6.5, y0, 10.8),
    ]
    hug_back = [((x0, y0), (x0, y1), -1.0, 0.0), ((x0, y1), (x1, y1), 0.0, 1.0)]
    hug_front = [((x0, y0), (x1, y0), 0.0, -1.0), ((x1, y0), (x1, y1), 1.0, 0.0)]

    roads, back_sub, front_sub = _plan_outer_suburbs(
        rng, excl_rect, (x0, y0, x1, y1), y0, gw, ruined=False,
        tower_pts=rect_towers, hug_back=hug_back, hug_front=hug_front,
    )
    for r_pts in roads:
        _draw_road_strip(pen, P, rng, r_pts, half_w=3.4)
    _draw_city_interior(pen, P, rng, back_sub)

    # 1. Back wall (3 bays, 2 intermediate towers!) & west wall (2 bays)
    draw_btower(x0, y1, sc=0.82)
    intact_curtain(pen, P, (x0, y1), (nx1, y1), H_w, t_w, rng, outer_side="north")
    draw_btower(nx1, y1, sc=0.76, flag=False)
    intact_curtain(pen, P, (nx1, y1), (nx2, y1), H_w, t_w, rng, outer_side="north")
    draw_btower(nx2, y1, sc=0.76, flag=False)
    intact_curtain(pen, P, (nx2, y1), (x1, y1), H_w, t_w, rng, outer_side="north")
    draw_btower(x1, y1, sc=0.82)
    intact_curtain(pen, P, (x0, 0.0), (x0, y1), H_w, t_w, rng, outer_side="west")
    draw_btower(x0, 0.0, sc=0.76, flag=False)
    intact_curtain(pen, P, (x0, y0), (x0, 0.0), H_w, t_w, rng, outer_side="west")

    # 2. 3-tier heart: Inner Walled Castle/Citadel + middle-city landmarks & ~58-72 rich townhouses!
    items, occ = [], []
    H_in = H_w * rng.uniform(1.22, 1.36)
    t_in = t_w * 0.94
    cw_in, cd_in = rng.uniform(88, 104), rng.uniform(58, 68)
    cx_c = rng.uniform(-34, 34)
    cy1 = rng.uniform(y1 - 36.0, y1 - 21.0)
    cy0 = cy1 - cd_in
    cx0, cx1 = cx_c - cw_in * 0.5, cx_c + cw_in * 0.5
    c_role = ("cathedral" if archetype == "cathedral_city" else str(rng.choice(["keep", "spire_keep"], p=[0.72, 0.28])))
    cit_fp = (cx0 - 9.5, cy0 - 9.5, cx1 + 9.5, cy1 + 9.5)
    occ.append(cit_fp)
    occ.append((cx_c - 14.0, cy0 - 20.0, cx_c + 14.0, cy0 - 8.5))  # clear cobbled plaza before inner castle gate!
    items.append((cit_fp, ("citadel", (cx0, cy0, cx1, cy1, H_in, t_in, round_t, cap_r, cap_s, c_role))))

    # Secondary rich middle-city landmarks outside the inner castle walls
    gx_h = rng.uniform(56, 98) if cx_c < 0 else rng.uniform(-98, -56)
    gy_h = rng.uniform(-24, 38)
    g_fp = (gx_h - 21, gy_h - 11, gx_h + 21, gy_h + 11)
    if not _overlaps(g_fp, occ, 2.5) and x0 + 12 < g_fp[0] and g_fp[2] < x1 - 12:
        occ.append(g_fp)
        items.append((g_fp, ("guildhall", (gx_h, gy_h, 0.74))))

    if archetype == "walled_citadel" and rng.random() < 0.62:
        cx, cy = (rng.uniform(-92, -46) if gx_h > 0 else rng.uniform(46, 92)), rng.uniform(-46, -12)
        c_fp = (cx - 31, cy - 15, cx + 33, cy + 15)
        if not _overlaps(c_fp, occ, 2.5) and x0 + 12 < c_fp[0] and c_fp[2] < x1 - 12 and c_fp[1] > y0 + 22:
            occ.append(c_fp)
            items.append((c_fp, ("cathedral", (cx, cy, 0.72))))
    else:
        for _ in range(4):
            tx = rng.uniform(x0 + 26, x1 - 26)
            ty = rng.uniform(y0 + 28, y1 - 24)
            t_fp = (tx - 9, ty - 9, tx + 9, ty + 9)
            if not _overlaps(t_fp, occ, 3.0):
                occ.append(t_fp)
                items.append((t_fp, ("tower", (tx, ty, 7.8, 46.0, "cone", True))))
                break

    pad_x = t_w + 3.5
    pad_y = t_w + 3.5
    gate_clr = (-12.0, y0, 12.0, y0 + 18.0)

    def inside_rect(ax, ay, bx, by):
        if ax < x0 + pad_x or bx > x1 - pad_x or ay < y0 + pad_y or by > y1 - pad_y:
            return False
        if not (bx < gate_clr[0] or ax > gate_clr[2] or by < gate_clr[1] or ay > gate_clr[3]):
            return False
        return True

    th_items, occ = _pack_dense_urban_buildings(
        rng, inside_rect, int(rng.integers(58, 72)), occ, scale=0.72,
        x_lim=(x0 + pad_x, x1 - pad_x), y_lim=(y0 + pad_y, y1 - pad_y),
    )
    items.extend(th_items)
    for mx_s in (-26.0, -12.0, 6.0, 20.0):
        s_fp = (mx_s, y0 + pad_y + 2.0, mx_s + 12.0, y0 + pad_y + 10.0)
        if not _overlaps(s_fp, occ, 0.5):
            occ.append(s_fp)
            items.append((s_fp, ("stall", (mx_s, y0 + pad_y + 2.0, 12.0, 8.0))))
    _draw_city_interior(pen, P, rng, items)

    # 3. East wall (2 bays) + south wall (flanking bastion towers + grand twin-towered gatehouse)
    intact_curtain(pen, P, (x1, 0.0), (x1, y1), H_w, t_w, rng, outer_side="east")
    draw_btower(x1, 0.0, sc=0.76, flag=False)
    intact_curtain(pen, P, (x1, y0), (x1, 0.0), H_w, t_w, rng, outer_side="east")

    sx_l, sx_r = x0 * 0.52, x1 * 0.52
    draw_btower(x0, y0, sc=0.82)
    intact_curtain(pen, P, (x0, y0), (sx_l, y0), H_w, t_w, rng, outer_side="south")
    draw_btower(sx_l, y0, sc=0.76, flag=False)
    intact_curtain(pen, P, (sx_l, y0), (-gw * 0.5 - 6.0, y0), H_w, t_w, rng, outer_side="south")
    intact_gatehouse(pen, P, rng, 0.0, y0, gw, H_w, t_w, style=("round" if round_t else "square"))
    intact_curtain(pen, P, (gw * 0.5 + 6.0, y0), (sx_r, y0), H_w, t_w, rng, outer_side="south")
    draw_btower(sx_r, y0, sc=0.76, flag=False)
    intact_curtain(pen, P, (sx_r, y0), (x1, y0), H_w, t_w, rng, outer_side="south")
    draw_btower(x1, y0, sc=0.82)

    # 4. Outer poor suburbs (south gate roads & east flank) cleanly in front of the walls!
    _draw_city_interior(pen, P, rng, front_sub)


# --------------------------------------------------------------------------- ruined townhouses & ruined cities

def plan_ruined_townhouse(rng, scale=1.0, severity=None, force_ridge_x=None):
    """Plan a tall, gutted 2-3 storey ruined urban townhouse shell compatible with `ruins.draw_house`."""
    sev = rng.uniform(0.28, 0.88) if severity is None else severity
    ridge_x = (rng.random() < 0.45) if force_ridge_x is None else force_ridge_x
    if ridge_x:
        W = rng.uniform(26, 36) * scale
        D = rng.uniform(18, 24) * scale
    else:
        W = rng.uniform(19, 26) * scale
        D = rng.uniform(26, 35) * scale
    H = rng.uniform(22, 32) * scale
    span = (D if ridge_x else W) * 0.5
    R = span * rng.uniform(1.05, 1.38)
    t = max(1.6, rng.uniform(2.2, 2.9) * scale)
    p = [0.42, 0.34, 0.24] if sev < 0.50 else [0.16, 0.36, 0.48]
    roof = str(rng.choice(["partial", "rafters", "none"], p=p))
    Lr = W if ridge_x else D
    hi_broken = rng.random() < 0.5
    roof_u = protect = None
    if roof != "none":
        sp_r = rng.uniform(0.35, 0.68) if roof == "partial" else rng.uniform(0.42, 0.88)
        roof_u = (0.0, sp_r) if hi_broken else (1.0 - sp_r, 1.0)
        protect = (roof_u[0] * Lr - 2.5, roof_u[1] * Lr + 2.5)
    allw = (-1e9, 1e9)
    g_lo = allw if roof_u and roof_u[0] == 0.0 else None
    g_hi = allw if roof_u and roof_u[1] == 1.0 else None

    def dp(L, gable, prot):
        return damage_profile(rng, L, H, R, gable, sev, prot)

    if ridge_x:
        walls = dict(front=dp(W, False, protect), back=dp(W, False, protect),
                     left=dp(D, True, g_lo), right=dp(D, True, g_hi))
    else:
        walls = dict(left=dp(D, False, protect), right=dp(D, False, protect),
                     front=dp(W, True, g_lo), back=dp(W, True, g_hi))

    def stacked_ops(L, door=True):
        ops = []
        ww = rng.uniform(2.8, 3.8) * scale
        wh = rng.uniform(4.2, 5.4) * scale
        n_col = max(1, int(L / (10.5 * scale)))
        cols = np.linspace(0.0, L, n_col + 2)[1:-1]
        di = int(rng.integers(0, n_col)) if door else -1
        for i, s in enumerate(cols):
            if i == di:
                ops.append((s - ww * 0.68, s + ww * 0.68, 0.0, min(H * 0.38, wh * 1.55)))
            else:
                ops.append((s - ww * 0.5, s + ww * 0.5, H * 0.14, H * 0.14 + wh))
            # Second-story urban window row
            ops.append((s - ww * 0.5, s + ww * 0.5, H * 0.54, H * 0.54 + wh))
        return ops

    return dict(
        W=W, D=D, H=H, R=R, t=t, k=scale, sev=sev, ridge_x=ridge_x,
        roof=roof, roof_u=roof_u, hi_broken=hi_broken, walls=walls,
        ops=dict(front=stacked_ops(W, True), right=stacked_ops(D, False)),
        keep=False, arched=(rng.random() < 0.45),
        chimney=(rng.random() < 0.52), ivy=(rng.random() < 0.50),
    )


def ruined_townhouse(pen, P, rng):
    """Standalone ruined urban townhouse: gutted tall single, attached ruined row-house pair, or ruined corner shell."""
    mode = str(rng.choice(["single", "row_pair", "corner_l"], p=[0.46, 0.36, 0.18]))
    items = []
    if mode == "single":
        pl = plan_ruined_townhouse(rng, scale=1.08)
        X0, Y0 = -pl["W"] * 0.5, -pl["D"] * 0.5
        draw_ruined_house(pen, P, X0, Y0, pl, exterior=True)
        return
    if mode == "row_pair":
        along_x = rng.random() < 0.72
        pls = [plan_ruined_townhouse(rng, scale=0.98, force_ridge_x=(not along_x if rng.random() < 0.72 else None))
               for _ in range(2)]
        if along_x:
            cur_x = -sum(p["W"] for p in pls) * 0.5
            for pl in pls:
                y0 = -pl["D"] * 0.5 + rng.uniform(-1.2, 1.2)
                items.append(((cur_x, y0, cur_x + pl["W"], y0 + pl["D"]), (cur_x, y0, pl)))
                cur_x += pl["W"] + 0.5
        else:
            cur_y = -sum(p["D"] for p in pls) * 0.5
            for pl in pls:
                x0 = -pl["W"] * 0.5 + rng.uniform(-1.2, 1.2)
                items.append(((x0, cur_y, x0 + pl["W"], cur_y + pl["D"]), (x0, cur_y, pl)))
                cur_y += pl["D"] + 0.5
    else:
        pl1 = plan_ruined_townhouse(rng, scale=0.96, force_ridge_x=True)
        pl2 = plan_ruined_townhouse(rng, scale=0.92, force_ridge_x=False)
        X1, Y1 = -pl1["W"] * 0.5, 0.0
        west_w = rng.random() < 0.55
        X2 = X1 if west_w else X1 + pl1["W"] - pl2["W"]
        Y2 = Y1 - pl2["D"] - 0.5
        items.append(((X1, Y1, X1 + pl1["W"], Y1 + pl1["D"]), (X1, Y1, pl1)))
        items.append(((X2, Y2, X2 + pl2["W"], Y1), (X2, Y2, pl2)))

    # Add foreground rubble strictly south of the combined bounding box
    all_b = [it[0] for it in items]
    bx0, by0, bx1 = min(b[0] for b in all_b), min(b[1] for b in all_b), max(b[2] for b in all_b)
    for X0, Y0, pl in sort_boxes(P, items):
        draw_ruined_house(pen, P, X0, Y0, pl, exterior=False)
    for _ in range(int(rng.integers(2, 4))):
        rx, ry = rng.uniform(bx0 + 2, bx1 - 2), by0 - rng.uniform(6.0, 12.0)
        pen.rubble(*P(rx, ry), rng.uniform(5.5, 9.5), rng.uniform(2.6, 4.4))
    for _ in range(int(rng.integers(2, 5))):
        pen.tuft(*P(rng.uniform(bx0, bx1), by0 - rng.uniform(3.0, 9.5)), rng.uniform(3.2, 5.2))


def _pack_dense_ruined_buildings(rng, inside_fn, target_n, existing_boxes, scale=0.72,
                                 x_lim=(-100, 100), y_lim=(-75, 75), sev_base=0.52):
    """Pack `target_n` densely clustered, overlapping gutted ruined townhouses inside `inside_fn`."""
    placed = []
    occ = list(existing_boxes)

    # 1. Seed flush 2-4 facade ruined row-house terraces
    for _ in range(160):
        if len(placed) >= int(target_n * 0.55):
            break
        n_row = int(rng.choice([2, 3, 4], p=[0.48, 0.36, 0.16]))
        along_x = rng.random() < 0.72
        pls = [plan_ruined_townhouse(rng, scale=scale * rng.uniform(0.90, 1.06),
                                     severity=np.clip(sev_base + rng.uniform(-0.22, 0.28), 0.22, 0.92),
                                     force_ridge_x=(not along_x if rng.random() < 0.72 else None))
               for _ in range(n_row)]
        cx = rng.uniform(x_lim[0] * 0.90, x_lim[1] * 0.90)
        cy = rng.uniform(y_lim[0] * 0.90, y_lim[1] * 0.90)
        cand_items, cand_fps = [], []
        ok = True
        if along_x:
            cur_x = cx - sum(p["W"] for p in pls) * 0.5
            for pl in pls:
                y0 = cy - pl["D"] * 0.5 + rng.uniform(-0.8, 0.8)
                fp = (cur_x, y0, cur_x + pl["W"], y0 + pl["D"])
                if not inside_fn(*fp) or _overlaps(fp, occ, 0.45):
                    ok = False
                    break
                cand_items.append((fp, ("rhouse", (cur_x, y0, pl))))
                cand_fps.append(fp)
                cur_x += pl["W"] + 0.4
        else:
            cur_y = cy - sum(p["D"] for p in pls) * 0.5
            for pl in pls:
                x0 = cx - pl["W"] * 0.5 + rng.uniform(-0.8, 0.8)
                fp = (x0, cur_y, x0 + pl["W"], cur_y + pl["D"])
                if not inside_fn(*fp) or _overlaps(fp, occ, 0.45):
                    ok = False
                    break
                cand_items.append((fp, ("rhouse", (x0, cur_y, pl))))
                cand_fps.append(fp)
                cur_y += pl["D"] + 0.4
        if ok:
            placed.extend(cand_items)
            occ.extend(cand_fps)

    # 2. Shuffled grid sweep packing gutted shells tightly into every corner
    gx_pts = np.arange(x_lim[0] + 6.0, x_lim[1] - 6.0, 14.0 * scale)
    gy_pts = np.arange(y_lim[0] + 6.0, y_lim[1] - 6.0, 14.0 * scale)
    cells = [(float(gx), float(gy)) for gx in gx_pts for gy in gy_pts]
    rng.shuffle(cells)
    for gx, gy in cells:
        if len(placed) >= target_n:
            break
        for sc_try in (scale * rng.uniform(0.90, 1.06), scale * rng.uniform(0.76, 0.88)):
            pl = plan_ruined_townhouse(rng, scale=sc_try,
                                       severity=np.clip(sev_base + rng.uniform(-0.22, 0.28), 0.22, 0.92))
            cx = gx + rng.uniform(-4.0, 4.0)
            cy = gy + rng.uniform(-4.0, 4.0)
            x0, y0 = cx - pl["W"] * 0.5, cy - pl["D"] * 0.5
            fp = (x0, y0, x0 + pl["W"], y0 + pl["D"])
            if inside_fn(*fp) and not _overlaps(fp, occ, 0.40):
                occ.append(fp)
                placed.append((fp, ("rhouse", (x0, y0, pl))))
                break

    # 3. Pocket-fill pass
    for _ in range(1500):
        if len(placed) >= target_n + 6:
            break
        pl = plan_ruined_townhouse(rng, scale=scale * rng.uniform(0.72, 0.88),
                                   severity=np.clip(sev_base + rng.uniform(-0.20, 0.30), 0.25, 0.92))
        cx = rng.uniform(*x_lim)
        cy = rng.uniform(*y_lim)
        x0, y0 = cx - pl["W"] * 0.5, cy - pl["D"] * 0.5
        fp = (x0, y0, x0 + pl["W"], y0 + pl["D"])
        if inside_fn(*fp) and not _overlaps(fp, occ, 0.35):
            occ.append(fp)
            placed.append((fp, ("rhouse", (x0, y0, pl))))

    return placed, occ


def _draw_ruined_city_interior(pen, P, rng, items):
    """Render all sorted ruined city structures, dead trees, rubble & ruined wells back-to-front."""
    for kind, args in sort_boxes(P, items):
        if kind == "rhouse":
            draw_ruined_house(pen, P, args[0], args[1], args[2], exterior=False)
        elif kind == "rchurch":
            draw_ruined_church(pen, P, args[0], args[1], args[2], exterior=False)
        elif kind == "rcitadel":
            _draw_ruined_inner_citadel(pen, P, rng, *args)
        elif kind == "rtower":
            ruined_tower(pen, P, rng, args[0], args[1], k=args[2], exterior=False)
        elif kind == "rwell":
            ruined_well(pen, P, *args)
        elif kind == "rfence":
            ruined_fence(pen, P, *args)
        elif kind == "rubble":
            pen.rubble(*P(args[0], args[1]), args[2], args[2] * 0.45)
        elif kind == "bush":
            pen.bush(*P(args[0], args[1]), args[2], outline=True)
        elif kind == "dead_tree":
            dh = args[2] if len(args) >= 3 else rng.uniform(15, 22)
            pen.dead_tree(*P(args[0], args[1]), dh, lean=rng.normal(0, 0.14))
        elif kind == "tuft":
            pen.tuft(*P(args[0], args[1]), rng.uniform(3.2, 5.2))


def _ruined_city_gatehouse(pen, P, rng, gx, y_front, gw, H, t, round_t=True, sev=0.45):
    """Battered twin-towered ruined city gatehouse with broken portcullis teeth & crumbled parapet."""
    gk = rng.uniform(0.46, 0.56)
    g_off = gw * 0.5 + 8.0 * gk / 0.55
    if round_t:
        ruined_tower(pen, P, rng, gx - g_off, y_front, k=gk * 0.82, exterior=False)
    else:
        _square_tower(pen, P, gx - g_off, y_front, gk, rng.uniform(0.25, 0.65))
    hA = H * rng.uniform(0.74, 0.92)
    arch = [P(gx - gw * 0.42, y_front - t * 0.4, 0), P(gx - gw * 0.42, y_front - t * 0.4, hA * 0.62),
            P(gx, y_front - t * 0.4, hA * 0.95), P(gx + gw * 0.42, y_front - t * 0.4, hA * 0.62),
            P(gx + gw * 0.42, y_front - t * 0.4, 0)]
    pen.blot(arch, jitter=0.1)
    for px in np.linspace(gx - gw * 0.30, gx + gw * 0.30, 4):
        if rng.random() < 0.78:
            p0 = P(px, y_front - t * 0.4, hA * 0.68)
            pen.cv.erase([(p0[0] - 0.5, p0[1]), (p0[0] + 0.5, p0[1]), (p0[0], p0[1] + 2.6)])
    if rng.random() < 0.68:
        box(pen, P, gx - gw * 0.5 - 2, y_front - t * 0.5, gx + gw * 0.5 + 2, y_front + t * 0.5, hA * 0.88, hA + 4.8)
    if round_t:
        ruined_tower(pen, P, rng, gx + g_off, y_front, k=gk * 0.82, exterior=False)
    else:
        _square_tower(pen, P, gx + g_off, y_front, gk, rng.uniform(0.25, 0.65))


def _draw_ruined_inner_citadel(pen, P, rng, cx0, cy0, cx1, cy1, H_in, t_in, round_t, sev, role="keep"):
    """Breached inner castle / citadel layer inside a ruined city: taller battered stone ramparts,
    4 corner turrets, broken inner barbican gatehouse, and a towering ruined keep or shattered abbey inside!"""
    cx_mid = (cx0 + cx1) * 0.5
    gw_in = 11.5

    def draw_irtower(X, Y):
        if round_t:
            if rng.random() < 0.35:
                _capped_turret(pen, P, rng, X, Y, 0.66)
            else:
                ruined_tower(pen, P, rng, X, Y, k=0.66, exterior=False)
        else:
            _square_tower(pen, P, X, Y, 0.58, rng.uniform(0.22, 0.62))

    # 1. Breached north & west inner ramparts + NW/NE turrets
    draw_irtower(cx0, cy1)
    curtain(pen, P, (cx0, cy1), (cx1, cy1), H_in, t_in, sev * 0.85)
    draw_irtower(cx1, cy1)
    curtain(pen, P, (cx0, cy0), (cx0, cy1), H_in, t_in, sev * 0.85)

    # 2. Ruined royal structures inside the inner citadel walls
    pad = t_in + 4.0
    ix0, iy0, ix1, iy1 = cx0 + pad, cy0 + pad, cx1 - pad, cy1 - pad
    inner_items, inner_occ = [], []

    if role == "cathedral":
        cpl = plan_ruined_church(rng, k=min(0.92, (ix1 - ix0) / 86.0))
        ccx = cx_mid - cpl["W"] * 0.42
        ccy = iy0 + max(2.0, (iy1 - iy0 - cpl["D"]) * 0.52)
        c_fp = (ccx - cpl["tw"] * 0.92, ccy - 2, ccx + cpl["W"] + 2, ccy + cpl["D"] + 2)
        inner_occ.append(c_fp)
        inner_items.append((c_fp, ("rchurch", (ccx, ccy, cpl))))
        for tx_c in (ix1 - 9.5, ix0 + 9.5):
            ty_c = iy1 - 9.5
            t_fp = (tx_c - 9, ty_c - 9, tx_c + 9, ty_c + 9)
            if not _overlaps(t_fp, inner_occ, 1.5):
                inner_occ.append(t_fp)
                inner_items.append((t_fp, ("rtower", (tx_c, ty_c, 0.62))))
                break
    else:
        kp = plan_ruined_house(rng, k=rng.uniform(1.16, 1.32), keep=True, severity=rng.uniform(0.22, 0.56))
        for w in kp["walls"]:
            kp["walls"][w] = crenellate(kp["walls"][w], kp["H"])
        kx = min(max(cx_mid - kp["W"] * 0.5 + rng.uniform(-8, 8), ix0 + 2.0), ix1 - kp["W"] - 2.0)
        ky = iy1 - kp["D"] - rng.uniform(2.0, max(2.5, (iy1 - iy0 - kp["D"]) * 0.38))
        k_fp = (kx - 2.5, ky - 2.5, kx + kp["W"] + 2.5, ky + kp["D"] + 2.5)
        inner_occ.append(k_fp)
        inner_items.append((k_fp, ("rhouse", (kx, ky, kp))))
        tx = (ix1 - 9.5) if (kx + kp["W"] * 0.5) <= cx_mid else (ix0 + 9.5)
        ty = iy1 - rng.uniform(10.0, max(10.5, (iy1 - iy0) * 0.45))
        t_fp = (tx - 9, ty - 9, tx + 9, ty + 9)
        if not _overlaps(t_fp, inner_occ, 1.5):
            inner_occ.append(t_fp)
            inner_items.append((t_fp, ("rtower", (tx, ty, 0.60))))

    for _ in range(3):
        rx, ry = rng.uniform(ix0 + 4, ix1 - 4), rng.uniform(iy0 + 4, iy1 - 4)
        rad = rng.uniform(4.5, 7.2)
        rfp = (rx - rad, ry - rad * 0.5, rx + rad, ry + rad * 0.5)
        if not _overlaps(rfp, inner_occ, 1.2):
            inner_occ.append(rfp)
            inner_items.append((rfp, ("rubble", (rx, ry, rad))))

    _draw_ruined_city_interior(pen, P, rng, inner_items)

    # 3. Breached east & south inner walls + battered SW/SE turrets + inner ruined barbican
    curtain(pen, P, (cx1, cy0), (cx1, cy1), H_in, t_in, sev * 0.85)
    draw_irtower(cx0, cy0)
    curtain(pen, P, (cx0, cy0), (cx_mid - gw_in * 0.5 - 4.5, cy0), H_in, t_in, sev * 0.85)
    _ruined_city_gatehouse(pen, P, rng, cx_mid, cy0, gw_in, H_in, t_in, round_t=round_t, sev=sev * 0.85)
    curtain(pen, P, (cx_mid + gw_in * 0.5 + 4.5, cy0), (cx1, cy0), H_in, t_in, sev * 0.85)
    draw_irtower(cx1, cy0)


def ruined_city(pen, P, rng):
    """Shattered ruined metropolis: ~22-30 densely clustered gutted townhouses, breached city walls, ruined keep or cathedral."""
    archetype = str(rng.choice(["walled_citadel", "cathedral_city", "hex_walled", "open_bastion"],
                               p=[0.32, 0.28, 0.24, 0.16]))
    H_w = rng.uniform(20, 25)
    t_w = rng.uniform(4.4, 5.4)
    sev = rng.uniform(0.32, 0.72)
    round_t = rng.random() < 0.68

    def draw_rtower(X, Y, sc=0.72, allow_cap=True):
        if round_t:
            if allow_cap and rng.random() < 0.28:
                _capped_turret(pen, P, rng, X, Y, sc)
            else:
                ruined_tower(pen, P, rng, X, Y, k=sc, exterior=False)
        else:
            _square_tower(pen, P, X, Y, sc * 0.88, rng.uniform(0.25, 0.72))

    def _add_ruin_clutter(items, occ, inside_fn, x_lim, y_lim):
        # Scatter 1-2 ruined wells, 3-6 dead trees & 7-12 rubble mounds strictly outside building boxes!
        for _ in range(2):
            for _ in range(18):
                wx, wy = rng.uniform(*x_lim), rng.uniform(*y_lim)
                wfp = (wx - 6.5, wy - 6.5, wx + 6.5, wy + 6.5)
                if inside_fn(*wfp) and not _overlaps(wfp, occ, 2.0):
                    occ.append(wfp)
                    items.append((wfp, ("rwell", (wx, wy))))
                    break
        for _ in range(int(rng.integers(3, 7))):
            for _ in range(25):
                tx, ty = rng.uniform(*x_lim), rng.uniform(*y_lim)
                tfp = (tx - 4.5, ty - 4.5, tx + 4.5, ty + 4.5)
                if inside_fn(*tfp) and not _overlaps(tfp, occ, 4.5):
                    occ.append(tfp)
                    items.append((tfp, ("dead_tree", (tx, ty, rng.uniform(14.0, 20.0)))))
                    break
        for _ in range(int(rng.integers(7, 13))):
            for _ in range(20):
                rx, ry = rng.uniform(*x_lim), rng.uniform(*y_lim)
                rad = rng.uniform(5.0, 8.5)
                rfp = (rx - rad, ry - rad * 0.5, rx + rad, ry + rad * 0.5)
                if inside_fn(*rfp) and not _overlaps(rfp, occ, 1.5):
                    occ.append(rfp)
                    items.append((rfp, ("rubble", (rx, ry, rad))))
                    break

    if archetype == "hex_walled":
        R = rng.uniform(158, 176)
        vs = [(R * math.cos(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3),
               R * math.sin(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3)) for k in range(6)]
        v0, v1, v2, v3, v4, v5 = vs
        mid = lambda a, b: ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        m43, m54, m32, m50, m21 = mid(v4, v3), mid(v5, v4), mid(v3, v2), mid(v5, v0), mid(v2, v1)
        gw = rng.uniform(15, 18)
        y_f = v0[1]

        h_excl = R * 0.866 + 3.6
        def excl_hex(ax, ay, bx, by):
            for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by), ((ax + bx) * 0.5, (ay + by) * 0.5)):
                if abs(py) < h_excl and (abs(px) * 0.866 + abs(py) * 0.5) < h_excl:
                    return True
            return False

        hex_towers = [(x, y, 10.8) for x, y in (v0, v1, v2, v3, v4, v5)] + [
            (x, y, 9.8) for x, y in (m43, m54, m32, m50, m21)
        ] + [(-gw * 0.5 - 6.0, y_f, 10.2), (gw * 0.5 + 6.0, y_f, 10.2)]
        hug_back = [
            (v4, v3, 0.0, 1.0),
            (v5, v4, -0.866, 0.5),
            (v3, v2, 0.866, 0.5),
        ]
        hug_front = [
            (v0, v1, 0.0, -1.0),
            (v1, v2, 0.866, -0.5),
            (v5, v0, -0.866, -0.5),
        ]

        roads, back_sub, front_sub = _plan_outer_suburbs(
            rng, excl_hex, (-R, y_f, R, -y_f), y_f, gw, ruined=True, hex_R=R,
            tower_pts=hex_towers, hug_back=hug_back, hug_front=hug_front,
        )
        for r_pts in roads:
            _ruined_track(pen, P, rng, r_pts, half_w=3.2)
        _draw_ruined_city_interior(pen, P, rng, back_sub)

        draw_rtower(*v4, sc=0.68)
        _curtain_seg(pen, P, v4, m43, H_w, t_w, sev, show_inner=True)
        draw_rtower(*m43, sc=0.62, allow_cap=False)
        _curtain_seg(pen, P, m43, v3, H_w, t_w, sev, show_inner=True)
        draw_rtower(*v3, sc=0.68)
        _curtain_seg(pen, P, m54, v4, H_w, t_w, sev, show_inner=True)
        draw_rtower(*m54, sc=0.62, allow_cap=False)
        _curtain_seg(pen, P, v5, m54, H_w, t_w, sev, show_inner=True)
        draw_rtower(*v5, sc=0.68)

        in_R = R * 0.866 - t_w - 2.5
        def inside_hex(x0, y0, x1, y1):
            for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                if abs(cy) > in_R - 1.5:
                    return False
                if abs(cx) * 0.866 + abs(cy) * 0.5 > in_R - 1.5:
                    return False
            return True

        items, occ = [], []
        # Breached Inner Citadel / Castle layer inside the hexagonal ruined city!
        H_in = H_w * rng.uniform(1.18, 1.32)
        t_in = t_w * 0.94
        cw_in, cd_in = rng.uniform(82, 94), rng.uniform(56, 64)
        cx_c, cy_c = rng.uniform(-16, 16), rng.uniform(22, 34)
        cx0, cy0, cx1, cy1 = cx_c - cw_in * 0.5, cy_c - cd_in * 0.5, cx_c + cw_in * 0.5, cy_c + cd_in * 0.5
        c_role = "keep" if rng.random() < 0.62 else "cathedral"
        cit_fp = (cx0 - 9.5, cy0 - 9.5, cx1 + 9.5, cy1 + 9.5)
        occ.append(cit_fp)
        occ.append((cx_c - 14.0, cy0 - 19.0, cx_c + 14.0, cy0 - 8.5))
        items.append((cit_fp, ("rcitadel", (cx0, cy0, cx1, cy1, H_in, t_in, round_t, sev, c_role))))

        # Secondary middle-city ruined church/tower outside the inner castle walls
        cpl = plan_ruined_church(rng, k=rng.uniform(0.76, 0.88))
        cx = rng.uniform(-86, -54) if cx_c > -4 else rng.uniform(44, 72)
        cy = rng.uniform(-34, 14)
        c_fp = (cx - cpl["tw"] * 0.92, cy - 3, cx + cpl["W"] + 3, cy + cpl["D"] + 3)
        if inside_hex(*c_fp) and not _overlaps(c_fp, occ, 2.5):
            occ.append(c_fp)
            items.append((c_fp, ("rchurch", (cx, cy, cpl))))

        th_items, occ = _pack_dense_ruined_buildings(
            rng, inside_hex, int(rng.integers(62, 76)), occ, scale=0.72,
            x_lim=(-in_R, in_R), y_lim=(-in_R, in_R), sev_base=sev,
        )
        items.extend(th_items)
        _add_ruin_clutter(items, occ, inside_hex, (-in_R, in_R), (-in_R, in_R))
        _draw_ruined_city_interior(pen, P, rng, items)

        _curtain_seg(pen, P, v3, m32, H_w, t_w, sev, show_inner=False)
        draw_rtower(*m32, sc=0.62, allow_cap=False)
        _curtain_seg(pen, P, m32, v2, H_w, t_w, sev, show_inner=False)
        draw_rtower(*v2, sc=0.68)
        _curtain_seg(pen, P, v5, m50, H_w, t_w, sev, show_inner=False)
        draw_rtower(*m50, sc=0.62, allow_cap=False)
        _curtain_seg(pen, P, m50, v0, H_w, t_w, sev, show_inner=False)
        draw_rtower(*v0, sc=0.68)
        _curtain_seg(pen, P, v2, m21, H_w, t_w, sev, show_inner=False)
        draw_rtower(*m21, sc=0.62, allow_cap=False)
        _curtain_seg(pen, P, m21, v1, H_w, t_w, sev, show_inner=False)
        curtain(pen, P, v0, (-gw * 0.5 - 5.0, y_f), H_w, t_w, sev)
        _ruined_city_gatehouse(pen, P, rng, 0.0, y_f, gw, H_w, t_w, round_t=round_t, sev=sev)
        curtain(pen, P, (gw * 0.5 + 5.0, y_f), v1, H_w, t_w, sev)
        draw_rtower(*v1, sc=0.68)
        _draw_ruined_city_interior(pen, P, rng, front_sub)
        return

    if archetype == "open_bastion":
        CW, CD = rng.uniform(268, 302), rng.uniform(198, 224)
        x0, y0, x1, y1 = -CW * 0.5, -CD * 0.5, CW * 0.5, CD * 0.5
        wf_x0, wf_x1 = -CW * 0.44, CW * 0.44
        y_arm = 22.0
        y_mid = (y0 + y_arm) * 0.5
        gw = rng.uniform(15, 18)

        def excl_ob(ax, ay, bx, by):
            return not (bx < wf_x0 - 3.6 or ax > wf_x1 + 3.6 or by < y0 - 3.6 or ay > y1 + 4.0)

        ms_l, ms_r = wf_x0 * 0.52, wf_x1 * 0.52
        ob_towers = [
            (wf_x0, y_arm, 10.5), (wf_x0, y_mid, 10.0), (wf_x0, y0, 11.2),
            (wf_x1, y_arm, 10.5), (wf_x1, y_mid, 10.0), (wf_x1, y0, 11.2),
            (ms_l, y0, 10.2), (ms_r, y0, 10.2),
            (-gw * 0.5 - 6.0, y0, 10.5), (gw * 0.5 + 6.0, y0, 10.5),
        ]
        hug_back = [((wf_x0, y0), (wf_x0, y_arm), -1.0, 0.0), ((wf_x0 + 18, y1), (wf_x1 - 18, y1), 0.0, 1.0)]
        hug_front = [((wf_x0, y0), (wf_x1, y0), 0.0, -1.0), ((wf_x1, y0), (wf_x1, y_arm), 1.0, 0.0)]

        roads, back_sub, front_sub = _plan_outer_suburbs(
            rng, excl_ob, (wf_x0, y0, wf_x1, y1), y0, gw, ruined=True,
            tower_pts=ob_towers, hug_back=hug_back, hug_front=hug_front,
        )
        for r_pts in roads:
            _ruined_track(pen, P, rng, r_pts, half_w=3.2)
        _draw_ruined_city_interior(pen, P, rng, back_sub)

        draw_rtower(wf_x0, y_arm, sc=0.66, allow_cap=False)
        curtain(pen, P, (wf_x0, y_mid), (wf_x0, y_arm), H_w, t_w, sev)
        draw_rtower(wf_x0, y_mid, sc=0.62, allow_cap=False)
        curtain(pen, P, (wf_x0, y0), (wf_x0, y_mid), H_w, t_w, sev)

        items, occ = [], []
        # Breached Inner Hilltop Castle / Citadel crowning the ruined open-bastion city!
        H_in = H_w * rng.uniform(1.18, 1.32)
        t_in = t_w * 0.94
        cw_in, cd_in = rng.uniform(86, 100), rng.uniform(58, 66)
        cx_c, cy_c = rng.uniform(-24, 24), rng.uniform(24, 38)
        cx0, cy0, cx1, cy1 = cx_c - cw_in * 0.5, cy_c - cd_in * 0.5, cx_c + cw_in * 0.5, cy_c + cd_in * 0.5
        cit_fp = (cx0 - 9.5, cy0 - 9.5, cx1 + 9.5, cy1 + 9.5)
        occ.append(cit_fp)
        occ.append((cx_c - 14.0, cy0 - 19.0, cx_c + 14.0, cy0 - 8.5))
        items.append((cit_fp, ("rcitadel", (cx0, cy0, cx1, cy1, H_in, t_in, round_t, sev, "keep"))))

        cpl = plan_ruined_church(rng, k=rng.uniform(0.78, 0.90))
        cx = rng.uniform(34, 68) if cx_c < 0 else rng.uniform(-86, -48)
        cy = rng.uniform(-34, 12)
        c_fp = (cx - cpl["tw"] * 0.92, cy - 3, cx + cpl["W"] + 3, cy + cpl["D"] + 3)
        if not _overlaps(c_fp, occ, 3.0) and c_fp[0] > wf_x0 + 8.0 and c_fp[2] < wf_x1 - 8.0:
            occ.append(c_fp)
            items.append((c_fp, ("rchurch", (cx, cy, cpl))))

        def inside_oval(ax, ay, bx, by):
            if ay < y0 + t_w + 4.0 or ax < wf_x0 + t_w + 3.0 or bx > wf_x1 - t_w - 3.0:
                return False
            for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by)):
                if (px / (CW * 0.48)) ** 2 + (py / (CD * 0.52)) ** 2 > 1.0:
                    return False
            return True

        th_items, occ = _pack_dense_ruined_buildings(
            rng, inside_oval, int(rng.integers(52, 66)), occ, scale=0.73,
            x_lim=(wf_x0 + t_w + 3, wf_x1 - t_w - 3), y_lim=(y0 + t_w + 4, y1), sev_base=sev,
        )
        items.extend(th_items)
        _add_ruin_clutter(items, occ, inside_oval, (wf_x0 + 8, wf_x1 - 8), (y0 + 10, y1 - 8))
        _draw_ruined_city_interior(pen, P, rng, items)

        draw_rtower(wf_x1, y_arm, sc=0.66, allow_cap=False)
        curtain(pen, P, (wf_x1, y_mid), (wf_x1, y_arm), H_w, t_w, sev)
        draw_rtower(wf_x1, y_mid, sc=0.62, allow_cap=False)
        curtain(pen, P, (wf_x1, y0), (wf_x1, y_mid), H_w, t_w, sev)

        ms_l, ms_r = wf_x0 * 0.52, wf_x1 * 0.52
        draw_rtower(wf_x0, y0, sc=0.72)
        curtain(pen, P, (wf_x0, y0), (ms_l, y0), H_w, t_w, sev)
        draw_rtower(ms_l, y0, sc=0.64, allow_cap=False)
        curtain(pen, P, (ms_l, y0), (-gw * 0.5 - 5.5, y0), H_w, t_w, sev)
        _ruined_city_gatehouse(pen, P, rng, 0.0, y0, gw, H_w, t_w, round_t=round_t, sev=sev)
        curtain(pen, P, (gw * 0.5 + 5.5, y0), (ms_r, y0), H_w, t_w, sev)
        draw_rtower(ms_r, y0, sc=0.64, allow_cap=False)
        curtain(pen, P, (ms_r, y0), (wf_x1, y0), H_w, t_w, sev)
        draw_rtower(wf_x1, y0, sc=0.72)
        _draw_ruined_city_interior(pen, P, rng, front_sub)
        return

    # Full 4-wall + multi-bastion breached ruined citadel OR cathedral metropolis (~2x area!) + ruined outer suburbs!
    CW = rng.uniform(262, 296)
    CD = rng.uniform(192, 218)
    x0, y0, x1, y1 = -CW * 0.5, -CD * 0.5, CW * 0.5, CD * 0.5
    gw = rng.uniform(15, 18)

    def excl_rect(ax, ay, bx, by):
        return not (bx < x0 - 3.6 or ax > x1 + 3.6 or by < y0 - 3.6 or ay > y1 + 3.6)

    nx1, nx2 = x0 + CW / 3.0, x0 + 2.0 * CW / 3.0
    sx_l, sx_r = x0 * 0.52, x1 * 0.52
    rect_towers = [
        (x0, y1, 11.2), (nx1, y1, 10.2), (nx2, y1, 10.2), (x1, y1, 11.2),
        (x0, 0.0, 10.2), (x1, 0.0, 10.2),
        (x0, y0, 11.2), (sx_l, y0, 10.2), (sx_r, y0, 10.2), (x1, y0, 11.2),
        (-gw * 0.5 - 6.0, y0, 10.5), (gw * 0.5 + 6.0, y0, 10.5),
    ]
    hug_back = [((x0, y0), (x0, y1), -1.0, 0.0), ((x0, y1), (x1, y1), 0.0, 1.0)]
    hug_front = [((x0, y0), (x1, y0), 0.0, -1.0), ((x1, y0), (x1, y1), 1.0, 0.0)]

    roads, back_sub, front_sub = _plan_outer_suburbs(
        rng, excl_rect, (x0, y0, x1, y1), y0, gw, ruined=True,
        tower_pts=rect_towers, hug_back=hug_back, hug_front=hug_front,
    )
    for r_pts in roads:
        _ruined_track(pen, P, rng, r_pts, half_w=3.2)
    _draw_ruined_city_interior(pen, P, rng, back_sub)

    draw_rtower(x0, y1, sc=0.70)
    curtain(pen, P, (x0, y1), (nx1, y1), H_w, t_w, sev)
    draw_rtower(nx1, y1, sc=0.62, allow_cap=False)
    curtain(pen, P, (nx1, y1), (nx2, y1), H_w, t_w, sev)
    draw_rtower(nx2, y1, sc=0.62, allow_cap=False)
    curtain(pen, P, (nx2, y1), (x1, y1), H_w, t_w, sev)
    draw_rtower(x1, y1, sc=0.70)
    curtain(pen, P, (x0, 0.0), (x0, y1), H_w, t_w, sev)
    draw_rtower(x0, 0.0, sc=0.64, allow_cap=False)
    curtain(pen, P, (x0, y0), (x0, 0.0), H_w, t_w, sev)

    items, occ = [], []
    # 3-tier ruined metropolis: Breached Inner Castle/Citadel + gutted middle-city ring!
    H_in = H_w * rng.uniform(1.18, 1.32)
    t_in = t_w * 0.94
    cw_in, cd_in = rng.uniform(88, 102), rng.uniform(58, 68)
    cx_c = rng.uniform(-34, 34)
    cy1 = rng.uniform(y1 - 36.0, y1 - 21.0)
    cy0 = cy1 - cd_in
    cx0, cx1 = cx_c - cw_in * 0.5, cx_c + cw_in * 0.5
    c_role = "cathedral" if archetype == "cathedral_city" else "keep"
    cit_fp = (cx0 - 9.5, cy0 - 9.5, cx1 + 9.5, cy1 + 9.5)
    occ.append(cit_fp)
    occ.append((cx_c - 14.0, cy0 - 19.0, cx_c + 14.0, cy0 - 8.5))
    items.append((cit_fp, ("rcitadel", (cx0, cy0, cx1, cy1, H_in, t_in, round_t, sev, c_role))))

    if archetype == "walled_citadel":
        cpl = plan_ruined_church(rng, k=rng.uniform(0.78, 0.90))
        cx = rng.uniform(42, 76) if cx_c < 0 else rng.uniform(-96, -54)
        cy = rng.uniform(-32, 24)
        c_fp = (cx - cpl["tw"] * 0.92, cy - 3, cx + cpl["W"] + 3, cy + cpl["D"] + 3)
        if not _overlaps(c_fp, occ, 2.5) and x0 + 10 < c_fp[0] and c_fp[2] < x1 - 10:
            occ.append(c_fp)
            items.append((c_fp, ("rchurch", (cx, cy, cpl))))
    else:
        for _ in range(4):
            tx = rng.uniform(x0 + 28, x1 - 28)
            ty = rng.uniform(y0 + 28, y1 - 24)
            t_fp = (tx - 11, ty - 11, tx + 11, ty + 11)
            if not _overlaps(t_fp, occ, 3.0):
                occ.append(t_fp)
                items.append((t_fp, ("rtower", (tx, ty, 0.64))))
                break

    pad_x = t_w + 4.0
    pad_y = t_w + 4.0
    gate_clr = (-12.0, y0, 12.0, y0 + 18.0)

    def inside_rect(ax, ay, bx, by):
        if ax < x0 + pad_x or bx > x1 - pad_x or ay < y0 + pad_y or by > y1 - pad_y:
            return False
        if not (bx < gate_clr[0] or ax > gate_clr[2] or by < gate_clr[1] or ay > gate_clr[3]):
            return False
        return True

    th_items, occ = _pack_dense_ruined_buildings(
        rng, inside_rect, int(rng.integers(54, 68)), occ, scale=0.72,
        x_lim=(x0 + pad_x, x1 - pad_x), y_lim=(y0 + pad_y, y1 - pad_y), sev_base=sev,
    )
    items.extend(th_items)
    _add_ruin_clutter(items, occ, inside_rect, (x0 + pad_x + 4, x1 - pad_x - 4), (y0 + pad_y + 4, y1 - pad_y - 4))
    _draw_ruined_city_interior(pen, P, rng, items)

    curtain(pen, P, (x1, 0.0), (x1, y1), H_w, t_w, sev)
    draw_rtower(x1, 0.0, sc=0.64, allow_cap=False)
    curtain(pen, P, (x1, y0), (x1, 0.0), H_w, t_w, sev)

    sx_l, sx_r = x0 * 0.52, x1 * 0.52
    draw_rtower(x0, y0, sc=0.70)
    curtain(pen, P, (x0, y0), (sx_l, y0), H_w, t_w, sev)
    draw_rtower(sx_l, y0, sc=0.62, allow_cap=False)
    curtain(pen, P, (sx_l, y0), (-gw * 0.5 - 5.5, y0), H_w, t_w, sev)
    _ruined_city_gatehouse(pen, P, rng, 0.0, y0, gw, H_w, t_w, round_t=round_t, sev=sev)
    curtain(pen, P, (gw * 0.5 + 5.5, y0), (sx_r, y0), H_w, t_w, sev)
    draw_rtower(sx_r, y0, sc=0.62, allow_cap=False)
    curtain(pen, P, (sx_r, y0), (x1, y0), H_w, t_w, sev)
    draw_rtower(x1, y0, sc=0.70)
    _draw_ruined_city_interior(pen, P, rng, front_sub)
